# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import SequenceFFN, TwoWayMambaBlock


class ConvPositionEncoding(nn.Module):
    """Add depthwise-convolutional position encoding to ``(batch, length, dim)`` features."""

    def __init__(self, d_model):
        super().__init__()
        self.pe_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )

    def forward(self, x):
        """Return position-enhanced features with the same shape as ``x``."""
        pe = self.pe_conv(x.transpose(1, 2)).transpose(1, 2)
        return x + pe


class PepMambaLayer(nn.Module):
    """Apply two-way protein-peptide interaction followed by stream-specific refinement."""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.3):
        super().__init__()
        self.interaction = TwoWayMambaBlock(n_head, d_model, d_k, d_v, dropout=dropout)
        self.protein_ffn = SequenceFFN(d_model, d_inner, dropout=dropout)
        self.peptide_ffn = SequenceFFN(d_model, d_inner, dropout=dropout)

    def forward(self, protein_feat, peptide_feat, protein_mask=None, peptide_mask=None):
        """Process both streams; mask values are ``True`` at padded positions."""
        protein_feat, peptide_feat, attn, _ = self.interaction(
            protein_feat, peptide_feat, protein_mask=protein_mask, peptide_mask=peptide_mask
        )
        protein_feat = self.protein_ffn(protein_feat)
        peptide_feat = self.peptide_ffn(peptide_feat)
        return protein_feat, peptide_feat, attn, None


class PepMambaEncoder(nn.Module):
    """Project protein and peptide ESM features and stack interaction layers."""

    def __init__(self, n_layers, d_model, n_head, d_k, d_v, d_inner, dropout=0.3):
        super().__init__()
        esm_feat_dim = 1280
        self.protein_proj = nn.Linear(esm_feat_dim, d_model)
        self.protein_skip_proj = nn.Linear(esm_feat_dim, d_model)
        self.peptide_proj = nn.Linear(esm_feat_dim, d_model)

        self.protein_pos = ConvPositionEncoding(d_model)
        self.peptide_pos = ConvPositionEncoding(d_model)

        self.layers = nn.ModuleList(
            [PepMambaLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, peptide_sequence, protein_sequence, class_mask=None):
        """Encode both sequences; ``class_mask`` is the legacy protein padding mask name."""
        # Build the peptide padding mask from padded ESM features before projection.
        peptide_pad_mask = peptide_sequence.abs().sum(dim=-1).eq(0)

        peptide_hidden = self.peptide_proj(peptide_sequence)
        protein_hidden = self.protein_proj(protein_sequence)
        protein_skip = self.protein_skip_proj(protein_sequence)

        peptide_hidden = self.peptide_pos(peptide_hidden)
        protein_hidden = self.protein_pos(protein_hidden)

        attention_maps = []
        for layer in self.layers:
            # Pass protein and peptide padding masks separately to each layer.
            protein_hidden, peptide_hidden, attn, _ = layer(
                protein_hidden, peptide_hidden, protein_mask=class_mask, peptide_mask=peptide_pad_mask
            )
            attention_maps.append(attn)

        return protein_hidden, peptide_hidden, attention_maps, protein_skip, peptide_pad_mask


class ResiduePredictionHead(nn.Module):
    """Predict per-residue logits from raw and interaction-refined protein streams."""

    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.recalibrator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=3, dilation=3)

        self.scale_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.norm_ms = nn.LayerNorm(d_model)

        # Lightweight Transformer classifier head.
        # Preserve classifier attention for downstream visualization.
        self.cls_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.cls_norm1 = nn.LayerNorm(d_model)
        self.cls_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.cls_norm2 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 2)

    def forward(
        self,
        protein_skip,
        protein_hidden,
        peptide_hidden=None,
        peptide_pad_mask=None,
        protein_pad_mask=None,
    ):
        """Return ``(batch, protein_length, 2)`` logits and classifier attention.

        ``peptide_hidden`` and ``peptide_pad_mask`` are retained for backward
        compatibility and are unused by the current two-stream protein head.
        ``protein_pad_mask`` is ``True`` at padded protein positions.
        """
        # Fuse the original and interaction-refined protein streams.
        x_concat = torch.cat([protein_skip, protein_hidden], dim=-1)
        x_base = self.recalibrator(x_concat)

        x_t = x_base.transpose(1, 2)
        weights = F.softmax(self.scale_weights, dim=0)
        x_ms = (
            weights[0] * self.conv3(x_t)
            + weights[1] * self.conv5(x_t)
            + weights[2] * self.conv7(x_t)
        ).transpose(1, 2)

        x_ms = self.norm_ms(x_ms + x_base)

        attn_out, cls_attn = self.cls_attn(
            query=x_ms,
            key=x_ms,
            value=x_ms,
            key_padding_mask=protein_pad_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x_cls = self.cls_norm1(x_ms + self.drop(attn_out))

        ffn_out = self.cls_ffn(x_cls)
        x_cls = self.cls_norm2(x_cls + self.drop(ffn_out))

        if protein_pad_mask is not None:
            x_cls = x_cls.masked_fill(protein_pad_mask.unsqueeze(-1), 0.0)

        logits = self.out(self.drop(x_cls))
        return logits, cls_attn


class PepMamba(nn.Module):
    """Predict protein-residue classes and return auxiliary attention maps."""

    def __init__(self, n_layers, d_model, n_head, d_k, d_v, d_inner, dropout=0.3):
        super().__init__()
        self.encoder = PepMambaEncoder(n_layers, d_model, n_head, d_k, d_v, d_inner, dropout)
        self.head = ResiduePredictionHead(d_model, dropout=dropout)

    def forward(self, peptide_sequence, protein_sequence, class_mask=None):
        """Return residue logits and attention maps.

        ``class_mask`` is retained for backward compatibility and denotes the
        protein padding mask, with ``True`` at invalid positions.
        """
        protein_hidden, peptide_hidden, attention_maps, protein_skip, peptide_pad_mask = self.encoder(
            peptide_sequence, protein_sequence, class_mask=class_mask
        )
        logits, cls_attn = self.head(
            protein_skip=protein_skip,
            protein_hidden=protein_hidden,
            peptide_hidden=peptide_hidden,
            peptide_pad_mask=peptide_pad_mask,
            protein_pad_mask=class_mask,
        )

        # Store classifier attention separately because legacy modules may return a tensor.

        attention_maps.append({"cls_transformer_attn": cls_attn})

        return logits, attention_maps


# Legacy aliases retained for backward compatibility.
ConvPositionalEncoding = ConvPositionEncoding
MultiModalLayer = PepMambaLayer
RepeatedModule = PepMambaEncoder
EfficientMultiScaleContextualHead = ResiduePredictionHead
