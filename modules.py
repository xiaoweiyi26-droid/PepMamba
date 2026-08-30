# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba3_bio import BioMamba3Block
except ImportError as exc:
    raise ImportError(
        "PepMamba requires the project-local mamba3_bio.py and a Mamba3-compatible mamba_ssm installation."
    ) from exc


def reverse_sequence(x, mask=None):
    """
    Safely reverse a sequence.
    If mask is provided (True = Padding), it only reverses the valid tokens
    and keeps padding at the end of the sequence.
    """
    if mask is None:
        return torch.flip(x, dims=[1])

    seqlen = x.size(1)
    # Number of valid tokens per batch
    lengths = (~mask).sum(dim=1)

    # Create indices for sequence reversal
    idx = torch.arange(seqlen, device=x.device).unsqueeze(0).expand(x.size(0), -1)
    rev_idx = lengths.unsqueeze(1) - 1 - idx

    # Preserve the original index for padding positions.
    rev_idx = torch.where(rev_idx < 0, idx, rev_idx)

    # Gather the reversed sequence
    return x.gather(1, rev_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class ChannelAttention1D(nn.Module):
    """Light channel reweighting on sequence features."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return x * self.sigmoid(y)


class LocalPatternBlock(nn.Module):
    """Extract short-range sequence patterns before long-range modeling."""

    def __init__(self, d_model, kernel_size=5, dropout=0.2):
        super().__init__()
        self.channel_gate = ChannelAttention1D(d_model)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.channel_gate(x)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class SequenceFFN(nn.Module):
    """Light feed-forward refinement after interaction."""

    def __init__(self, d_model, d_inner, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.depthwise_conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class BidirectionalMambaBlock(nn.Module):
    """Run Mamba3 in forward and backward directions (padding-safe) and merge."""

    def __init__(self, d_model, d_state=128, dropout=0.2, **mamba_kwargs):
        super().__init__()
        self.fwd_mamba = BioMamba3Block(d_model=d_model, d_state=d_state, dropout=dropout, **mamba_kwargs)
        self.bwd_mamba = BioMamba3Block(d_model=d_model, d_state=d_state, dropout=dropout, **mamba_kwargs)

    def forward(self, x, mask=None):
        y_fwd = self.fwd_mamba(x)

        # Reverse valid positions while retaining padding at the tail.
        x_rev = reverse_sequence(x, mask)
        y_bwd_rev = self.bwd_mamba(x_rev)
        y_bwd = reverse_sequence(y_bwd_rev, mask)

        return y_fwd + y_bwd


class TwoWayMambaBlock(nn.Module):
    """
    Protein-peptide interaction block.
    1) learn local patterns
    2) model long-range context with bidirectional Mamba
    3) update protein with peptide context and peptide with protein context
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.3):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.protein_local = LocalPatternBlock(d_model, kernel_size=5, dropout=dropout)
        self.peptide_local = LocalPatternBlock(d_model, kernel_size=3, dropout=dropout)
        self.protein_mamba = BidirectionalMambaBlock(d_model, d_state=128, dropout=dropout)
        self.peptide_mamba = BidirectionalMambaBlock(d_model, d_state=128, dropout=dropout)

        self.protein_query = nn.Linear(d_model, n_head * d_k)
        self.peptide_key = nn.Linear(d_model, n_head * d_k)
        self.peptide_value = nn.Linear(d_model, n_head * d_v)
        self.protein_out = nn.Linear(n_head * d_v, d_model)
        self.protein_gate = nn.Linear(d_model * 2, d_model)

        self.peptide_query = nn.Linear(d_model, n_head * d_k)
        self.protein_key = nn.Linear(d_model, n_head * d_k)
        self.protein_value = nn.Linear(d_model, n_head * d_v)
        self.peptide_out = nn.Linear(n_head * d_v, d_model)
        self.peptide_gate = nn.Linear(d_model * 2, d_model)

        self.protein_res_scale = nn.Parameter(torch.full((1,), 0.1))
        self.peptide_res_scale = nn.Parameter(torch.full((1,), 0.1))

        self.dropout = nn.Dropout(dropout)
        self.protein_norm = nn.LayerNorm(d_model)
        self.peptide_norm = nn.LayerNorm(d_model)

    def _cross_attention(self, q, k, v, key_mask=None, query_mask=None):
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) / math.sqrt(self.d_k)

        if key_mask is not None:
            # A large negative mask value prevents padded positions from receiving fp32 attention mass.
            scores = scores.masked_fill(key_mask[:, None, None, :], -1e9)

        attn = F.softmax(scores, dim=-1)

        if key_mask is not None:
            valid_keys = (~key_mask).to(attn.dtype)[:, None, None, :]
            attn = attn * valid_keys
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if query_mask is not None:
            valid_queries = (~query_mask).to(attn.dtype)[:, None, :, None]
            attn = attn * valid_queries

        attn = self.dropout(attn)
        fused = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        fused = rearrange(fused, "b h l d -> b l (h d)")
        return fused, attn

    def forward(self, protein_feat, peptide_feat, protein_mask=None, peptide_mask=None):
        """Fuse protein and peptide features with separate padding masks.

        ``protein_mask`` and ``peptide_mask`` are Boolean tensors where
        ``True`` denotes a padded position.
        """
        protein_local = self.protein_local(protein_feat)
        peptide_local = self.peptide_local(peptide_feat)

        # Propagate padding masks to the bidirectional sequence reversal.
        protein_context = self.protein_mamba(protein_local, mask=protein_mask) + protein_local
        peptide_context = self.peptide_mamba(peptide_local, mask=peptide_mask) + peptide_local

        q_protein = rearrange(self.protein_query(protein_context), "b l (h d) -> b h l d", h=self.n_head)
        k_peptide = rearrange(self.peptide_key(peptide_context), "b l (h d) -> b h l d", h=self.n_head)
        v_peptide = rearrange(self.peptide_value(peptide_context), "b l (h d) -> b h l d", h=self.n_head)

        fused_protein, attn_probs = self._cross_attention(
            q=q_protein,
            k=k_peptide,
            v=v_peptide,
            key_mask=peptide_mask,
            query_mask=protein_mask,
        )
        fused_protein = self.protein_out(fused_protein)

        q_peptide = rearrange(self.peptide_query(peptide_context), "b l (h d) -> b h l d", h=self.n_head)
        k_protein = rearrange(self.protein_key(protein_context), "b l (h d) -> b h l d", h=self.n_head)
        v_protein = rearrange(self.protein_value(protein_context), "b l (h d) -> b h l d", h=self.n_head)

        fused_peptide, _ = self._cross_attention(
            q=q_peptide,
            k=k_protein,
            v=v_protein,
            key_mask=protein_mask,
            query_mask=peptide_mask,
        )
        fused_peptide = self.peptide_out(fused_peptide)

        protein_gate = torch.sigmoid(self.protein_gate(torch.cat([protein_context, fused_protein], dim=-1)))
        peptide_gate = torch.sigmoid(self.peptide_gate(torch.cat([peptide_context, fused_peptide], dim=-1)))

        protein_final = (fused_protein * protein_gate) + (self.protein_res_scale * protein_context)
        peptide_final = (fused_peptide * peptide_gate) + (self.peptide_res_scale * peptide_context)

        protein_out = self.protein_norm(protein_final + protein_feat)
        peptide_out = self.peptide_norm(peptide_final + peptide_feat)

        if protein_mask is not None:
            protein_out = protein_out.masked_fill(protein_mask.unsqueeze(-1), 0.0)
        if peptide_mask is not None:
            peptide_out = peptide_out.masked_fill(peptide_mask.unsqueeze(-1), 0.0)

        return protein_out, peptide_out, attn_probs, None


# Legacy aliases retained for compatibility with existing scripts.
BioMamba2Block = BioMamba3Block
ECA1D = ChannelAttention1D
LocalMotifExtractor = LocalPatternBlock
ConvFFN = SequenceFFN
BidirectionalMamba2 = BidirectionalMambaBlock
ReciprocalMambaFusionBlock = TwoWayMambaBlock