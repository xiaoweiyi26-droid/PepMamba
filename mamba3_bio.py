# Copyright (c) 2024-2026, Tri Dao, Albert Gu.
# Bio-adapted Mamba3: simplified for training, following BioMamba2Block design philosophy.
# The explicit B/C normalization and bias path avoids fused-kernel state inconsistencies.

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

# The custom causal-conv CUDA backward is not covered by PyTorch deterministic checks.
# Use the mathematically equivalent grouped torch Conv1d path for reproducible runs.
if os.environ.get("PEPMAMBA_DETERMINISTIC_KERNELS") == "1":
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class BioMamba3Block(nn.Module):
    """
    Bio-adapted Mamba3 block, following BioMamba2Block design philosophy.

    Differences vs original Mamba3:
    - No RoPE / rotary embeddings (removed angle_dt logic)
    - No trap gate
    - No MIMO support
    - No step / inference mode
    - Added causal conv1d (like BioMamba2Block)
    - A as learnable parameter (not input-dependent, for kernel compatibility)
    - Optional learnable initial SSM states
    - Training-oriented: clean in-proj / out-proj structure
    - Disabled memory-efficient path to ensure B_bias/C_bias stability
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=5,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        # SSM dynamics
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        # Architecture
        bias=False,
        conv_bias=True,
        dropout=0.0,
        chunk_size=64,
        use_mem_eff_path=True, # Retained for API compatibility; execution uses the explicit path.
        # Output projection
        norm_before_gate=False,
        # Retained for API compatibility.
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        del layer_idx
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if RMSNormGated is None:
            raise ImportError(
                "mamba_ssm.ops.triton.layernorm_gated.RMSNorm is required for BioMamba3Block. "
                "Please make sure the Triton-enabled mamba_ssm package is installed correctly."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.norm_before_gate = norm_before_gate

        # xBC = x + B + C packed together
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        # Causal conv1d over x + B + C
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -conv_init, conv_init)
        else:
            nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)

        # RMS Norm on B and C (Mamba3 style)
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        # Learnable initial SSM states
        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )

        # A: learnable log-space parameter (per head)
        _A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(_A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        # dt_bias parameterization
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # B and C biases (Mamba3 style)
        self.B_bias = nn.Parameter(
            torch.zeros((self.nheads, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            torch.zeros((self.nheads, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Output norm and projection
        self.norm = RMSNormGated(
            self.d_inner,
            eps=1e-5,
            norm_before_gate=norm_before_gate,
            group_size=self.headdim,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u, seq_idx=None, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("BioMamba3Block does not support inference_params (step mode).")
        batch, seqlen, _ = u.shape

        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else {"dt_limit": self.dt_limit}
        initial_states = (
            repeat(self.init_states, "... -> b ...", b=batch)
            if self.learnable_init_states
            else None
        )

        # Use the non-fused path to preserve B/C normalization and bias semantics.
        z, xBC, dd_dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dd_dt + self.dt_bias)

        # Causal conv1d
        if causal_conv1d_fn is None:
            xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))[:, :seqlen, :]
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # Apply RMSNorm on B and C (Mamba3 style)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Add B/C biases: kernel broadcasts G -> H internally
        B = B + self.B_bias
        C = C + self.C_bias

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            B,
            C,
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            dt_softplus=True,
            **dt_limit_kwargs,
        )
        y = self.norm(rearrange(y, "b l h p -> b l (h p)"), z)
        out = self.out_proj(y)

        return self.dropout(out)