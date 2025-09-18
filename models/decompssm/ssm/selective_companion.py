import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from models.decompssm.ssm.closed_loop.companion import ClosedLoopCompanionSSM
from models.decompssm.ssm.companion import CompanionSSM


class SelectiveCompanionSSM(CompanionSSM):
    """
    Open-loop Companion SSM with an additional selective-scan residual path.
    - Returns only y (for encoder Block compatibility)
    - Keeps original companion performance and adds selective residual with a learnable gate
    """

    def __init__(
        self,
        d_state: int | None = None,
        dt_rank: int | str = "auto",
        dt_scale: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        sel_dropout_p: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # Ensure required args for CompanionSSM are present
        kwargs.setdefault("norm_order", 1)
        super().__init__(**kwargs)
        # Reduce selective path capacity to mitigate overfitting
        self.d_state = d_state if d_state is not None else max(4, min(8, self.kernel_dim))
        self.dt_rank = math.ceil(self.model_dim / 32) if dt_rank == "auto" else int(dt_rank)
        self.dt_scale = dt_scale
        self.dt_min = dt_min
        self.dt_max = dt_max
        # Gate initialized slightly towards companion but softer to avoid over-confident mixing
        self.gate_logits_enc = nn.Parameter(torch.full((self.model_dim,), 1.0))
        # Dropout regularization for selective path
        self.sel_dropout_enc = nn.Dropout(p=sel_dropout_p)
        self._init_selective_params_open()

    def _init_selective_params_open(self) -> None:
        H = self.n_kernels
        S = self.d_state
        self.A_log_enc = nn.Parameter(torch.full((H, S), math.log(0.5)))
        self.x_proj_enc = nn.Linear(self.model_dim, self.dt_rank + 2 * S, bias=False)
        self.dt_proj_enc = nn.Linear(self.dt_rank, self.model_dim, bias=True)
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        nn.init.uniform_(self.dt_proj_enc.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.model_dim) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_enc.bias.copy_(inv_dt)
        self.D_enc = nn.Parameter(torch.ones(self.model_dim))

    def _selective_scan_seq(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, L, D] -> selective scan over same length
        B, L, D = u.shape
        u_dp = self.sel_dropout_enc(u)
        xz = self.x_proj_enc(u_dp)
        delta_part, BC_part = xz.split([self.dt_rank, 2 * self.d_state], dim=-1)
        B_part, C_part = BC_part.split([self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj_enc(delta_part))
        u_scan = rearrange(u, "b l d -> b d l")
        delta_scan = rearrange(delta, "b l d -> b d l")
        B_scan = rearrange(B_part, "b l s -> b s l")
        C_scan = rearrange(C_part, "b l s -> b s l")
        A = -torch.exp(self.A_log_enc.float())  # [H, S]
        if self.n_kernels < D:
            reps = (D + self.n_kernels - 1) // self.n_kernels
            A = A.repeat(reps, 1)[:D]
        elif self.n_kernels > D:
            A = A[:D]
        y = selective_scan_fn(u_scan, delta_scan, A, B_scan, C_scan, self.D_enc.float(), z=None, delta_bias=None, delta_softplus=False)
        y = rearrange(y, "b d l -> b l d")
        return cast(torch.Tensor, self.sel_dropout_enc(y))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # Base companion open-loop
        y_comp = super().forward(u)
        # Selective residual
        y_sel = self._selective_scan_seq(u)
        # Mix
        gate = torch.sigmoid(self.gate_logits_enc).view(1, 1, -1)
        y = gate * y_comp + (1.0 - gate) * y_sel
        return cast(torch.Tensor, y)


class ClosedLoopSelectiveCompanionSSM(ClosedLoopCompanionSSM):
    def __init__(
        self,
        lag: int = 1,
        horizon: int = 1,
        d_state: int | None = None,
        dt_rank: int | str = "auto",
        dt_scale: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        sel_dropout_p: float = 0.1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("norm_order", 1)
        kwargs.pop("top_k", None)
        super().__init__(lag=lag, horizon=horizon, **kwargs)
        # Reduce selective path capacity to mitigate overfitting
        self.d_state = d_state if d_state is not None else max(4, min(8, self.kernel_dim))
        self.dt_rank = math.ceil(self.model_dim / 32) if dt_rank == "auto" else int(dt_rank)
        self.dt_scale = dt_scale
        self.dt_min = dt_min
        self.dt_max = dt_max
        # Soften gate init
        self.gate_logits = nn.Parameter(torch.full((self.model_dim,), 1.0))
        self.sel_dropout = nn.Dropout(p=sel_dropout_p)
        self._init_selective_params()

    def _init_selective_params(self) -> None:
        H = self.n_kernels
        S = self.d_state
        self.A_log = nn.Parameter(torch.full((H, S), math.log(0.5)))
        self.x_proj = nn.Linear(self.model_dim, self.dt_rank + 2 * S, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.model_dim, bias=True)
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.model_dim) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.D = nn.Parameter(torch.ones(self.model_dim))

    def _selective_scan_horizon(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.shape
        device, dtype = u.device, u.dtype
        pad = torch.zeros(B, self.horizon, D, device=device, dtype=dtype)
        u_pad = torch.cat([u, pad], dim=1)
        u_pad = self.sel_dropout(u_pad)
        xz = self.x_proj(u_pad)
        delta_part, BC_part = xz.split([self.dt_rank, 2 * self.d_state], dim=-1)
        B_part, C_part = BC_part.split([self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_part))
        u_scan = rearrange(u_pad, "b l d -> b d l")
        delta_scan = rearrange(delta, "b l d -> b d l")
        B_scan = rearrange(B_part, "b l s -> b s l")
        C_scan = rearrange(C_part, "b l s -> b s l")
        A = -torch.exp(self.A_log.float())
        if self.n_kernels < D:
            reps = (D + self.n_kernels - 1) // self.n_kernels
            A = A.repeat(reps, 1)[:D]
        elif self.n_kernels > D:
            A = A[:D]
        y_pad = selective_scan_fn(u_scan, delta_scan, A, B_scan, C_scan, self.D.float(), z=None, delta_bias=None, delta_softplus=False)
        y_pad = rearrange(y_pad, "b d l -> b l d")
        return cast(torch.Tensor, self.sel_dropout(y_pad[:, -self.horizon :, :]))

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        y_comp, y_u = super().forward(u)
        y_sel = self._selective_scan_horizon(u)
        gate = torch.sigmoid(self.gate_logits).view(1, 1, -1)
        y = gate * y_comp + (1.0 - gate) * y_sel
        return y, y_u
