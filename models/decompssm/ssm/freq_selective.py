import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from models.decompssm.functional.period import estimate_topk_freqs_stft
from models.decompssm.ssm.base import SSM


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class FrequencySelectiveSSM(SSM):
    """
    Seasonal SSM with frequency-selective pre-filtering followed by selective scan.
    - Estimate top-k seasonal frequencies per head via STFT with EMA smoothing
    - Build soft bandpass masks around those frequencies in the rFFT domain
    - Filter the sequence and run a selective scan over the filtered signal
    This biases the model towards seasonal bands while retaining selective dynamics.
    """

    def __init__(
        self,
        model_dim: int,
        n_kernels: int,
        kernel_dim: int,
        kernel_repeat: int,
        n_heads: int | None = None,
        head_dim: int = 1,
        kernel_weights: torch.Tensor | None = None,
        kernel_init: str = "normal",
        kernel_train: bool = True,
        skip_connection: bool = False,
        seed: int = 42,
        # Frequency-selective params
        top_k: int = 5,
        band_width_frac: float = 0.03,  # fraction of Nyquist for each band half-width
        freq_momentum: float = 0.2,
        # Selective scan params
        d_state: int | None = None,
        dt_rank: int | str = "auto",
        dt_scale: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        sel_dropout_p: float = 0.1,
    ) -> None:
        if n_heads is None:
            n_heads = 1
        # Initialize attributes needed by init_weights BEFORE calling super().__init__
        self.top_k = top_k
        self.band_width_frac = band_width_frac
        self.freq_momentum = freq_momentum
        # Use provided args (not yet set on self by base) to compute dependent dims
        computed_d_state = d_state if d_state is not None else max(4, min(8, kernel_dim))
        self.d_state = int(computed_d_state)
        self.dt_rank = math.ceil(model_dim / 32) if dt_rank == "auto" else int(dt_rank)
        self.dt_scale = dt_scale
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Call base init (this will invoke init_weights which relies on fields above)
        super().__init__(
            model_dim=model_dim,
            n_kernels=n_kernels,
            kernel_dim=kernel_dim,
            kernel_repeat=kernel_repeat,
            n_heads=n_heads,
            head_dim=head_dim,
            kernel_weights=kernel_weights,
            kernel_init=kernel_init,
            kernel_train=kernel_train,
            skip_connection=skip_connection,
            seed=seed,
        )

        # Define modules that don't need to exist before init_weights
        self.sel_dropout = nn.Dropout(p=sel_dropout_p)

    def init_weights(self) -> None:
        super().init_weights()
        H = self.n_kernels
        S = self.d_state
        # Parameters for selective scan
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
        # STFT EMA buffers
        self.register_buffer("omega_ema", torch.zeros(H, self.top_k))
        self.register_buffer("w_ema", torch.ones(H, self.top_k) / max(1, self.top_k))
        self.register_buffer("ema_initialized", torch.zeros(1, dtype=torch.bool))

    def _estimate_freqs(self, u_bdl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # u_bdl: [B, D, L]
        B, D, L = u_bdl.shape
        u = u_bdl.permute(0, 2, 1).contiguous()  # [B, L, D]
        # Detrend
        u = u - u.mean(dim=1, keepdim=True)
        n_fft = min(256, max(32, _next_pow2(L)))
        win_length = n_fft
        hop_length = max(8, n_fft // 4)
        top_idx, weights, _ = estimate_topk_freqs_stft(u, top_k=self.top_k, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        omega = 2 * math.pi * top_idx.to(u.dtype) / L  # [D, K]
        # Align to H=n_kernels
        if self.n_kernels == D:
            pass
        elif self.n_kernels < D:
            omega = omega[: self.n_kernels]
            weights = weights[: self.n_kernels]
        else:
            reps = (self.n_kernels + D - 1) // D
            omega = omega.repeat(reps, 1)[: self.n_kernels]
            weights = weights.repeat(reps, 1)[: self.n_kernels]
        # EMA smoothing
        if self.training and self.freq_momentum > 0:
            if not self.ema_initialized.item():
                self.omega_ema.copy_(omega.detach())
                self.w_ema.copy_(weights.detach())
                self.ema_initialized.fill_(True)
            else:
                m = self.freq_momentum
                self.omega_ema.mul_(1 - m).add_(omega.detach(), alpha=m)
                self.w_ema.mul_(1 - m).add_(weights.detach(), alpha=m)
            omega = self.omega_ema.to(u.dtype)
            weights = (self.w_ema / (self.w_ema.sum(dim=1, keepdim=True) + 1e-8)).to(u.dtype)
        return omega, weights

    def _build_masks(self, L_out: int, omega: torch.Tensor, weights: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Return masks_DF: [D, F] where F = L_out//2 + 1
        H, K = omega.shape
        F = L_out // 2 + 1
        # Convert omega to rFFT bin indices
        center = torch.round(omega * L_out / (2 * math.pi)).long().clamp(min=0, max=F - 1)  # [H, K]
        width = max(1, int(self.band_width_frac * F))
        masks = torch.zeros(H, F, device=device, dtype=dtype)
        for h in range(H):
            for k in range(K):
                c = int(center[h, k].item())
                lo = max(0, c - width)
                hi = min(F - 1, c + width)
                masks[h, lo : hi + 1] = torch.maximum(masks[h, lo : hi + 1], weights[h, k].expand(hi - lo + 1))
        # Normalize per-head mask to [0,1]
        m_max = masks.amax(dim=1, keepdim=True).clamp(min=1e-6)
        masks = masks / m_max
        return masks  # [H, F]

    def _apply_freq_filter(self, u_bdl: torch.Tensor, L_out: int, omega: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # u_bdl: [B, D, L], returns filtered [B, D, L_out]
        B, D, L = u_bdl.shape
        device, dtype = u_bdl.device, u_bdl.dtype
        U = torch.fft.rfft(u_bdl, n=L_out, dim=-1)  # [B, D, F]
        masks_HF = self._build_masks(L_out, omega, weights, device, dtype)  # [H, F]
        # Align masks to D
        if self.n_kernels == D:
            masks_DF = masks_HF
        elif self.n_kernels < D:
            reps = (D + self.n_kernels - 1) // self.n_kernels
            masks_DF = masks_HF.repeat(reps, 1)[:D]
        else:
            masks_DF = masks_HF[:D]
        U_filt = U * masks_DF.unsqueeze(0)  # broadcast over batch
        u_filt = torch.fft.irfft(U_filt, n=L_out, dim=-1)
        return u_filt

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, L, D]
        B, L, D = u.shape
        xz = self.x_proj(self.sel_dropout(u))
        delta_part, BC_part = xz.split([self.dt_rank, 2 * self.d_state], dim=-1)
        B_part, C_part = BC_part.split([self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_part))  # [B, L, D]
        u_scan = rearrange(u, "b l d -> b d l")
        delta_scan = rearrange(delta, "b l d -> b d l")
        B_scan = rearrange(B_part, "b l s -> b s l")
        C_scan = rearrange(C_part, "b l s -> b s l")
        A = -torch.exp(self.A_log.float())  # [H, S]
        if self.n_kernels < D:
            reps = (D + self.n_kernels - 1) // self.n_kernels
            A = A.repeat(reps, 1)[:D]
        elif self.n_kernels > D:
            A = A[:D]
        y = selective_scan_fn(u_scan, delta_scan, A, B_scan, C_scan, self.D.float(), z=None, delta_bias=None, delta_softplus=False)
        y = rearrange(y, "b d l -> b l d")
        return y

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, L, D]
        B, L, D = u.shape
        u_bdl = rearrange(u, "b l d -> b d l")
        omega, weights = self._estimate_freqs(u_bdl)  # [H, K], [H, K]
        u_filt = self._apply_freq_filter(u_bdl, L, omega, weights)  # [B, D, L]
        u_t = rearrange(u_filt, "b d l -> b l d")
        y = self._selective_scan(u_t)
        if self.skip_connection:
            y = y + u
        return y


class ClosedLoopFrequencySelectiveSSM(FrequencySelectiveSSM):
    def __init__(self, lag: int = 1, horizon: int = 1, **kwargs: Any) -> None:
        self.lag = lag
        self.horizon = horizon
        self.closed_loop = True
        self.inference_only = False
        super().__init__(**kwargs)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # u: [B, L, D]
        B, L, D = u.shape
        device, dtype = u.device, u.dtype
        # Estimate freqs on context window
        u_bdl = rearrange(u, "b l d -> b d l")
        omega, weights = self._estimate_freqs(u_bdl)
        # Pad and filter
        pad = torch.zeros(B, D, self.horizon, device=device, dtype=dtype)
        u_pad = torch.cat([u_bdl, pad], dim=-1)  # [B, D, L+H]
        u_filt = self._apply_freq_filter(u_pad, L + self.horizon, omega, weights)  # [B, D, L+H]
        u_filt_t = rearrange(u_filt, "b d l -> b l d")
        # Selective scan and take horizon
        y_full = self._selective_scan(u_filt_t)  # [B, L+H, D]
        y_h = y_full[:, -self.horizon :, :]  # [B, H, D]
        y_u = None if self.inference_only else y_h
        return y_h, y_u
