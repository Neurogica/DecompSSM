import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from models.decompssm.functional.period import estimate_topk_freqs_stft
from models.decompssm.ssm.base import SSM


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class RotationalSeasonalSSM(SSM):
    """
    Seasonal SSM using rotation matrices in the A matrix to capture periodic patterns.
    Instead of implicit kernels, we directly construct A as block-diagonal rotation matrices
    corresponding to different frequencies, leveraging mamba-ssm's selective_scan_fn.
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
        # Seasonal params
        top_k: int = 5,
        rho_min: float = 0.90,
        rho_max: float = 0.995,
        freq_momentum: float = 0.2,
        max_delta_omega: float = 0.02,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        # Disable learnable tweaks by default to improve generalization
        use_delta: bool = False,
    ) -> None:
        self.top_k = top_k
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.freq_momentum = freq_momentum
        self.max_delta_omega = max_delta_omega
        self.use_delta = use_delta
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale

        if n_heads is None:
            n_heads = 1

        # State dimension for rotation matrices (2 per frequency component)
        self.d_state = 2 * top_k
        self.dt_rank = math.ceil(model_dim / 16) if dt_rank == "auto" else dt_rank

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

    def init_weights(self) -> None:
        super().init_weights()
        H, K = self.n_kernels, self.top_k

        # Parameters for rotation matrices
        self.rho_logits = nn.Parameter(torch.full((H, K), 2.0))
        self.delta_omega = nn.Parameter(torch.zeros(H, K))

        # Projection for input-dependent B and C
        self.x_proj = nn.Linear(self.model_dim, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.model_dim, bias=True)

        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias
        dt = torch.exp(torch.rand(self.model_dim) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.model_dim))

        # STFT EMA buffers
        self.register_buffer("omega_ema", torch.zeros(H, K))
        self.register_buffer("ema_initialized", torch.zeros(1, dtype=torch.bool))

    def _bounded_rho(self) -> torch.Tensor:
        sig = torch.sigmoid(self.rho_logits)
        return self.rho_min + (self.rho_max - self.rho_min) * sig

    def _estimate_omega(self, u_bdl: torch.Tensor) -> torch.Tensor:
        B, D, L = u_bdl.shape
        u = u_bdl.permute(0, 2, 1).contiguous()  # [B, L, D]

        # Detrend
        u = u - u.mean(dim=1, keepdim=True)

        n_fft = min(256, max(32, _next_pow2(L)))
        win_length = n_fft
        hop_length = max(8, n_fft // 4)

        top_idx, _, _ = estimate_topk_freqs_stft(u, top_k=self.top_k, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        omega = 2 * math.pi * top_idx.to(u.dtype) / L  # [D, K]

        # Handle dimension mismatch
        if self.n_kernels == D:
            pass
        elif self.n_kernels < D:
            omega = omega[: self.n_kernels]
        else:
            reps = (self.n_kernels + D - 1) // D
            omega = omega.repeat(reps, 1)[: self.n_kernels]

        # EMA update during training
        if self.training and self.freq_momentum > 0:
            if not self.ema_initialized.item():
                self.omega_ema.copy_(omega.detach())
                self.ema_initialized.fill_(True)
            else:
                m = self.freq_momentum
                self.omega_ema.mul_(1 - m).add_(omega.detach(), alpha=m)
            omega = self.omega_ema.to(u.dtype)

        return omega

    def _construct_rotation_A(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Construct block-diagonal rotation matrices as A.
        Each 2x2 block represents a rotation for one frequency component.
        """
        H, K = omega.shape
        device, dtype = omega.device, omega.dtype

        # Get decay rates
        rho = self._bounded_rho().to(dtype)  # [H, K]

        # Apply learnable frequency adjustments if enabled
        if self.use_delta:
            omega = omega + self.max_delta_omega * torch.tanh(self.delta_omega).to(dtype)

        # Construct A as block-diagonal rotation matrices
        # Each 2x2 block: [[rho*cos(omega), -rho*sin(omega)], [rho*sin(omega), rho*cos(omega)]]
        A_blocks = []
        for k in range(K):
            cos_omega = torch.cos(omega[:, k])  # [H]
            sin_omega = torch.sin(omega[:, k])  # [H]
            rho_k = rho[:, k]  # [H]

            # Create 2x2 rotation blocks
            block = torch.stack(
                [torch.stack([rho_k * cos_omega, -rho_k * sin_omega], dim=-1), torch.stack([rho_k * sin_omega, rho_k * cos_omega], dim=-1)], dim=-2
            )  # [H, 2, 2]

            A_blocks.append(block)

        # Stack all blocks diagonally
        A = torch.zeros(H, self.d_state, self.d_state, device=device, dtype=dtype)
        for k in range(K):
            A[:, 2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = A_blocks[k]

        # Convert to log space for numerical stability (as done in Mamba)
        # We need to handle the rotation matrix eigenvalues carefully
        # For a rotation matrix with decay rho, the eigenvalues are rho * exp(±i*omega)
        # We'll use the negative log of the magnitude (rho) as the effective A
        A_log = -torch.log(A.abs().clamp(min=1e-8))

        return A_log

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, L, D]
        B, L, D = u.shape
        device, dtype = u.device, u.dtype

        # Rearrange for convolution: [B, D, L]
        u_bdl = rearrange(u, "b l d -> b d l")

        # Estimate frequencies
        omega = self._estimate_omega(u_bdl)  # [H, K]

        # Construct rotation-based A matrix
        A_log = self._construct_rotation_A(omega)  # [H, d_state, d_state]

        # For selective_scan_fn, we need A to be [d_inner, d_state]
        # We'll repeat A_log for each dimension if needed
        if self.n_kernels < D:
            # Repeat the A matrices to match dimension
            reps = (D + self.n_kernels - 1) // self.n_kernels
            A_log = repeat(A_log, "h s1 s2 -> (rep h) s1 s2", rep=reps)[:D]
        elif self.n_kernels > D:
            A_log = A_log[:D]

        # Project input to get dt, B, C
        xz = self.x_proj(u)  # [B, L, dt_rank + 2*d_state]
        delta, BC = xz.split([self.dt_rank, 2 * self.d_state], dim=-1)
        B, C = BC.split([self.d_state, self.d_state], dim=-1)

        # Apply dt projection
        delta = F.softplus(self.dt_proj(delta))  # [B, L, D]

        # Rearrange for selective scan
        u_scan = rearrange(u, "b l d -> b d l")
        delta_scan = rearrange(delta, "b l d -> b d l")
        B_scan = rearrange(B, "b l s -> b s l")
        C_scan = rearrange(C, "b l s -> b s l")

        # For rotation matrices, we need to handle A differently
        # Since selective_scan expects diagonal A, we'll use the eigenvalues
        # Extract diagonal elements as an approximation
        A_diag = torch.diagonal(A_log, dim1=-2, dim2=-1)  # [D, d_state]
        A = -torch.exp(A_diag.float())

        # Apply selective scan
        y = selective_scan_fn(u_scan, delta_scan, A, B_scan, C_scan, self.D.float(), z=None, delta_bias=None, delta_softplus=False)

        # Rearrange back
        y = rearrange(y, "b d l -> b l d")

        if self.skip_connection:
            y = y + u

        return y


class ClosedLoopRotationalSeasonalSSM(RotationalSeasonalSSM):
    """
    Closed-loop forecasting version of RotationalSeasonalSSM.
    """

    def __init__(self, lag: int = 1, horizon: int = 1, **kwargs) -> None:
        self.lag = lag
        self.horizon = horizon
        self.closed_loop = True
        self.inference_only = False
        super().__init__(**kwargs)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # u: [B, L, D]
        B, L, D = u.shape
        device, dtype = u.device, u.dtype

        # Pad input for horizon
        pad = torch.zeros(B, self.horizon, D, device=device, dtype=dtype)
        u_pad = torch.cat([u, pad], dim=1)  # [B, L+H, D]

        # Apply parent forward
        y_full = super().forward(u_pad)  # [B, L+H, D]

        # Extract horizon predictions
        y_h = y_full[:, -self.horizon :, :]  # [B, H, D]

        y_u = None if self.inference_only else y_h
        return y_h, y_u
