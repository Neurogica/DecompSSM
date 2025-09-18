import math
from typing import Any, cast

import opt_einsum as oe
import torch
from einops import rearrange, repeat
from torch import nn

from models.decompssm.functional.krylov import krylov
from models.decompssm.functional.period import estimate_topk_freqs_stft
from models.decompssm.ssm.base import SSM


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class SeasonalSSM(SSM):
    """
    Seasonal SSM using companion matrix structure with frequency-aware poles.
    Combines companion matrix efficiency with seasonal decomposition.
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
        # Seasonal-specific
        top_k: int = 3,
        freq_momentum: float = 0.2,
        norm_order: int = 2,
    ) -> None:
        self.top_k = top_k
        self.freq_momentum = freq_momentum
        self.norm_order = norm_order

        # Ensure n_heads is not None for SSM base class
        if n_heads is None:
            n_heads = 1

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

    def init_kernel_weights(self, kernel_init: str) -> torch.Tensor:
        if kernel_init == "normal":
            kernel = torch.randn(self.n_kernels, self.kernel_dim)
        elif kernel_init == "xavier":
            stdv = 1.0 / math.sqrt(self.kernel_dim)
            kernel = torch.FloatTensor(self.n_kernels, self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel

    def init_weights(self) -> None:
        super().init_weights()

        # Frequency estimation buffers
        self.register_buffer("omega_ema", torch.zeros(self.n_kernels, self.top_k))
        self.register_buffer("w_ema", torch.ones(self.n_kernels, self.top_k) / max(1, self.top_k))
        self.register_buffer("ema_initialized", torch.zeros(1, dtype=torch.bool))

        # Companion matrix structure
        self._fp = (self.n_kernels, self.kernel_dim)
        self.shift_matrix = nn.Parameter(torch.zeros(self.n_kernels, self.kernel_dim, self.kernel_dim), requires_grad=False)
        self.shift_matrix.data[:, 1:, :-1] = torch.eye(self.kernel_dim - 1)
        self.p_padding = nn.Parameter(torch.zeros(*self._fp), requires_grad=False)
        self.p_padding.data[:, -1] = 1.0

        # Companion matrix coefficients (last column)
        a = self.init_kernel_weights(self.kernel_init)
        self.register("a", a, trainable=True, lr=None, wd=None)

        # B and C matrices for SSM
        b = self.init_kernel_weights(self.kernel_init)
        self.register("b", b, trainable=True, lr=None, wd=None)

        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)

        # Seasonal modulation parameters
        self.seasonal_mix = nn.Parameter(torch.zeros(self.n_kernels, self.top_k))
        self.frequency_shift = nn.Parameter(torch.zeros(self.n_kernels, self.top_k))

    def norm(self, x: torch.Tensor, ord: int = 1) -> torch.Tensor:
        if ord <= 0:
            return x
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        x = x / (x_norm + 1e-8)
        return x

    def _estimate_frequencies(self, u_bdl: torch.Tensor) -> torch.Tensor:
        B, D, L = u_bdl.shape
        u = u_bdl.permute(0, 2, 1).contiguous()

        # Adaptive FFT size
        n_fft = min(256, max(32, _next_pow2(L)))
        win_length = n_fft
        hop_length = max(8, n_fft // 4)

        top_idx, weights, _ = estimate_topk_freqs_stft(u, top_k=self.top_k, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # Map to n_kernels
        if self.n_kernels == D:
            omega = 2 * math.pi * top_idx.to(u.dtype) / L
        else:
            omega = 2 * math.pi * top_idx.to(u.dtype) / L
            if self.n_kernels < D:
                omega = omega[: self.n_kernels]
                weights = weights[: self.n_kernels]
            else:
                omega = omega[0:1].repeat(self.n_kernels, 1)
                weights = weights[0:1].repeat(self.n_kernels, 1)

        # EMA update during training
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

        return omega

    def _modulate_companion_coeffs(self, a: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Modulate companion matrix coefficients with seasonal frequencies"""
        # omega: [H, K]
        # a: [H, D]
        H, D = a.shape
        K = omega.shape[1]

        # Apply frequency shift
        omega_shifted = omega + 0.1 * torch.tanh(self.frequency_shift)

        # Create seasonal modulation pattern
        # For each frequency, create a modulation across the state dimension
        t = torch.linspace(0, 1, D, device=a.device, dtype=a.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        seasonal_pattern = torch.cos(omega_shifted.unsqueeze(-1) * t * 2 * math.pi)  # [H, K, D]

        # Mix seasonal patterns
        mix_weights = torch.softmax(self.seasonal_mix, dim=1).unsqueeze(-1)  # [H, K, 1]
        seasonal_modulation = (seasonal_pattern * mix_weights).sum(dim=1)  # [H, D]

        # Combine with original coefficients
        a_modulated = a + 0.1 * seasonal_modulation

        return a_modulated

    def matrix_power(self, l: int, c: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Krylov sequence using companion matrix"""
        A = self.shift_matrix + oe.contract("h i, h j -> h j i", self.p_padding, p)
        g = krylov(l, A, b, c)
        return g

    def get_kernel(self, u: torch.Tensor) -> torch.Tensor:
        """Override base class method - note it passes u, not u_bdl"""
        l = u.shape[-1]

        # Normalize parameters
        a = self.norm(self.a, ord=self.norm_order) if self.norm_order > 0 else self.a
        c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c

        # Estimate frequencies and modulate companion coefficients
        omega = self._estimate_frequencies(u)
        a_modulated = self._modulate_companion_coeffs(a, omega)

        # Generate kernel via Krylov
        f = self.matrix_power(l, c, self.b, a_modulated).to(u.device)

        return f

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().forward(u))


class ClosedLoopSeasonalSSM(SeasonalSSM):
    """
    Closed-loop seasonal SSM for multi-step forecasting.
    """

    def __init__(self, lag: int = 1, horizon: int = 1, **kwargs: Any) -> None:
        self.lag = lag
        self.horizon = horizon
        self.closed_loop = True
        self.inference_only = False
        kwargs["skip_connection"] = False
        super().__init__(**kwargs)

    def init_weights(self) -> None:
        super().init_weights()
        # K matrix for closed-loop control
        k = self.init_kernel_weights(self.kernel_init)
        self.register("k", k, trainable=True, lr=None, wd=None)

    def get_companion_matrix(self, p: torch.Tensor) -> torch.Tensor:
        return self.shift_matrix + oe.contract("h i, h j -> h j i", self.p_padding, p)

    def fft_conv_d(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        L = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2 * L, dim=2)
        v_f = torch.fft.rfft(v, n=2 * L, dim=2)
        y_f = oe.contract("b h l, h d l -> b h l d", u_f, v_f)
        y = torch.fft.irfft(y_f, n=2 * L, dim=2)[:, :, :L, :]
        return y

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        u = rearrange(u, "b l d -> b d l")
        b_size, d, l = u.shape
        l_horizon = self.horizon

        # Normalize parameters
        a = self.norm(self.a, ord=self.norm_order) if self.norm_order > 0 else self.a

        # Estimate frequencies and modulate
        omega = self._estimate_frequencies(u)
        a_modulated = self._modulate_companion_coeffs(a, omega)

        # Get companion matrix
        A = self.get_companion_matrix(a_modulated)

        if self.closed_loop:
            # Compute hidden state x_lag
            k_x = krylov(l, A, self.b, c=None).to(u.device)
            x = self.fft_conv_d(u, k_x)  # [B, H, L, D]

            # Compute A + BK matrix
            b = self.norm(self.b, ord=self.norm_order) if self.norm_order > 0 else self.b
            k = self.norm(self.k, ord=self.norm_order) if self.norm_order > 0 else self.k
            A_BK = A + oe.contract("h i, h j -> h i j", b, k)

            # Rollout for horizon steps
            x_horizon = krylov(l_horizon, A_BK, x[:, :, -1, :], c=None)

            # Compute predictions
            c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c
            y = torch.einsum("...nl, ...n -> ...l", x_horizon, c).contiguous()
            y = rearrange(y, "b d l -> b l d")

            # Compute next input for closed-loop training
            if not self.inference_only and self.closed_loop:
                u_next = torch.einsum("...nl, ...n -> ...l", x_horizon, self.k).contiguous()
                u_next = rearrange(u_next, "b d l -> b l d")
            else:
                u_next = None

            return y, u_next
        else:
            # Open-loop forecast
            b = self.norm(self.b, ord=self.norm_order) if self.norm_order > 0 else self.b
            c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c
            k = krylov(l, A, b, c).to(u.device)
            k = repeat(k, "nk kd -> (kr nk nh hd) kd", kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
            y = rearrange(self.fft_conv(u, k), "b d l -> b l d")

            if not self.inference_only:
                _k = self.norm(self.k, ord=self.norm_order)
                k_u = krylov(l, A, b, _k).to(u.device)
                k_u = repeat(k_u, "nk kd -> (kr nk nh hd) kd", kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)
                y_u = rearrange(self.fft_conv(u, k_u), "b d l -> b l d")
            else:
                y_u = None

            return y, y_u
