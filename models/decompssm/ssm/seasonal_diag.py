import math

import torch
from einops import rearrange, repeat
from torch import nn

from models.decompssm.functional.period import estimate_topk_freqs_stft
from models.decompssm.ssm.base import SSM


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class DiagonalSeasonalSSM(SSM):
    """
    Seasonal SSM with diagonal/block-rotation A encoded implicitly via kernels.
    Each head h has K seasonal components with parameters:
      - rho in (rho_min, rho_max)
      - omega estimated from input (STFT) + learnable delta
      - phase in R
      - amplitude in [-amp_scale, amp_scale]
      - softmax mixing across K (initialized from STFT weights via EMA)
    get_kernel(u) returns [H, L] kernel built as sum_k amp_k * rho_k^t * cos(omega_k t + phase_k).
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
        top_k: int = 3,
        rho_min: float = 0.90,
        rho_max: float = 0.995,
        amp_scale: float = 0.2,
        freq_momentum: float = 0.2,
        max_delta_omega: float = 0.02,
        # Weight smoothing
        weight_smooth: float = 0.4,
        # Disable learnable tweaks by default to improve generalization
        use_delta: bool = False,
        use_phase: bool = False,
    ) -> None:
        self.top_k = top_k
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.amp_scale = amp_scale
        self.freq_momentum = freq_momentum
        self.max_delta_omega = max_delta_omega
        self.weight_smooth = weight_smooth
        self.use_delta = use_delta
        self.use_phase = use_phase
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

    def init_weights(self) -> None:
        super().init_weights()
        H, K = self.n_kernels, self.top_k
        # Parameters per head and component
        self.rho_logits = nn.Parameter(torch.full((H, K), 2.0))
        self.delta_omega = nn.Parameter(torch.zeros(H, K))
        self.phase = nn.Parameter(torch.zeros(H, K))
        # small non-zero amp init to break symmetry
        self.amp_logits = nn.Parameter(0.1 * torch.randn(H, K))
        self.mix_logits = nn.Parameter(torch.zeros(H, K))
        self.log_gain = nn.Parameter(torch.zeros(H))
        # STFT EMA buffers
        self.register_buffer("omega_ema", torch.zeros(H, K))
        self.register_buffer("w_ema", torch.ones(H, K) / max(1, K))
        self.register_buffer("ema_initialized", torch.zeros(1, dtype=torch.bool))

    def _bounded_rho(self) -> torch.Tensor:
        sig = torch.sigmoid(self.rho_logits)
        return self.rho_min + (self.rho_max - self.rho_min) * sig

    def _smooth_weights(self, weights: torch.Tensor) -> torch.Tensor:
        # Blend with uniform to reduce peaky mixes
        if self.weight_smooth <= 0:
            return weights
        K = weights.shape[1]
        uniform = torch.full_like(weights, 1.0 / max(1, K))
        return (1 - self.weight_smooth) * weights + self.weight_smooth * uniform

    def _estimate_omega(self, u_bdl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # u_bdl: [B, D, L]
        B, D, L = u_bdl.shape
        u = u_bdl.permute(0, 2, 1).contiguous()  # [B, L, D]
        # Detrend: remove per-series mean over time to focus on seasonal
        u = u - u.mean(dim=1, keepdim=True)
        n_fft = min(256, max(32, _next_pow2(L)))
        win_length = n_fft
        hop_length = max(8, n_fft // 4)
        top_idx, weights, _ = estimate_topk_freqs_stft(u, top_k=self.top_k, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        omega = 2 * math.pi * top_idx.to(u.dtype) / L  # [D, K]
        if self.n_kernels == D:
            pass
        elif self.n_kernels < D:
            omega = omega[: self.n_kernels]
            weights = weights[: self.n_kernels]
        else:
            reps = (self.n_kernels + D - 1) // D
            omega = omega.repeat(reps, 1)[: self.n_kernels]
            weights = weights.repeat(reps, 1)[: self.n_kernels]
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
        # smooth weights
        weights = self._smooth_weights(weights)
        return omega, weights

    def _bounded_amp(self) -> torch.Tensor:
        return self.amp_scale * torch.tanh(self.amp_logits)

    def _l2_normalize_kernel(self, k: torch.Tensor) -> torch.Tensor:
        # k: [H, L]
        # Enforce zero-mean seasonal kernel per head
        k = k - k.mean(dim=-1, keepdim=True)
        kn = k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return k / kn

    def _build_kernel(self, u_bdl: torch.Tensor) -> torch.Tensor:
        B, D, L = u_bdl.shape
        device, dtype = u_bdl.device, u_bdl.dtype
        H, K = self.n_kernels, self.top_k
        t = torch.arange(L, device=device, dtype=dtype)
        omega_base, weights = self._estimate_omega(u_bdl)  # [H, K]
        if self.use_delta:
            omega = omega_base + self.max_delta_omega * torch.tanh(self.delta_omega).to(dtype)
        else:
            omega = omega_base
        rho = self._bounded_rho().to(dtype)
        phi = (self.phase if self.use_phase else torch.zeros_like(self.phase)).to(dtype)
        amp = self._bounded_amp().to(dtype)
        mix = torch.softmax(torch.log(weights + 1e-6) + self.mix_logits.to(dtype), dim=1)
        gain = torch.exp(self.log_gain).to(dtype).unsqueeze(-1)
        phase = omega.unsqueeze(-1) * t + phi.unsqueeze(-1)
        decay = torch.exp(torch.log(rho + 1e-8).unsqueeze(-1) * t)
        components = amp.unsqueeze(-1) * decay * torch.cos(phase)
        k = (mix.unsqueeze(-1) * components).sum(dim=1)  # [H, L]
        k = self._l2_normalize_kernel(k)
        return (gain * k).contiguous()

    def _build_kernel_with_fixed_omega(
        self,
        L_out: int,
        omega: torch.Tensor,
        weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = torch.arange(L_out, device=device, dtype=dtype)
        rho = self._bounded_rho().to(dtype)
        phi = (self.phase if self.use_phase else torch.zeros_like(self.phase)).to(dtype)
        amp = self._bounded_amp().to(dtype)
        # smooth weights
        weights = self._smooth_weights(weights)
        mix = torch.softmax(torch.log(weights + 1e-6) + self.mix_logits.to(dtype), dim=1)
        gain = torch.exp(self.log_gain).to(dtype).unsqueeze(-1)
        phase = omega.unsqueeze(-1) * t + phi.unsqueeze(-1)
        decay = torch.exp(torch.log(rho + 1e-8).unsqueeze(-1) * t)
        components = amp.unsqueeze(-1) * decay * torch.cos(phase)
        k = (mix.unsqueeze(-1) * components).sum(dim=1)  # [H, L_out]
        k = self._l2_normalize_kernel(k)
        return (gain * k).contiguous()

    def get_kernel(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, D, L]
        return self._build_kernel(u)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return super().forward(u)


class ClosedLoopDiagonalSeasonalSSM(DiagonalSeasonalSSM):
    """
    Closed-loop forecasting by zero-padding inputs and convolving with a horizon-length kernel,
    then taking the last horizon samples. This avoids modifying companion code while enabling
    multi-step prediction with seasonal diagonal A.
    """

    def __init__(self, lag: int = 1, horizon: int = 1, **kwargs) -> None:
        self.lag = lag
        self.horizon = horizon
        self.closed_loop = True
        self.inference_only = False
        super().__init__(**kwargs)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # u: [B, L, D]
        u_bdl = rearrange(u, "b l d -> b d l")
        B, D, L = u_bdl.shape
        device, dtype = u_bdl.device, u_bdl.dtype

        omega_base, weights = self._estimate_omega(u_bdl)  # [H, K]
        if self.use_delta:
            omega = omega_base + self.max_delta_omega * torch.tanh(self.delta_omega).to(dtype)
        else:
            omega = omega_base

        k_long = self._build_kernel_with_fixed_omega(L + self.horizon, omega, weights, device, dtype)  # [H, L+H]
        k_long = repeat(k_long, "nk kd -> (kr nk nh hd) kd", kr=self.kernel_repeat, nh=self.n_heads, hd=self.head_dim)

        pad = torch.zeros(B, D, self.horizon, device=device, dtype=dtype)
        u_pad = torch.cat([u_bdl, pad], dim=-1)  # [B, D, L+H]

        y_full = self.fft_conv(u_pad, k_long)  # [B, D, L+H]
        y_h = y_full[:, :, -self.horizon :]  # [B, D, H]
        y = rearrange(y_h, "b d l -> b l d")

        y_u = None if self.inference_only else y
        return y, y_u
