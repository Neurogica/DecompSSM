import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from models.decompssm.functional.period import estimate_topk_freqs_stft
from models.decompssm.ssm.base import SSM


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class SelectiveSSMBlock(nn.Module):
    """
    A selective SSM block for processing specific time series components.
    Uses mamba_ssm's selective scan for state-dependent processing.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int | str = "auto",
        dt_scale: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout_p: float = 0.1,
        component_type: Literal["trend", "seasonal", "residual"] = "trend",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.component_type = component_type
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Selective scan parameters
        self.A_log = nn.Parameter(torch.full((d_model, d_state), math.log(0.5)))
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.D = nn.Parameter(torch.ones(d_model))

        # Component-specific initialization
        self._init_component_params(dt_scale, dt_min, dt_max)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)

        # Component-specific processing layers
        if component_type == "trend":
            # Trend components benefit from lower frequency emphasis
            self.trend_emphasis = nn.Parameter(torch.ones(d_model) * 0.1)
        elif component_type == "seasonal":
            # Seasonal components need frequency-aware processing
            self.freq_mix = nn.Linear(d_model, d_model)
        elif component_type == "residual":
            # Residual components focus on high-frequency noise
            self.noise_gate = nn.Linear(d_model, d_model)

    def _init_component_params(self, dt_scale: float, dt_min: float, dt_max: float) -> None:
        """Initialize parameters based on component type"""
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Component-specific dt initialization
        if self.component_type == "trend":
            # Trend: slower dynamics (larger dt)
            dt_range = (dt_min * 2, dt_max * 2)
        elif self.component_type == "seasonal":
            # Seasonal: medium dynamics
            dt_range = (dt_min, dt_max)
        else:  # residual
            # Residual: faster dynamics (smaller dt)
            dt_range = (dt_min * 0.5, dt_max * 0.5)

        dt = torch.exp(torch.rand(self.d_model) * (math.log(dt_range[1]) - math.log(dt_range[0])) + math.log(dt_range[0])).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
        Returns:
            [B, L, D] processed tensor
        """
        B, L, D = x.shape

        # Apply component-specific pre-processing
        if self.component_type == "trend":
            # Emphasize low-frequency components for trend
            x = x * torch.sigmoid(self.trend_emphasis).view(1, 1, -1)
        elif self.component_type == "seasonal":
            # Apply frequency mixing for seasonal patterns
            x = self.freq_mix(x)
        elif self.component_type == "residual":
            # Gate high-frequency noise for residual
            x = x * torch.sigmoid(self.noise_gate(x))

        # Selective scan processing
        x_dropout = self.dropout(x)
        xz = self.x_proj(x_dropout)
        delta_part, BC_part = xz.split([self.dt_rank, 2 * self.d_state], dim=-1)
        B_part, C_part = BC_part.split([self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta_part))

        # Rearrange for selective scan
        u_scan = rearrange(x, "b l d -> b d l")
        delta_scan = rearrange(delta, "b l d -> b d l")
        B_scan = rearrange(B_part, "b l s -> b s l")
        C_scan = rearrange(C_part, "b l s -> b s l")

        A = -torch.exp(self.A_log.float())

        # Selective scan
        y = selective_scan_fn(u_scan, delta_scan, A, B_scan, C_scan, self.D.float(), z=None, delta_bias=None, delta_softplus=False)

        y = rearrange(y, "b d l -> b l d")
        return self.dropout(y)  # type: ignore


class DeepSSMDecomposition(SSM):
    """
    Deep SSM-based time series decomposition model that separates time series into
    trend, seasonal, and residual components using specialized selective SSM blocks.

    Combines multiple decomposition techniques:
    - Trend extraction using low-frequency selective SSM
    - Seasonal pattern modeling with frequency-aware selective SSM
    - Residual modeling with high-frequency selective SSM
    - Ensemble fusion of decomposed components
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
        # Decomposition-specific parameters
        d_state: int = 16,
        dt_rank: int | str = "auto",
        n_decomp_layers: int = 3,
        decomp_dropout: float = 0.1,
        top_k_freqs: int = 5,
        freq_momentum: float = 0.2,
        fusion_method: Literal["learned", "attention", "gate"] = "attention",
        # STL-like parameters
        seasonal_smoother: float = 0.5,
        trend_smoother: float = 0.7,
    ) -> None:
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

        self.d_state = d_state
        self.dt_rank = dt_rank
        self.n_decomp_layers = n_decomp_layers
        self.top_k_freqs = top_k_freqs
        self.freq_momentum = freq_momentum
        self.fusion_method = fusion_method
        self.seasonal_smoother = seasonal_smoother
        self.trend_smoother = trend_smoother

        # Initialize decomposition modules
        self._init_decomposition_modules(decomp_dropout)

        # Frequency estimation buffers
        self.register_buffer("omega_ema", torch.zeros(n_kernels, top_k_freqs))
        self.register_buffer("freq_weights_ema", torch.ones(n_kernels, top_k_freqs) / max(1, top_k_freqs))
        self.register_buffer("freq_ema_initialized", torch.zeros(1, dtype=torch.bool))

    def _init_decomposition_modules(self, dropout_p: float) -> None:
        """Initialize the three decomposition pathways"""
        # Trend extraction pathway (multiple layers for better smoothing)
        self.trend_blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(d_model=self.model_dim, d_state=self.d_state, dt_rank=self.dt_rank, dropout_p=dropout_p, component_type="trend")
                for _ in range(self.n_decomp_layers)
            ]
        )

        # Seasonal pattern extraction pathway
        self.seasonal_blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(d_model=self.model_dim, d_state=self.d_state, dt_rank=self.dt_rank, dropout_p=dropout_p, component_type="seasonal")
                for _ in range(self.n_decomp_layers)
            ]
        )

        # Residual modeling pathway
        self.residual_blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(d_model=self.model_dim, d_state=self.d_state, dt_rank=self.dt_rank, dropout_p=dropout_p, component_type="residual")
                for _ in range(self.n_decomp_layers)
            ]
        )

        # Component normalization layers
        self.trend_norm = nn.LayerNorm(self.model_dim)
        self.seasonal_norm = nn.LayerNorm(self.model_dim)
        self.residual_norm = nn.LayerNorm(self.model_dim)

        # Fusion mechanism
        if self.fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(embed_dim=self.model_dim, num_heads=max(1, self.model_dim // 64), dropout=dropout_p, batch_first=True)
            self.component_embeddings = nn.Parameter(torch.randn(3, self.model_dim))
        elif self.fusion_method == "gate":
            self.fusion_gate = nn.Sequential(nn.Linear(self.model_dim * 3, self.model_dim), nn.GELU(), nn.Linear(self.model_dim, 3), nn.Softmax(dim=-1))
        else:  # learned
            self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # STL-like smoothing parameters
        self.trend_smoother_param = nn.Parameter(torch.tensor(self.trend_smoother))
        self.seasonal_smoother_param = nn.Parameter(torch.tensor(self.seasonal_smoother))

    def init_weights(self) -> None:
        """Initialize weights for the decomposition model"""
        super().init_weights()

        # Initialize component embeddings if using attention fusion
        if self.fusion_method == "attention":
            nn.init.normal_(self.component_embeddings, std=0.02)

    def _estimate_frequencies(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate dominant frequencies in the input using STFT"""
        B, L, D = u.shape
        u_rearranged = u.permute(0, 2, 1).contiguous()  # [B, D, L]

        # Adaptive FFT size
        n_fft = min(256, max(32, _next_pow2(L)))
        win_length = n_fft
        hop_length = max(8, n_fft // 4)

        # Estimate frequencies across all batch and dimension
        u_flat = rearrange(u_rearranged, "b d l -> (b d) l")
        top_idx, weights, _ = estimate_topk_freqs_stft(u_flat, top_k=self.top_k_freqs, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # Reshape back and map to model dimensions
        top_idx = rearrange(top_idx, "(b d) k -> b d k", b=B, d=D)
        weights = rearrange(weights, "(b d) k -> b d k", b=B, d=D)

        # Map to n_kernels if needed
        if self.n_kernels != D:
            if self.n_kernels < D:
                top_idx = top_idx[:, : self.n_kernels]
                weights = weights[:, : self.n_kernels]
            else:
                # Repeat the first dimension's frequencies
                top_idx = repeat(top_idx[:, 0:1], "b 1 k -> b h k", h=self.n_kernels)
                weights = repeat(weights[:, 0:1], "b 1 k -> b h k", h=self.n_kernels)

        # Convert to normalized frequencies
        omega = 2 * math.pi * top_idx.float() / L

        # EMA update during training
        if self.training and self.freq_momentum > 0:
            # Average across batch dimension for EMA
            omega_avg = omega.mean(dim=0)  # [H, K]
            weights_avg = weights.mean(dim=0)  # [H, K]

            if not bool(self.freq_ema_initialized.item()):
                self.omega_ema.copy_(omega_avg.detach())
                self.freq_weights_ema.copy_(weights_avg.detach())
                self.freq_ema_initialized.fill_(True)
            else:
                m = self.freq_momentum
                self.omega_ema.mul_(1 - m).add_(omega_avg.detach(), alpha=m)
                self.freq_weights_ema.mul_(1 - m).add_(weights_avg.detach(), alpha=m)

            # Use EMA values
            omega = self.omega_ema.unsqueeze(0).expand(B, -1, -1)

        return omega, weights

    def _apply_stl_smoothing(self, component: torch.Tensor, smoother_param: torch.Tensor) -> torch.Tensor:
        """Apply STL-like smoothing to a component"""
        # Simple moving average smoothing
        B, L, D = component.shape
        smoothing_window = max(1, int(L * torch.sigmoid(smoother_param).item()))

        if smoothing_window > 1:
            # Apply 1D convolution for smoothing
            component_reshaped = rearrange(component, "b l d -> (b d) l").unsqueeze(1)
            kernel = torch.ones(1, 1, smoothing_window, device=component.device) / smoothing_window
            smoothed = F.conv1d(F.pad(component_reshaped, (smoothing_window // 2, smoothing_window // 2), mode="reflect"), kernel, padding=0)
            component = rearrange(smoothed.squeeze(1), "(b d) l -> b l d", b=B, d=D)

        return component

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose input into trend, seasonal, and residual components

        Args:
            x: [B, L, D] input time series
        Returns:
            trend: [B, L, D] trend component
            seasonal: [B, L, D] seasonal component
            residual: [B, L, D] residual component
        """
        B, L, D = x.shape

        # Estimate frequencies for seasonal component guidance
        omega, freq_weights = self._estimate_frequencies(x)

        # Initialize components
        trend_input = x
        seasonal_input = x
        residual_input = x

        # Process through trend blocks
        for trend_block in self.trend_blocks:
            trend_input = trend_block(trend_input) + trend_input  # Residual connection
        trend = self.trend_norm(trend_input)

        # Apply STL-like smoothing to trend
        trend = self._apply_stl_smoothing(trend, self.trend_smoother_param)

        # Process through seasonal blocks
        for seasonal_block in self.seasonal_blocks:
            seasonal_input = seasonal_block(seasonal_input) + seasonal_input
        seasonal = self.seasonal_norm(seasonal_input)

        # Apply STL-like smoothing to seasonal
        seasonal = self._apply_stl_smoothing(seasonal, self.seasonal_smoother_param)

        # Residual is what's left after removing trend and seasonal
        residual_input = x - trend - seasonal
        for residual_block in self.residual_blocks:
            residual_input = residual_block(residual_input) + residual_input
        residual = self.residual_norm(residual_input)

        return trend, seasonal, residual

    def _fuse_components(self, trend: torch.Tensor, seasonal: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Fuse the decomposed components using the specified fusion method"""
        B, L, D = trend.shape

        if self.fusion_method == "attention":
            # Stack components and add component embeddings
            components = torch.stack([trend, seasonal, residual], dim=1)  # [B, 3, L, D]
            components = components + self.component_embeddings.view(1, 3, 1, 1)

            # Reshape for attention
            components_flat = rearrange(components, "b c l d -> b (c l) d")

            # Self-attention across components and time
            fused_flat, _ = self.fusion_attention(components_flat, components_flat, components_flat)

            # Reshape back and average across components
            fused = rearrange(fused_flat, "b (c l) d -> b c l d", c=3, l=L)
            fused = fused.mean(dim=1)  # Average across components

        elif self.fusion_method == "gate":
            # Concatenate components and learn gating weights
            combined = torch.cat([trend, seasonal, residual], dim=-1)  # [B, L, 3*D]
            gates = self.fusion_gate(combined)  # [B, L, 3]

            # Apply gates
            components = torch.stack([trend, seasonal, residual], dim=-1)  # [B, L, D, 3]
            fused = (components * gates.unsqueeze(-2)).sum(dim=-1)  # [B, L, D]

        else:  # learned weights
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = weights[0] * trend + weights[1] * seasonal + weights[2] * residual

        return fused

    def get_kernel(self, u: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Override base class method to provide decomposition-based kernel

        Args:
            u: [B, D, L] input tensor (note: different from decompose method)
        Returns:
            [n_kernels, kernel_dim] kernel tensor
        """
        # Convert to expected format for decomposition
        u_decomp = rearrange(u, "b d l -> b l d")  # [B, L, D]

        # Decompose into components
        trend, seasonal, residual = self.decompose(u_decomp)

        # Fuse components
        fused = self._fuse_components(trend, seasonal, residual)

        # Generate kernel from fused representation
        # Use final time step as kernel representation
        kernel_repr = fused[:, -1, :]  # [B, D]

        # Map to kernel dimensions
        if self.n_kernels != kernel_repr.shape[-1]:
            # Project to kernel dimensions
            kernel_repr = F.linear(kernel_repr, torch.randn(self.n_kernels, kernel_repr.shape[-1], device=u.device))

        # Expand to kernel_dim
        kernel = kernel_repr.unsqueeze(-1).expand(-1, -1, self.kernel_dim)

        # Average across batch dimension
        kernel = kernel.mean(dim=0)  # [n_kernels, kernel_dim]

        return kernel

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the reconstructed time series

        Args:
            u: [B, L, D] input tensor
        Returns:
            [B, L, D] reconstructed tensor
        """
        # Decompose input
        trend, seasonal, residual = self.decompose(u)

        # Fuse components
        output = self._fuse_components(trend, seasonal, residual)

        # Apply skip connection if enabled
        if self.skip_connection:
            output = output + u

        return output

    def forecast(self, u: torch.Tensor, horizon: int, return_components: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Generate forecasts using the decomposed components

        Args:
            u: [B, L, D] input time series
            horizon: number of future steps to predict
            return_components: whether to return individual components
        Returns:
            forecast: [B, horizon, D] predicted values
            components (if return_components=True): dict with trend, seasonal, residual forecasts
        """
        B, L, D = u.shape

        # Decompose input
        trend, seasonal, residual = self.decompose(u)

        # Extend each component
        # For trend: use last values with linear extrapolation
        trend_last = trend[:, -1:, :]  # [B, 1, D]
        trend_slope = trend[:, -1:, :] - trend[:, -2:-1, :]  # Simple slope
        trend_forecast = trend_last + trend_slope * torch.arange(1, horizon + 1, device=u.device, dtype=u.dtype).view(1, -1, 1)

        # For seasonal: repeat the pattern
        if L >= horizon:
            seasonal_forecast = seasonal[:, -horizon:, :]
        else:
            # Repeat the entire seasonal pattern if horizon > L
            n_repeats = (horizon + L - 1) // L
            seasonal_extended = seasonal.repeat(1, n_repeats, 1)
            seasonal_forecast = seasonal_extended[:, :horizon, :]

        # For residual: predict using last few residual values
        # Simple approach: exponential decay to zero
        residual_last = residual[:, -1:, :]
        decay_factor = 0.8  # Residual decays over time
        residual_forecast = residual_last * (decay_factor ** torch.arange(1, horizon + 1, device=u.device, dtype=u.dtype).view(1, -1, 1))

        # Combine forecasts
        forecast = trend_forecast + seasonal_forecast + residual_forecast

        if return_components:
            components = {"trend": trend_forecast, "seasonal": seasonal_forecast, "residual": residual_forecast}
            return forecast, components

        return forecast


class ClosedLoopDeepSSMDecomposition(DeepSSMDecomposition):
    """
    Closed-loop version of the deep SSM decomposition model for multi-step forecasting.
    Includes feedback control and iterative forecasting capabilities.
    """

    def __init__(self, lag: int = 1, horizon: int = 1, feedback_strength: float = 0.1, **kwargs: Any) -> None:
        self.lag = lag
        self.horizon = horizon
        self.feedback_strength = feedback_strength
        self.closed_loop = True
        self.inference_only = False

        # Disable skip connection for closed-loop
        kwargs["skip_connection"] = False
        super().__init__(**kwargs)

        # Add feedback control parameters
        self._init_feedback_control()

    def _init_feedback_control(self) -> None:
        """Initialize feedback control mechanisms"""
        # Feedback matrices for each component
        self.trend_feedback = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.seasonal_feedback = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.residual_feedback = nn.Linear(self.model_dim, self.model_dim, bias=False)

        # Feedback mixing weights
        self.feedback_mix = nn.Parameter(torch.ones(3) / 3)

        # Initialize with small weights to start
        nn.init.normal_(self.trend_feedback.weight, std=0.01)
        nn.init.normal_(self.seasonal_feedback.weight, std=0.01)
        nn.init.normal_(self.residual_feedback.weight, std=0.01)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Closed-loop forward pass with forecasting

        Args:
            u: [B, L, D] input tensor
        Returns:
            forecast: [B, horizon, D] forecast values
            control_signal: [B, horizon, D] control signal for training (or None)
        """
        B, L, D = u.shape

        # Initial decomposition
        trend, seasonal, residual = self.decompose(u)

        # Extract initial state from last time step
        trend_state = trend[:, -1, :]  # [B, D]
        seasonal_state = seasonal[:, -1, :]  # [B, D]
        residual_state = residual[:, -1, :]  # [B, D]

        # Generate forecasts iteratively
        forecasts = []
        control_signals = []

        for step in range(self.horizon):
            # Current prediction
            current_pred = trend_state + seasonal_state + residual_state
            forecasts.append(current_pred)

            # Generate control signal for closed-loop training
            if not self.inference_only:
                # Feedback control based on current prediction
                trend_feedback = self.trend_feedback(trend_state)
                seasonal_feedback = self.seasonal_feedback(seasonal_state)
                residual_feedback = self.residual_feedback(residual_state)

                # Mix feedback signals
                feedback_weights = F.softmax(self.feedback_mix, dim=0)
                control_signal = feedback_weights[0] * trend_feedback + feedback_weights[1] * seasonal_feedback + feedback_weights[2] * residual_feedback
                control_signals.append(control_signal)

            # Update states for next step
            # Trend: linear progression with feedback
            trend_delta = (trend[:, -1, :] - trend[:, -2, :]) if L > 1 else torch.zeros_like(trend_state)
            trend_state = trend_state + trend_delta
            if not self.inference_only:
                trend_state = trend_state + self.feedback_strength * trend_feedback

            # Seasonal: cyclical pattern (simplified)
            # In practice, this would use learned seasonal patterns
            seasonal_state = seasonal_state * 0.95  # Simple decay

            # Residual: decay to zero with noise
            residual_state = residual_state * 0.8

        # Stack predictions
        forecast = torch.stack(forecasts, dim=1)  # [B, horizon, D]

        if not self.inference_only and control_signals:
            control_signal = torch.stack(control_signals, dim=1)  # [B, horizon, D]
        else:
            control_signal = None

        return forecast, control_signal

    def set_inference_mode(self, inference_only: bool = True) -> None:
        """Set inference mode to disable control signal generation"""
        self.inference_only = inference_only
 