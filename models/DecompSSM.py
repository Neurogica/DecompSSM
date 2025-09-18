"""
DecompSSM
"""

from typing import Any

import torch
import torch.nn.functional as F

# Direct imports with fallback
from s5 import S5
from torch import Tensor, nn

from mamba_ssm import Mamba, Mamba2


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with S5-compatible interface."""

    def __init__(
        self,
        width: int,
        state_width: int = None,  # Not used but kept for compatibility
        dt_min: float = 0.001,  # Not used but kept for compatibility
        dt_max: float = 0.1,  # Not used but kept for compatibility
        bidir: bool = False,  # Not used but kept for compatibility
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.width = width
        self.n_heads = n_heads
        self.head_dim = width // n_heads

        assert width % n_heads == 0, f"width {width} must be divisible by n_heads {n_heads}"

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, step_scale: Tensor = None) -> Tensor:
        """Forward pass with optional step_scale (ignored for compatibility)."""
        B, N, D = x.shape

        # Multi-head attention
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(attn_output)


class GTSSM(nn.Module):
    """S5 layer with input-dependent step scaling."""

    def __init__(
        self,
        d_model: int,
        state_size: int,
        component_type: str,
        enc_in: int,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidir: bool = False,
        block_count: int = 1,
        factor_rank: int | None = None,
        liquid: bool = False,
        degree: int = 1,
        model_type: str = "s5",  # "s5", "mamba", "mamba2", "attention"
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.component_type = component_type
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.bidir = bidir

        # Base S5 layer
        # S5 constraint: Can't have both factor_rank and bcInit != 'factorized'
        # When using factor_rank, we must set bcInit='factorized' or not set it at all
        # When not using factor_rank, we can set bcInit to other values like 'dense'

        # For higher-order degrees (degree > 1), S5 requires B_bar to be square matrix
        # This means state_width must equal input width (d_model) when degree > 1
        if degree > 1 and state_size != d_model:
            print(f"Warning: For degree={degree}, adjusting state_size from {state_size} to {d_model} for {component_type}")
            state_size = d_model

        # Ensure state_size is compatible with block_count to avoid rank issues
        if state_size % block_count != 0:
            adjusted_state_size = ((state_size // block_count) + 1) * block_count
            print(f"Warning: Adjusted state_size from {state_size} to {adjusted_state_size} for {component_type}")
            state_size = adjusted_state_size

        # Create model based on model_type
        self.model_type = model_type

        if model_type == "s5":
            s5_kwargs = {
                "width": d_model,
                "state_width": state_size,
                "dt_min": max(dt_min, 1e-4),
                "dt_max": min(dt_max, 1.0),
                "bidir": bidir,
                "block_count": block_count,
                "liquid": liquid,
                "degree": degree,
                "bcInit": "dense",
            }
            self.model = S5(**s5_kwargs)

        elif model_type == "mamba":
            mamba_kwargs = {
                "d_model": d_model,
                "d_state": min(max(state_size, 16), 64),  # Mamba typically uses 16-64
                "d_conv": 4,
                "expand": 2,
            }
            self.model = Mamba(**mamba_kwargs)
            if bidir:
                self.model_reverse = Mamba(**mamba_kwargs)

        elif model_type == "mamba2":
            mamba2_kwargs = {
                "d_model": d_model,
                "d_state": min(max(state_size, 16), 128),  # Mamba2 typically uses 64-128
                "d_conv": 4,
                "expand": 2,
                "dt_min": max(dt_min, 1e-4),
                "dt_max": min(dt_max, 1.0),
            }
            self.model = Mamba2(**mamba2_kwargs)
            if bidir:
                self.model_reverse = Mamba2(**mamba2_kwargs)

        elif model_type == "attention":
            self.model = MultiHeadAttention(
                width=d_model,
                state_width=state_size,
                dt_min=dt_min,
                dt_max=dt_max,
                bidir=bidir,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"Created {model_type} model for {component_type} component")

        # Channel-specific position embedding
        self.position_embedding = nn.Parameter(torch.randn(enc_in, d_model))

        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Learned step-scale predictor (per-sample, per-channel)
        # Pool over feature dim per channel to 1 value, then predict step
        hidden_size = max(4, d_model // 8)
        self.step_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            # nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize step_proj to output values close to 0 for stable step_scale
        with torch.no_grad():
            last_layer = self.step_proj[-1]
            if isinstance(last_layer, nn.Linear) and last_layer.weight is not None:
                # More conservative initialization for numerical stability
                nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)
                if last_layer.bias is not None:
                    nn.init.constant_(last_layer.bias, -2.0)  # Bias towards smaller steps

        # Component-specific frequency gating with regularization
        self.freq_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),  # Add normalization
            self._get_component_activation(),
            # nn.ReLU(),
            nn.Dropout(dropout),  # Add hidden dropout
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def _get_component_activation(self) -> nn.Module:
        """Component-specific activation based on frequency characteristics."""
        if self.component_type == "trend":
            return nn.Tanh()  # Smooth for low-frequency
        elif self.component_type == "seasonal":
            return nn.GELU()  # Balanced for mid-frequency
        else:  # residual
            return nn.ReLU()  # Sharp for high-frequency

    def forward(self, x: Tensor) -> Tensor:
        """
        Adaptive forward pass with input-dependent step scaling.
        """
        # Add channel-specific position embedding
        x = x + self.position_embedding.unsqueeze(0)  # [B, N, d_model]

        # Input normalization
        x = self.input_norm(x)

        # Predict positive step scale per sample and channel
        # Pool over feature dim -> [B, N, 1]
        pooled_ch = x.mean(dim=-1, keepdim=True)
        step_logits = self.step_proj(pooled_ch).squeeze(-1)  # [B, N]
        # Ensure step_scale is within a safe range to avoid rank issues
        # Clamp the logits to prevent extreme values
        step_logits = torch.clamp(step_logits, min=-5.0, max=5.0)
        step_scale = F.softplus(step_logits) + 0.01  # [B, N] - increased minimum value

        # Additional safety: ensure step_scale is within reasonable bounds
        step_scale = torch.clamp(step_scale, min=0.01, max=2.0)

        try:
            # Model-specific forward pass
            if self.model_type == "s5":
                # S5 supports step_scale
                if step_scale is not None:
                    try:
                        output = self.model(x, step_scale=step_scale)
                    except:
                        output = self.model(x)
                else:
                    output = self.model(x)

            elif self.model_type in ["mamba", "mamba2"]:
                # Handle bidirectional processing for Mamba models
                if hasattr(self, "model_reverse"):
                    # Forward direction
                    forward_out = self.model(x)
                    # Reverse direction
                    x_reversed = torch.flip(x, dims=[1])
                    reverse_out = self.model_reverse(x_reversed)
                    reverse_out = torch.flip(reverse_out, dims=[1])
                    # Combine outputs
                    output = (forward_out + reverse_out) / 2
                else:
                    output = self.model(x)

            elif self.model_type == "attention":
                output = self.model(x, step_scale=step_scale)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        except Exception as e:
            print(f"{self.model_type} forward failed for {self.component_type}: {e}")
            # Try without step_scale as fallback
            try:
                if self.model_type == "s5":
                    output = self.model(x)
                elif self.model_type in ["mamba", "mamba2"]:
                    output = self.model(x)
                else:
                    output = self.model(x)
            except Exception as e2:
                print(f"{self.model_type} forward failed again for {self.component_type}: {e2}")
                output = x  # Final fallback to input

        # Apply component-specific frequency gating
        gate = self.freq_gate(output)
        output = output * gate

        # Output normalization and dropout
        output = self.output_norm(output)
        final_output: torch.Tensor = self.dropout(output)
        return final_output


def decomposition_loss(
    trend: Tensor,
    seasonal: Tensor,
    residual: Tensor,
    original: Tensor,
    lambda_reconstruction: float = 1.0,
    lambda_orthogonality: float = 0.1,
) -> Tensor:
    """Decomposition loss combining reconstruction and orthogonality terms."""
    # Shapes: trend/seasonal/residual/original are [B, N, D]
    batch_size, _, _ = trend.shape

    # 1. Perfect Reconstruction Loss
    reconstruction = trend + seasonal + residual
    reconstruction_loss = F.mse_loss(reconstruction, original)

    def orthogonality_loss() -> torch.Tensor:
        """Enforce statistical independence between components."""
        # Flatten for correlation computation
        trend_flat = trend.view(batch_size, -1)  # [B, N*D]
        seasonal_flat = seasonal.view(batch_size, -1)
        residual_flat = residual.view(batch_size, -1)

        # Compute cross-correlations (should be minimized)
        ts_correlation = torch.abs(F.cosine_similarity(trend_flat, seasonal_flat, dim=1).mean())
        tr_correlation = torch.abs(F.cosine_similarity(trend_flat, residual_flat, dim=1).mean())
        sr_correlation = torch.abs(F.cosine_similarity(seasonal_flat, residual_flat, dim=1).mean())

        # Sum of absolute correlations (minimize for independence)
        total_correlation = ts_correlation + tr_correlation + sr_correlation

        return total_correlation

    orthogonality_loss_value = orthogonality_loss()

    # Combine simplified theoretically grounded losses
    # Focus on the two most fundamental principles of time series decomposition
    total_loss = lambda_reconstruction * reconstruction_loss + lambda_orthogonality * orthogonality_loss_value

    return total_loss


class Model(nn.Module):
    """
    DecompSSM
    """

    def __init__(self, configs: Any):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.model_type = getattr(configs, "model_type", "s5")

        # Model dimensions
        self.d_model = getattr(configs, "d_model", 128)
        self.state_size = getattr(configs, "state_size", 64)  # S5 state dimension
        self.dropout = getattr(configs, "dropout", 0.1)

        # S5 temporal dynamics parameters - optimized for long-term forecasting (720 steps)
        # Larger dt ranges for better long-term dependency modeling
        self.trend_dt_min = getattr(configs, "trend_dt_min", 0.02)  # Increased for longer trends
        self.trend_dt_max = getattr(configs, "trend_dt_max", 0.2)  # Increased for longer trends
        self.seasonal_dt_min = getattr(configs, "seasonal_dt_min", 0.002)  # Slightly increased
        self.seasonal_dt_max = getattr(configs, "seasonal_dt_max", 0.02)  # Increased for longer patterns
        self.residual_dt_min = getattr(configs, "residual_dt_min", 0.0002)  # Slightly increased
        self.residual_dt_max = getattr(configs, "residual_dt_max", 0.002)  # Increased

        self.trend_bidir = getattr(configs, "trend_bidir", False)  # Keep unidirectional for trend
        self.seasonal_bidir = getattr(configs, "seasonal_bidir", True)  # Bidirectional for seasonal patterns
        self.residual_bidir = getattr(configs, "residual_bidir", True)  # Enable bidirectional for residual

        # Channel interaction parameters - enhanced for long-term forecasting
        self.use_channel_interaction = getattr(configs, "use_channel_interaction", True)
        self.channel_interaction_strength_init = getattr(configs, "channel_interaction_strength", 0.15)  # Increased for better multivariate modeling

        # Simplified theoretically grounded auxiliary loss weights - adjusted for long-term stability
        self.lambda_reconstruction = getattr(configs, "lambda_reconstruction", 1.0)
        self.lambda_orthogonality = getattr(configs, "lambda_orthogonality", 0.02)

        # Embedding from temporal dimension to feature dimension
        # Input [B, N, seq_len] -> [B, N, d_model] where N = enc_in (channels)
        self.value_embedding = nn.Linear(self.seq_len, self.d_model)

        # Three frequency-specialized S5 layers with natural decomposition
        self.trend_s5 = GTSSM(
            d_model=self.d_model,
            state_size=self.state_size,
            component_type="trend",
            enc_in=self.enc_in,
            dropout=self.dropout,
            dt_min=self.trend_dt_min,
            dt_max=self.trend_dt_max,
            bidir=self.trend_bidir,
            model_type=self.model_type,
        )
        self.seasonal_s5 = GTSSM(
            d_model=self.d_model,
            state_size=self.state_size,
            component_type="seasonal",
            enc_in=self.enc_in,
            dropout=self.dropout,
            dt_min=self.seasonal_dt_min,
            dt_max=self.seasonal_dt_max,
            bidir=self.seasonal_bidir,
            model_type=self.model_type,
        )
        self.residual_s5 = GTSSM(
            d_model=self.d_model,
            state_size=self.state_size,
            component_type="residual",
            enc_in=self.enc_in,
            dropout=self.dropout,
            dt_min=self.residual_dt_min,
            dt_max=self.residual_dt_max,
            bidir=self.residual_bidir,
            model_type=self.model_type,
        )

        # Conditional channel interaction setup
        if self.use_channel_interaction:
            # Learnable channel interaction strength (scalar)
            self.channel_interaction_strength = nn.Parameter(torch.tensor(self.channel_interaction_strength_init))

            # Simple global channel context projection and normalization (SSM-friendly)
            self.global_context_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.channel_norm = nn.LayerNorm(self.d_model)

        # Map concatenated components to d_model, then project to prediction length (iTransformer style)
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len, bias=True),
        )

        # Training parameters
        self.training_step = 0
        self.aux_loss_weight = getattr(configs, "aux_loss_weight", 0.1)

    def global_context_refinement_module(self, component: torch.Tensor) -> torch.Tensor:
        """
        GlobalContextRefinementModule: Apply universal channel interaction for multivariate learning.

        Args:
            component: Input component tensor [B, N, d_model]

        Returns:
            Enhanced component with channel interactions [B, N, d_model]
        """
        if not self.use_channel_interaction:
            return component

        # Global context across channels (mean over channel axis)
        global_context = component.mean(dim=1, keepdim=True)  # [B, 1, d_model]
        global_context = self.global_context_proj(global_context)  # [B, 1, d_model]

        # Adaptive residual mixing with learnable strength
        interaction_strength = torch.sigmoid(self.channel_interaction_strength)
        out = component + interaction_strength * global_context.expand_as(component)

        # Stabilize with LayerNorm
        final_component: torch.Tensor = self.channel_norm(out)
        return final_component

    def forward(self, x_enc: torch.Tensor, x_mark_enc: Any = None, x_dec: Any = None, x_mark_dec: Any = None, mask: Any = None) -> torch.Tensor:
        """
        Forward pass with frequency-aware S5 decomposition.

        The key innovation is natural decomposition through frequency-specialized S5 layers,
        each with different temporal dynamics and frequency-aware auxiliary regularization.
        """
        batch_size, seq_len, num_channels = x_enc.shape

        # iTransformer-style Non-stationary normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.transpose(1, 2)  # [B, enc_in, seq_len] -> [B, N, seq_len]
        _, N, _ = x_enc.shape

        # Embedding: [B, N, seq_len] -> [B, N, d_model]
        x_embedded = self.value_embedding(x_enc)  # [B, N, d_model]

        trend = self.trend_s5(x_embedded)  # [B, N, d_model]
        seasonal = self.seasonal_s5(x_embedded)  # [B, N, d_model]
        residual = self.residual_s5(x_embedded)  # [B, N, d_model]

        # Universal multivariate enhancement: apply channel interaction for all data
        trend = self.global_context_refinement_module(trend)
        seasonal = self.global_context_refinement_module(seasonal)
        residual = self.global_context_refinement_module(residual)

        # Auxiliary loss using embedded features (before position encoding)
        if self.training:
            try:
                aux_loss = decomposition_loss(
                    trend,
                    seasonal,
                    residual,
                    x_embedded,  # Use embedded features before position encoding
                    lambda_reconstruction=self.lambda_reconstruction,
                    lambda_orthogonality=self.lambda_orthogonality,
                )
                self.aux_loss = aux_loss * self.aux_loss_weight
                self.training_step += 1
            except Exception as e:
                print(f"Auxiliary loss computation failed: {e}")
                self.aux_loss = torch.tensor(0.0, device=x_embedded.device)

        # iTransformer-style prediction head
        concat_features = torch.cat([trend, seasonal, residual], dim=-1)  # [B, N, 3*d_model]
        output = self.output_projection(concat_features)  # [B, N, pred_len]

        # Transpose to [B, pred_len, N] to match expected output format
        output = output.permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]

        # iTransformer-style de-normalization
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return output

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Get the auxiliary loss for training."""
        if hasattr(self, "aux_loss"):
            return self.aux_loss
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
