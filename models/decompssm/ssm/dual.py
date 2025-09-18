from typing import Any

import torch
from torch import nn


class DualSSM(nn.Module):
    """
    Open-loop dual-stream SSM wrapper that merges two SSMs with a dynamic gate.
    - ssm_a: e.g., companion (trend-like)
    - ssm_b: e.g., frequency-selective seasonal SSM
    Gate is computed per-sample and per-channel from pooled inputs.
    """

    def __init__(
        self,
        ssm_a_config: dict[str, Any],
        ssm_b_config: dict[str, Any],
        model_dim: int,
        num_channels: int | None = None,
        use_dynamic_gate: bool = True,
        gate_hidden_dim: int | None = None,
        gate_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Lazy import to avoid circular import during package initialization
        from . import init_ssm  # noqa: WPS433

        self.ssm_a = init_ssm(ssm_a_config)
        self.ssm_b = init_ssm(ssm_b_config)
        self._lag = getattr(self.ssm_a, "lag", 1)
        self._horizon = getattr(self.ssm_a, "horizon", 1)
        self.model_dim = model_dim
        self.num_channels = num_channels if num_channels is not None else model_dim
        self.use_dynamic_gate = use_dynamic_gate
        hidden_dim = gate_hidden_dim if gate_hidden_dim is not None else max(16, model_dim // 2)
        if self.use_dynamic_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=gate_dropout),
                nn.Linear(hidden_dim, self.num_channels),
            )
        else:
            self.register_module("gate_mlp", None)
            self.alpha_logits = nn.Parameter(torch.zeros(self.num_channels))

    @property
    def lag(self) -> int:
        return self._lag

    @lag.setter
    def lag(self, value: int) -> None:
        self._lag = int(value)
        if hasattr(self.ssm_a, "lag"):
            self.ssm_a.lag = int(value)
        if hasattr(self.ssm_b, "lag"):
            self.ssm_b.lag = int(value)

    @property
    def horizon(self) -> int:
        return self._horizon

    @horizon.setter
    def horizon(self, value: int) -> None:
        self._horizon = int(value)
        if hasattr(self.ssm_a, "horizon"):
            self.ssm_a.horizon = int(value)
        if hasattr(self.ssm_b, "horizon"):
            self.ssm_b.horizon = int(value)

    def _merge(self, u: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        # u,y_: [B, L, D]
        if self.use_dynamic_gate and getattr(self, "gate_mlp", None) is not None:
            ctx = u.mean(dim=1)  # [B, D]
            alpha = torch.sigmoid(self.gate_mlp(ctx)).view(-1, 1, self.num_channels)
        else:
            alpha = torch.sigmoid(self.alpha_logits).view(1, 1, self.num_channels)
        return alpha * y_a + (1.0 - alpha) * y_b

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        y_a = self.ssm_a(u)
        y_b = self.ssm_b(u)
        return self._merge(u, y_a, y_b)


class ClosedLoopDualSSM(DualSSM):
    """
    Closed-loop dual-stream SSM wrapper: merges both outputs and the next-step inputs.
    Requires that ssm_a and ssm_b are closed-loop SSMs returning (y, y_u).
    """

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out_a = self.ssm_a(u)
        out_b = self.ssm_b(u)
        if not isinstance(out_a, tuple) or not isinstance(out_b, tuple):
            # Fallback: treat as open-loop if children don't return tuples
            y = self._merge(u, out_a if not isinstance(out_a, tuple) else out_a[0], out_b if not isinstance(out_b, tuple) else out_b[0])
            return y, None
        y_a, u_next_a = out_a
        y_b, u_next_b = out_b
        y = self._merge(u, y_a, y_b)
        # Merge next-time-step inputs using the same gate computed from current u
        if u_next_a is not None and u_next_b is not None:
            u_next = self._merge(u, u_next_a, u_next_b)
        else:
            u_next = None
        return y, u_next
