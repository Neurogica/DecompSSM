"""
DecompSSM forecaster network architecture.
Simplified to use dual SSM for internal decomposition.
"""

import torch
from torch import nn

from models.decompssm.block import Decoder, Encoder
from models.decompssm.embedding import init_embedding
from models.decompssm.mlp import init_mlp


class DecompSSM(nn.Module):
    """
    DecompSSM architecture with dual SSM handling decomposition internally.

    The dual SSM combines:
    - SSM A: Companion SSM for trend modeling
    - SSM B: Seasonal SSM (freq-selective or diagonal) for seasonal modeling
    """

    def __init__(
        self,
        embedding_config: dict,
        encoder_config: dict,
        decoder_config: dict,
        output_config: dict,
        inference_only: bool = False,
        lag: int = 1,
        horizon: int = 1,
        num_channels: int = 1,
    ):
        super().__init__()
        self.lag = lag
        self.horizon = horizon
        self.num_channels = num_channels
        self.inference_only = inference_only

        # Core components
        self.embedding = init_embedding(embedding_config)
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)
        self.decoder.blocks.ssm.lag = lag
        self.decoder.blocks.ssm.horizon = horizon
        self.output = init_mlp(output_config)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (batch, seq_len, channels)

        Returns:
            y: Output tensor of shape (batch, horizon, channels)
        """
        # Encode
        z = self.embedding(u)
        z = self.encoder(z)

        # Decode with dual SSM (handles trend/seasonal internally)
        y, (u_next, u_true) = self.decoder(z)

        # Project to output dimension
        y = self.output(y)

        return y
