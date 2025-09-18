import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class StandardNormalization(nn.Module):
    """
    Standard z-score normalization (mean=0, std=1)
    Most stable normalization for time series forecasting.
    """

    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")
        return x

    def _get_statistics(self, x):
        # Calculate mean and std across time dimension (dim=1)
        # x shape: [B, T, N] -> reduce over time dimension
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()  # [B, 1, N]
        self.std = torch.std(x, dim=1, keepdim=True, unbiased=False).detach() + self.eps  # [B, 1, N]

    def _normalize(self, x):
        # Standard z-score normalization: (x - mean) / std
        return (x - self.mean) / self.std

    def _denormalize(self, x):
        # Reverse z-score normalization: x * std + mean
        return x * self.std + self.mean


def main_freq_part(x, k, rfft=True):
    """
    Extract main frequency components and residual from input signal.

    Args:
        x: Input tensor of shape (B, T, N)
        k: Number of top frequency components to keep
        rfft: Whether to use real FFT (True) or complex FFT (False)

    Returns:
        norm_input: Residual signal (input - main frequency components)
        x_filtered: Main frequency components
    """
    # freq normalization
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    return norm_input, x_filtered


class MLPfreq(nn.Module):
    """
    MLP model for frequency domain prediction.
    """

    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(nn.Linear(64 + seq_len, 128), nn.ReLU(), nn.Linear(128, pred_len))

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)


class FAN(nn.Module):
    """
    Frequency-based Adaptive Normalization (FAN).

    FAN first subtracts bottom k frequency components from the original series,
    then uses an MLP to predict the main frequency signal for normalization.

    Args:
        seq_len: Length of input sequence
        pred_len: Length of prediction sequence
        enc_in: Number of input features/channels
        freq_topk: Number of top frequency components to keep (default: 20)
        rfft: Whether to use real FFT (default: True)
    """

    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk)
        self.rfft = rfft

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def loss(self, true):
        """
        Calculate loss for frequency decomposition.

        Args:
            true: Ground truth tensor of shape (B, O, N)

        Returns:
            Combined MSE loss for main frequency signal and residual
        """
        # freq normalization
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)

        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)

    def normalize(self, input_tensor):
        """
        Normalize input by removing main frequency components.

        Args:
            input_tensor: Input tensor of shape (B, T, N)

        Returns:
            Normalized input (residual signal)
        """
        # (B, T, N)
        original_shape = input_tensor.shape
        norm_input, x_filtered = main_freq_part(input_tensor, self.freq_topk, self.rfft)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input_tensor.transpose(1, 2)).transpose(1, 2)

        return norm_input.reshape(original_shape)

    def denormalize(self, input_norm):
        """
        Denormalize by adding back predicted main frequency components.

        Args:
            input_norm: Normalized input tensor of shape (B, O, N)

        Returns:
            Denormalized output
        """
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        original_shape = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(original_shape)

    def forward(self, batch_x, mode="n"):
        """
        Forward pass for FAN normalization.

        Args:
            batch_x: Input tensor
            mode: 'n' for normalize, 'd' for denormalize

        Returns:
            Normalized or denormalized tensor
        """
        if mode == "n":
            return self.normalize(batch_x)
        elif mode == "d":
            return self.denormalize(batch_x)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")


class MLP(nn.Module):
    """
    MLP helper class for SAN normalization.
    """

    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.period_len = period_len
        self.mode = mode

        if mode == "std":
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()

        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == "std" else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)


class SAN(nn.Module):
    """
    Seasonal Adaptive Normalization (SAN).

    SAN performs adaptive normalization based on seasonal patterns in the data.
    It uses separate MLPs to predict mean and standard deviation adjustments.

    Args:
        seq_len: Length of input sequence
        pred_len: Length of prediction sequence
        period_len: Length of seasonal period
        enc_in: Number of input features/channels
        station_type: Type of normalization ('adaptive' or other)
    """

    def __init__(self, seq_len, pred_len, period_len, enc_in, station_type="adaptive"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in
        self.enc_in = enc_in
        self.station_type = station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode="mean").float()
        self.model_std = MLP(seq_len, pred_len, enc_in, self.period_len, mode="std").float()

    def normalize(self, input_tensor):
        """
        Normalize input using seasonal adaptive normalization.

        Args:
            input_tensor: Input tensor of shape (B, T, N)

        Returns:
            Tuple of (normalized_input, station_predictions)
        """
        # (B, T, N)
        if self.station_type == "adaptive":
            original_shape = input_tensor.shape
            batch_size, seq_length, num_features = original_shape
            input_reshaped = input_tensor.reshape(batch_size, -1, self.period_len, num_features)
            mean = torch.mean(input_reshaped, dim=-2, keepdim=True)
            std = torch.std(input_reshaped, dim=-2, keepdim=True)
            norm_input = (input_reshaped - mean) / (std + self.epsilon)
            input_flat = input_tensor.reshape(original_shape)
            mean_all = torch.mean(input_flat, dim=1, keepdim=True)

            outputs_mean = self.model(mean.squeeze(2) - mean_all, input_flat - mean_all) * self.weight[0] + mean_all * self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input_flat)

            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)

            return norm_input.reshape(original_shape), outputs[:, -self.pred_len_new :, :]
        else:
            return input_tensor, None

    def denormalize(self, input_tensor, station_pred):
        """
        Denormalize input using station predictions.

        Args:
            input_tensor: Normalized input tensor of shape (B, O, N)
            station_pred: Station predictions from normalize method

        Returns:
            Denormalized output
        """
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        if self.station_type == "adaptive":
            original_shape = input_tensor.shape
            batch_size, seq_length, num_features = original_shape
            input_reshaped = input_tensor.reshape(batch_size, -1, self.period_len, num_features)
            mean = station_pred[:, :, : self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels :].unsqueeze(2)
            output = input_reshaped * (std + self.epsilon) + mean
            return output.reshape(original_shape)
        else:
            return input_tensor

    def forward(self, batch_x, mode="n", station_pred=None):
        """
        Forward pass for SAN normalization.

        Args:
            batch_x: Input tensor
            mode: 'n' for normalize, 'd' for denormalize
            station_pred: Station predictions (required for denormalization)

        Returns:
            Normalized/denormalized tensor or tuple for normalization
        """
        if mode == "n":
            return self.normalize(batch_x)
        elif mode == "d":
            return self.denormalize(batch_x, station_pred)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")


class NonStationaryNormalization(nn.Module):
    """
    Non-stationary normalization for time series data.

    This normalization method computes mean and standard deviation across the time dimension
    and stores them for later denormalization. It's particularly effective for non-stationary
    time series data where statistical properties change over time.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.means: torch.Tensor | None = None
        self.stdev: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize using non-stationary normalization.

        Args:
            x: Input tensor of shape (B, T, N)

        Returns:
            Normalized tensor of shape (B, T, N)
        """
        # Compute statistics across time dimension (dim=1)
        self.means = x.mean(1, keepdim=True).detach()  # [B, 1, N]
        x_centered = x - self.means
        self.stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + self.eps).detach()  # [B, 1, N]

        # Normalize
        x_norm = x_centered / self.stdev
        return x_norm

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize using stored statistics.

        Args:
            x: Normalized tensor of shape (B, T, N)

        Returns:
            Denormalized tensor of shape (B, T, N)
        """
        if self.means is None or self.stdev is None:
            raise RuntimeError("normalize() must be called before denormalize()")

        # Non-stationary denormalization
        # Handle different shapes: means/stdev are [B, 1, N], x might be [B, pred_len, N]
        b, t, n = x.shape

        # Expand means and stdev to match prediction length
        means_expanded = self.means[:, 0, :].unsqueeze(1).repeat(1, t, 1)  # [B, T, N]
        stdev_expanded = self.stdev[:, 0, :].unsqueeze(1).repeat(1, t, 1)  # [B, T, N]

        # Denormalize: x_denorm = x * stdev + means
        x_denorm = x * stdev_expanded + means_expanded
        return x_denorm

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """Forward pass with explicit mode."""
        if mode in ["norm", "normalize"]:
            return self.normalize(x)
        elif mode in ["denorm", "denormalize"]:
            return self.denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: 'norm', 'denorm'")


def get_normalization(norm_type, **kwargs):
    """
    Factory function to get different normalization methods.

    Args:
        norm_type: Type of normalization ('revin', 'standard', 'fan', 'san', 'nonstationary')
        **kwargs: Additional arguments for the normalization method

    Returns:
        Normalization module instance
    """
    if norm_type.lower() == "revin":
        return RevIN(kwargs.get("num_features", 1), kwargs.get("eps", 1e-5), kwargs.get("affine", True))
    elif norm_type.lower() == "standard":
        return StandardNormalization(kwargs.get("num_features", 1), kwargs.get("eps", 1e-8))
    elif norm_type.lower() == "fan":
        return FAN(kwargs.get("seq_len", 96), kwargs.get("pred_len", 96), kwargs.get("enc_in", 1), kwargs.get("freq_topk", 20), kwargs.get("rfft", True))
    elif norm_type.lower() == "san":
        return SAN(
            kwargs.get("seq_len", 96), kwargs.get("pred_len", 96), kwargs.get("period_len", 24), kwargs.get("enc_in", 1), kwargs.get("station_type", "adaptive")
        )
    elif norm_type.lower() == "nonstationary":
        return NonStationaryNormalization(kwargs.get("num_features", 1), kwargs.get("eps", 1e-5))
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Supported types: 'revin', 'standard', 'fan', 'san', 'nonstationary'")


class Normalization(nn.Module):
    """
    Unified wrapper for all normalization methods.

    This wrapper provides a consistent interface for all normalization types,
    handling the different calling conventions (mode parameters, return types, etc.)
    automatically based on the normalization method used.
    """

    def __init__(self, norm_type, **kwargs):
        super().__init__()
        self.norm_type = norm_type.lower()
        self.normalizer = get_normalization(norm_type, **kwargs)
        self.station_pred = None  # For SAN normalization

    def normalize(self, x):
        """Unified normalization interface."""
        if hasattr(self.normalizer, "station_type") and self.normalizer.station_type == "adaptive":
            # SAN normalization returns tuple (normalized_data, station_predictions)
            x_norm, self.station_pred = self.normalizer(x, mode="n")
            return x_norm
        elif hasattr(self.normalizer, "normalize"):
            # FAN or NonStationary normalization
            if self.norm_type == "nonstationary":
                return self.normalizer.normalize(x)
            else:
                return self.normalizer(x, mode="n")
        else:
            # RevIN or Standard normalization
            return self.normalizer(x, mode="norm")

    def denormalize(self, x):
        """Unified denormalization interface."""
        if hasattr(self.normalizer, "station_type") and self.normalizer.station_type == "adaptive":
            # SAN denormalization requires station predictions
            if self.station_pred is not None:
                return self.normalizer(x, mode="d", station_pred=self.station_pred)
            else:
                print("Warning: SAN normalization used but no station predictions available")
                return x
        elif hasattr(self.normalizer, "denormalize"):
            # FAN or NonStationary normalization
            if self.norm_type == "nonstationary":
                return self.normalizer.denormalize(x)
            else:
                return self.normalizer(x, mode="d")
        else:
            # RevIN or Standard normalization
            return self.normalizer(x, mode="denorm")

    def forward(self, x, mode="norm"):
        """Forward pass with explicit mode."""
        if mode in ["norm", "normalize"]:
            return self.normalize(x)
        elif mode in ["denorm", "denormalize"]:
            return self.denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: 'norm', 'denorm'")
