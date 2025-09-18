import torch


def estimate_periods_stft(u: torch.Tensor, n_fft: int = 32, hop_length: int = 16, win_length: int = 32) -> torch.Tensor:
    """
    u: [B, L, D]
    Returns: [D] periods by taking peak over frequency bins averaged across batch and time.
    Vectorized over channels for speed.
    """
    B, L, D = u.shape
    device = u.device
    dtype = u.dtype
    window = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)

    # Merge batch and channel dims to run a single STFT
    x = u.permute(0, 2, 1).reshape(B * D, L)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    # stft: [B*D, F, T']
    mag = torch.abs(stft).mean(dim=0)  # average over batch -> [F, T']
    mag = mag.mean(dim=-1)  # average over time -> [F]
    # To produce a per-channel estimate, use per-channel magnitudes
    mag_cd = torch.abs(stft).mean(dim=-1)  # [B*D, F]
    mag_cd = mag_cd.view(B, D, -1).mean(dim=0)  # average over batch -> [D, F]

    # Remove DC component
    mag_cd[:, 0] = 0

    # Peak frequency bin per channel
    freqs = torch.argmax(mag_cd, dim=1)  # [D]

    # Convert freq bin to approximate period in samples; clamp to at least 2
    periods = torch.clamp(L // torch.clamp(freqs, min=1), min=2)
    return periods.to(device=device, dtype=dtype)


def estimate_topk_freqs_stft(u: torch.Tensor, top_k: int = 3, n_fft: int = 64, hop_length: int = 32, win_length: int = 64):
    """
    Estimate top-K frequency bins per channel with weights.
    Args:
      u: [B, L, D]
    Returns:
      idx: [D, K] (long) top-K frequency bin indices
      w:   [D, K] (float) normalized weights per channel (sum to 1)
      L:   sequence length (for converting to periods)
    """
    B, L, D = u.shape
    device = u.device
    dtype = u.dtype
    window = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)

    x = u.permute(0, 2, 1).reshape(B * D, L)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    mag_cd = torch.abs(stft).mean(dim=-1)  # [B*D, F]
    mag_cd = mag_cd.view(B, D, -1).mean(dim=0)  # [D, F]
    mag_cd[:, 0] = 0

    top_vals, top_idx = torch.topk(mag_cd, k=min(top_k, mag_cd.shape[1]), dim=1)  # [D, K]
    weights = top_vals / (top_vals.sum(dim=1, keepdim=True) + 1e-8)
    return top_idx.long(), weights.to(dtype), L
