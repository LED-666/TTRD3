import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift
from typing import Tuple

def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """
    x = x_in

    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fftn(x, dim=(-2, -1))
    x_freq = fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(dtype=x_in.dtype)


def apply_freeu(hidden_states, res_hidden_states, index):
    if index<2:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * 1
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=0.8)
    return hidden_states, res_hidden_states
