# from einops import rearrange
# import torch
# from typing import Tuple

# def fixed_segmentation(inputs: torch.Tensor, window_size: int) -> torch.Tensor:
#     """
#     Segments a single time series into fixed window intervals, padding with zeros if necessary.
    
#     Args:
#         inputs (torch.Tensor): Time series data of shape [seq_len, features (channels + pos encoding)].
#         window_size (int): Size of each window.

#     Returns:
#         torch.Tensor: Segmented tensor of shape [num_segments, window_size, features].
#     """
#     seq_len, features = inputs.shape

#     padding = (window_size - (seq_len % window_size)) % window_size
#     if padding > 0:
#         pad_tensor = torch.zeros((padding, features), device=inputs.device, dtype=inputs.dtype)
#         inputs = torch.cat([inputs, pad_tensor], dim=0)

#     segmented = rearrange(inputs, '(n w) f -> n w f', w=window_size)
#     return segmented

# inputs = torch.randn(103, 13)  # [seq_len, features]
# window_size = 10
# segmented_inputs = fixed_segmentation(inputs, window_size)
# print(segmented_inputs.shape)  # Output: [11, 10, 13]

from einops import rearrange
import torch
from typing import Tuple, List

def fixed_segmentation(inputs: torch.Tensor, window_size: int, overlap_size: int = 0) -> Tuple[torch.Tensor, List[float]]:
    """
    Segments a single time series into fixed window intervals, allowing for overlap and padding with zeros if necessary.

    Args:
        inputs (torch.Tensor): Time series data of shape [seq_len, features (channels + pos encoding)].
        window_size (int): Size of each window.
        overlap_size (int): Size of overlap between consecutive windows.

    Returns:
        Tuple[torch.Tensor, List[float]]:
            - Segmented tensor of shape [num_windows, window_size, features].
            - List of time points corresponding to the start of each segment.
    """
    seq_len, features = inputs.shape
    stride = window_size - overlap_size
    assert stride > 0, "Stride must be greater than 0. Ensure window_size > overlap_size."

    total_steps = (seq_len - overlap_size + stride - 1) // stride  # Total steps needed including overlap
    padded_len = total_steps * stride + overlap_size
    padding = max(0, padded_len - seq_len)

    if padding > 0:
        pad_tensor = torch.zeros((padding, features), device=inputs.device, dtype=inputs.dtype)
        inputs = torch.cat([inputs, pad_tensor], dim=0)

    segmented = inputs.unfold(0, window_size, stride).permute(0, 2, 1)  # [num_windows, window_size, features]
    return segmented

# inputs = torch.randn(103, 13)  # [seq_len, features]
# window_size = 10
# overlap_size = 5
# segmented_inputs = fixed_segmentation(inputs, window_size, overlap_size)
# print(segmented_inputs.shape)  # Output: [num_windows, 10, 13]
