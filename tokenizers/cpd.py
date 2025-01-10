import ruptures as rpt
from typing import List, Tuple
import torch

def cpd_change_points(inputs: torch.Tensor, model: str, pen: int, min_size: int, num_points: int, **kwargs) -> List[int]:
    """
    Detects change points in a time series using ruptures.

    Args:
        inputs (torch.Tensor): Time series data of shape [seq_len, features].
        model (str): The change point detection model to use ('l1', 'l2', 'rbf', etc.).
        pen (int): penalty for algorithm
        min_size (int): parameter for algorithm to detect change points at least `min_size`
        **kwargs: Additional parameters to pass to the ruptures change point detection algorithm.

    Returns:
        List[int]: List of change points indicating the start of each segment.
    """
    inputs_np = inputs.cpu().numpy()

    if model == "kernel":
        algo = rpt.KernelCPD(kernel="rbf", min_size=min_size, **kwargs)
    elif model == "pelt":
        algo = rpt.Pelt(model="rbf", min_size=min_size, **kwargs)
    elif model == "dynamic":
        algo = rpt.Dynp(model="rbf", min_size=min_size, **kwargs)
    elif model == "binseg":
        algo = rpt.Binseg(model="rbf", min_size=min_size, **kwargs)
    elif model == "bottomup":
        algo = rpt.BottomUp(model="rbf", min_size=min_size).fit(inputs_np)
    else:
        raise ValueError(f"Unsupported model: {model}")

    algo.fit(inputs_np)
    change_points = algo.predict(pen)

    return change_points


def segmentation(inputs: torch.Tensor, change_points: List[int], segment_length: int) -> List[torch.Tensor]:
    """
    Segments a time series into fixed-length intervals and returns a tensor of segments.

    Args:
        inputs (torch.Tensor): Time series data of shape [seq_len, features].
        change_points (List[int]): Change points indicating segment boundaries.
        segment_length (int): Desired length of each segment.

    Returns:
        Tuple[torch.Tensor, List[Tuple[int, int]]]:
            - Tensor of shape [num_segments, segment_length, features] containing all segments.
            - List of tuples indicating the start and end indices of each segment.
    """

    segments = []
    intervals = []
    half_length = segment_length // 2
    seq_len, _ = inputs.shape

    for i in range(len(change_points) - 1):
        midpoint = (change_points[i] + change_points[i + 1]) // 2

        start = max(0, midpoint - half_length)
        end = min(seq_len, midpoint + half_length)

        segment = inputs[start:end]
        if segment.shape[0] < segment_length:
            padding_needed = segment_length - segment.shape[0]

            if start == 0: 
                end = min(seq_len, end + padding_needed)
            elif end == seq_len:  # If at the end of the sequence
                start = max(0, start - padding_needed)
            else:
                left_extend = padding_needed // 2
                right_extend = padding_needed - left_extend
                start = max(0, start - left_extend)
                end = min(seq_len, end + right_extend)
                
            segment = inputs[start:end]

        # Ensure the segment is exactly `segment_length`
        if segment.shape[0] != segment_length:
            raise RuntimeError(f"Segment length mismatch: expected {segment_length}, got {segment.shape[0]}")

        segments.append(segment)
        intervals.append([start, end])

    segments_tensor = torch.stack(segments)
    return segments_tensor, intervals



def cpd_segmentation(inputs: torch.Tensor, model: str, segment_length: int, pen: int, min_size: int,  num_points: int,  **kwargs) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Combines change point detection and fixed-length segmentation.

    Args:
        inputs (torch.Tensor): Time series data of shape [seq_len, features].
        model (str): The change point detection model to use ('l1', 'l2', 'rbf', etc.).
        segment_length (int): Desired length of each segment.
        pen (int): penalty for algorithm
        min_size (int): parameter for algorithm to detect change points at least `min_size`
        num_points (int): number of change points to predict
        **kwargs: Additional parameters for change point detection.

    Returns:
        Tuple[List[torch.Tensor], List[int], List[List[int]]:
            - List of tensors, each representing a fixed-length segment of the time series.
            - List of change points.
            - List of intervals of segment start, end index
    """
    seq_len, _ = inputs.shape
    change_points = cpd_change_points(inputs, model, pen, min_size, num_points, **kwargs)
    if change_points[0] != 0:
        change_points = [0] + change_points
    if change_points[-1] != seq_len - 1:
        change_points += [seq_len - 1]

    segments, intervals = segmentation(inputs, change_points, segment_length)
    return segments, change_points, intervals

# inputs = torch.randn(103, 13)  # [seq_len, features]
# model = 'l2'  # Change point detection model
# segment_length = 20
# segments, change_points = cpd_segmentation(inputs, model, segment_length, pen=10)
# print(f"Number of segments: {len(segments)}")
# print(f"Change points: {change_points}")
