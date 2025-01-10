import numpy as np
import matplotlib.pyplot as plt
import torch

# Simulate a time series for visualization
np.random.seed(0)
time_series = np.cumsum(np.random.randn(103))  # Random walk for time series
change_points = [0, 25, 50, 75, 103]  # Example change points from CPD

# Plot the time series with segments
plt.figure(figsize=(12, 6))
plt.plot(range(len(time_series)), time_series, label="Time Series", color="blue")
for cp in change_points:
    plt.axvline(x=cp, color="red", linestyle="--", label="Change Point" if cp == change_points[0] else None)

# Add segments for better visualization
for i in range(len(change_points) - 1):
    start = change_points[i]
    end = change_points[i + 1]
    plt.axvspan(start, end, color=f"C{i % 10}", alpha=0.2, label=f"Segment {i + 1}")

plt.title("Time Series with Change Point Detection Segments")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Simulate a time series for visualization
np.random.seed(0)
time_series = np.cumsum(np.random.randn(103))  # Random walk for time series
change_points = [0, 25, 50, 75, 103]  # Example change points from CPD
segment_length = 20

# Extract actual segments and padded/extended segments
actual_segments = []
padded_segments = []

for i in range(len(change_points) - 1):
    start = change_points[i]
    end = change_points[i + 1]

    # Actual segment
    actual_segment = time_series[start:end]
    actual_segments.append((start, end, actual_segment))

    # Calculate midpoint and extend to desired length
    midpoint = (start + end) // 2
    padded_start = max(0, midpoint - segment_length // 2)
    padded_end = min(len(time_series), padded_start + segment_length)
    padded_start = max(0, padded_end - segment_length)
    padded_segment = time_series[padded_start:padded_end]

    padded_segments.append((padded_start, padded_end, padded_segment))

# Plot time series with actual and extended segments
plt.figure(figsize=(12, 8))
plt.plot(range(len(time_series)), time_series, label="Time Series", color="blue")

for idx, (start, end, segment) in enumerate(actual_segments):
    plt.axvspan(start, end, color=f"C{idx % 10}", alpha=0.3, label=f"Actual Segment {idx + 1}")

for idx, (start, end, segment) in enumerate(padded_segments):
    plt.axvspan(start, end, color=f"C{idx % 10}", alpha=0.1, label=f"Padded Segment {idx + 1}" if idx == 0 else None)

# Add change point markers
for cp in change_points:
    plt.axvline(x=cp, color="red", linestyle="--", label="Change Point" if cp == change_points[0] else None)

plt.title("Time Series with Actual and Padded Segments")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

def visualize_segments(inputs: torch.Tensor, change_points: List[int], segments: List[torch.Tensor], segment_length: int):
    """
    Visualizes the original time series, original segments, and adjusted segments with padding.

    Args:
        inputs (torch.Tensor): Time series data of shape [seq_len, features].
        change_points (List[int]): List of change points.
        segments (List[torch.Tensor]): List of fixed-length segments.
        segment_length (int): Desired length of each segment.
    """
    time_series = inputs[:, 0].cpu().numpy()  # Visualize the first feature for simplicity
    seq_len = len(time_series)

    # Plot original time series
    plt.figure(figsize=(12, 6))
    plt.plot(range(seq_len), time_series, label="Original Time Series", color="blue")

    # Plot change points
    for cp in change_points:
        plt.axvline(x=cp, color="red", linestyle="--", label="Change Point" if cp == change_points[0] else None)

    # Plot original and adjusted segments
    for i, segment in enumerate(segments):
        start = max(0, change_points[i] - segment_length // 2)
        end = min(seq_len, start + segment_length)
        adjusted_segment = segment.cpu().numpy()[:, 0]  # First feature for simplicity

        # Plot adjusted segment
        plt.plot(range(start, end), adjusted_segment, label=f"Adjusted Segment {i + 1}", alpha=0.7)

    plt.title("Time Series Segmentation with Adjusted Padding")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
