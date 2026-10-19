# parcs_v3/radiance_diagnostics.py

import numpy as np
import matplotlib.pyplot as plt

def compute_radiance_stats(field: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(field)),
        "std": float(np.std(field)),
        "min": float(np.min(field)),
        "max": float(np.max(field)),
    }

def compute_sequence_radiance_stats(fields_seq: list) -> dict:
    per_frame = [compute_radiance_stats(f) for f in fields_seq]
    means = [s["mean"] for s in per_frame]
    stds  = [s["std"]  for s in per_frame]

    return {
        "per_frame": per_frame,
        "mean_of_means": float(np.mean(means)),
        "mean_of_stds": float(np.mean(stds)),
    }

def plot_radiance_stats_over_time(fields_seq: list, title: str):
    stats = compute_sequence_radiance_stats(fields_seq)
    means = [s["mean"] for s in stats["per_frame"]]
    stds  = [s["std"]  for s in stats["per_frame"]]
    frames = np.arange(len(fields_seq))

    plt.figure(figsize=(8,4))
    plt.plot(frames, means, "-o", label="Mean Radiance")
    plt.plot(frames, stds, "-o", label="Std Radiance")
    plt.xlabel("Frame index")
    plt.ylabel("Radiance value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

