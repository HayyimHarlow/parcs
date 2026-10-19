# parcs_v3/radiance_anomaly.py

import numpy as np
from parcs_v3.radiance_diagnostics import compute_radiance_stats

def detect_radiance_anomalies(
    fields_seq: list,
    z_threshold_mean: float = 3.0,
    z_threshold_std: float = 3.0,
):
    """
    Flag frames whose mean or std radiance is an outlier.
    Returns:
        - anomalies: list of dicts with frame index and stats
        - z_scores_mean: list of z-scores for mean
        - z_scores_std: list of z-scores for std
    """
    stats_per_frame = [compute_radiance_stats(f) for f in fields_seq]
    means = np.array([s["mean"] for s in stats_per_frame])
    stds  = np.array([s["std"]  for s in stats_per_frame])

    mean_mu, mean_sigma = np.mean(means), np.std(means) + 1e-12
    std_mu, std_sigma   = np.mean(stds),  np.std(stds)  + 1e-12

    z_mean = (means - mean_mu) / mean_sigma
    z_std  = (stds  - std_mu)  / std_sigma

    anomalies = []
    for k, (zm, zs) in enumerate(zip(z_mean, z_std)):
        if abs(zm) >= z_threshold_mean or abs(zs) >= z_threshold_std:
            anomalies.append({
                "frame": k,
                "mean": float(means[k]),
                "std": float(stds[k]),
                "z_mean": float(zm),
                "z_std": float(zs),
            })

    return {
        "anomalies": anomalies,
        "z_scores_mean": z_mean.tolist(),
        "z_scores_std": z_std.tolist(),
    }
