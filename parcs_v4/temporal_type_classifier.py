# parcs_v3/temporal_type_classifier.py

import numpy as np

def extract_component_time_series(trajectories: list, key: str = "strength") -> list:
    """
    For each trajectory, extract a time series of a given component attribute (e.g., 'strength').
    Returns: list of np.ndarray, one per trajectory.
    """
    series = []
    for traj in trajectories:
        vals = [p.get(key, 0.0) for p in traj]
        series.append(np.array(vals, dtype=float))
    return series


def classify_temporal_signature(
    series: np.ndarray,
    missile_growth_threshold: float = 0.02,
    drone_variance_threshold: float = 0.005,
) -> str:
    """
    Classify a trajectory based on temporal radiance signature.
    Heuristic:
        - missiles: monotonic growth with sufficient slope
        - drones: moderate variance, no strong trend
        - decoys: low variance, low mean
    """
    if len(series) < 2:
        return "decoy"

    diffs = np.diff(series)
    slope = np.mean(diffs)
    var   = np.var(series)
    mean  = np.mean(series)

    if slope >= missile_growth_threshold and mean >= 0.5 * missile_growth_threshold:
        return "missile"
    elif var >= drone_variance_threshold and mean > 0.0:
        return "drone"
    else:
        return "decoy"


def classify_trajectories_temporal(trajectories: list, key: str = "strength") -> list:
    """
    Classify each trajectory using its temporal radiance signature.
    Returns: list of labels, one per trajectory.
    """
    series_list = extract_component_time_series(trajectories, key=key)
    labels = [classify_temporal_signature(s) for s in series_list]
    return labels


def attach_temporal_types_to_trajectories(trajectories: list, key: str = "strength") -> list:
    """
    Attach 'temporal_type' to each point in each trajectory.
    """
    labels = classify_trajectories_temporal(trajectories, key=key)
    labeled_traj = []
    for traj, lab in zip(trajectories, labels):
        new_traj = []
        for p in traj:
            pp = dict(p)
            pp["temporal_type"] = lab
            new_traj.append(pp)
        labeled_traj.append(new_traj)
    return labeled_traj
