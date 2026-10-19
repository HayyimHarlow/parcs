import numpy as np

def compute_trajectory_errors(trajectories, gt_traj):
    """
    Compute trajectory-level physical validation metrics.
    """

    # 1. Position error per frame
    pos_errors = []

    for traj in trajectories:
        for p in traj:
            t = p["t"]
            det_x, det_y = p["centroid"]

            # nearest ground-truth target at time t
            gt_positions = [gt["centroid"] for gt in gt_traj[t]]
            dists = [np.sqrt((det_x - gx)**2 + (det_y - gy)**2) for gx, gy in gt_positions]
            pos_errors.append(min(dists))

    mean_position_error = np.mean(pos_errors) if pos_errors else 0.0

    # 2. Track continuity
    # continuity = fraction of frames where trajectory has a detection
    continuities = []
    for traj in trajectories:
        frames_present = {p["t"] for p in traj}
        continuity = len(frames_present) / len(gt_traj)
        continuities.append(continuity)

    track_continuity = np.mean(continuities) if continuities else 0.0

    return {
        "mean_position_error": mean_position_error,
        "track_continuity": track_continuity
    }
