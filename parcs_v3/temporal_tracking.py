# parcsv3/temporal_tracking.py

import numpy as np
from parcs_v2 import compute_D, operator_A_v2, associate_targets_v2

def associate_across_frames(detections_seq: list,
                            max_spatial_jump: float) -> list:
    """
    Link detections across frames into trajectories.
    detections_seq: list over time, each element = list of components (dicts).
    """
    trajectories = []  # each trajectory: list of {"t": k, "centroid": (x,y), "radius": r}

    # simple nearest-neighbor tracker
    for k, dets in enumerate(detections_seq):
        if k == 0:
            for d in dets:
                trajectories.append([{"t": k, "centroid": d["centroid"], "radius": d["radius"]}])
            continue

        prev_dets = [traj[-1] for traj in trajectories]
        used = [False] * len(dets)

        for traj in trajectories:
            px, py = traj[-1]["centroid"]
            best_j = None
            best_dist = None

            for j, d in enumerate(dets):
                if used[j]:
                    continue
                dx, dy = d["centroid"]
                dist = np.sqrt((px - dx)**2 + (py - dy)**2)
                if dist <= max_spatial_jump and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                used[best_j] = True
                d = dets[best_j]
                traj.append({"t": k, "centroid": d["centroid"], "radius": d["radius"]})

        # new trajectories for unmatched detections
        for j, d in enumerate(dets):
            if not used[j]:
                trajectories.append([{"t": k, "centroid": d["centroid"], "radius": d["radius"]}])

    return trajectories


def run_v3_pipeline(deltaS_seq: list,
                    gt_traj: list,
                    d_threshold: float,
                    min_component_size: int,
                    max_component_size: int,
                    max_match_distance: float,
                    max_spatial_jump: float) -> dict:
    """
    Full v3 pipeline: ΔS sequence → detections → per-frame association → trajectories.
    """
    detections_seq = []
    assoc_seq = []

    for k, deltaS in enumerate(deltaS_seq):
        D = compute_D(deltaS)
        dets = operator_A_v2(deltaS, D,
                             d_threshold,
                             min_component_size,
                             max_component_size)
        detections_seq.append(dets)

        assoc = associate_targets_v2(gt_traj[k], dets, max_match_distance)
        assoc_seq.append(assoc)

    trajectories = associate_across_frames(detections_seq, max_spatial_jump)

    return {
        "detections_seq": detections_seq,
        "assoc_seq": assoc_seq,
        "trajectories": trajectories
    }
