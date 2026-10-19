# parcs_v3/assoc_stabilizer.py

import numpy as np

def _centroid(obj):
    c = obj.get("centroid", None)
    if c is None:
        return None
    try:
        return float(c[0]), float(c[1])
    except Exception:
        return None


def _find_gt_index(gt_frame, gt_obj):
    target = _centroid(gt_obj)
    if target is None:
        return None
    for i, g in enumerate(gt_frame):
        if np.allclose(_centroid(g), target, atol=1e-3):
            return i
    return None


def _find_det_index(det_traj, det_obj, frame_idx):
    target = _centroid(det_obj)
    if target is None:
        return None
    for j, traj in enumerate(det_traj):
        try:
            det = traj[frame_idx]
            if np.allclose(_centroid(det), target, atol=1e-3):
                return j
        except Exception:
            pass
    return None


def stabilize_assoc_dict_frame(gt_frame, assoc_dict, det_traj, frame_idx):
    n_gt = len(gt_frame)
    assoc = [None] * n_gt

    if not isinstance(assoc_dict, dict):
        return assoc

    # Process matches
    for pair in assoc_dict.get("matches", []):
        if not isinstance(pair, dict):
            continue

        gt_obj = pair.get("gt", {})
        det_obj = pair.get("det", {})

        gt_idx = _find_gt_index(gt_frame, gt_obj)
        det_idx = _find_det_index(det_traj, det_obj, frame_idx)

        if gt_idx is not None:
            assoc[gt_idx] = det_idx

    # Process missed GTs
    for gt_obj in assoc_dict.get("missed", []):
        gt_idx = _find_gt_index(gt_frame, gt_obj)
        if gt_idx is not None:
            assoc[gt_idx] = None

    return assoc


def stabilize_assoc_dict_sequence(gt_traj, assoc_seq, det_traj):
    stabilized = []
    for frame_idx, (gt_frame, assoc_dict) in enumerate(zip(gt_traj, assoc_seq)):
        stabilized.append(
            stabilize_assoc_dict_frame(gt_frame, assoc_dict, det_traj, frame_idx)
        )
    return stabilized





