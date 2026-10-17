# parcs_v3/assoc_rewrite.py

import numpy as np

def _centroid(obj):
    c = obj.get("centroid", None)
    if c is None:
        return None
    try:
        return float(c[0]), float(c[1])
    except Exception:
        return None


def rewrite_assoc_frame(gt_frame, det_frame):
    """
    Deterministic centroid-nearest association.
    Produces canonical dict:
      {
        "matches": [ {"gt":..., "det":...}, ... ],
        "missed":  [ {"gt":...}, ... ],
        "false_positives": [ {"det":...}, ... ],
        "localization_errors": [],
        "reconstruction_errors": []
      }
    """
    out = {
        "matches": [],
        "missed": [],
        "false_positives": [],
        "localization_errors": [],
        "reconstruction_errors": []
    }

    used_det = set()

    # Match each GT to nearest detection
    for gt in gt_frame:
        gt_c = _centroid(gt)
        if gt_c is None:
            out["missed"].append({"gt": gt})
            continue

        best = None
        best_d = 1e9

        for det in det_frame:
            det_c = _centroid(det)
            if det_c is None:
                continue
            d = np.linalg.norm(np.array(gt_c) - np.array(det_c))
            if d < best_d:
                best_d = d
                best = det

        if best is None:
            out["missed"].append({"gt": gt})
        else:
            out["matches"].append({"gt": gt, "det": best})
            used_det.add(id(best))

    # False positives
    for det in det_frame:
        if id(det) not in used_det:
            out["false_positives"].append({"det": det})

    return out


def rewrite_assoc_sequence(gt_traj, det_traj):
    return [
        rewrite_assoc_frame(gt_frame, det_frame)
        for gt_frame, det_frame in zip(gt_traj, det_traj)
    ]

