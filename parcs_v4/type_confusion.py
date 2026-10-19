# parcs_v3/type_confusion.py

import numpy as np
from collections import defaultdict

def extract_joint_type(det_point: dict) -> str:
    sem = det_point.get("semantic_type", "unknown")
    tmp = det_point.get("temporal_type", "unknown")
    return sem if sem == tmp else f"{sem}_{tmp}"


def _normalize_det_index(det_idx):
    if det_idx is None:
        return None
    if isinstance(det_idx, int):
        return det_idx if det_idx >= 0 else None
    if isinstance(det_idx, float):
        return int(det_idx) if det_idx.is_integer() and det_idx >= 0 else None
    if isinstance(det_idx, str):
        try:
            val = int(det_idx)
            return val if val >= 0 else None
        except Exception:
            return None
    return None


def _force_list(x):
    """
    Convert assoc_frame or gt_frame into a list.
    Handles:
        - list
        - tuple
        - numpy array
        - dict (values)
        - pandas Series
        - slice objects (treated as empty)
        - any iterable
        - non-iterables
    """
    if x is None:
        return []

    if isinstance(x, slice):
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, tuple):
        return list(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, dict):
        try:
            return list(x.values())
        except Exception:
            return []

    if hasattr(x, "values"):
        try:
            return list(x.values)
        except Exception:
            pass

    try:
        return list(x)
    except Exception:
        return []


def _sanitize_gt_frame(gt_frame_raw):
    """
    Ensure GT frame is a list of dicts.
    Replace slice or malformed entries with {'type': 'none'}.
    """
    gt_frame = _force_list(gt_frame_raw)
    sanitized = []
    for item in gt_frame:
        if isinstance(item, dict):
            sanitized.append(item)
        else:
            sanitized.append({"type": "none"})
    return sanitized


def build_type_confusion_matrix(gt_traj, assoc_seq, det_traj):
    matrix = defaultdict(lambda: defaultdict(int))
    joint_labels = set()

    for frame_idx, (gt_frame_raw, assoc_frame_raw) in enumerate(zip(gt_traj, assoc_seq)):

        # Sanitize GT frame
        gt_frame = _sanitize_gt_frame(gt_frame_raw)

        # Sanitize assoc frame
        assoc_frame = _force_list(assoc_frame_raw)

        # Empty GT frame
        if len(gt_frame) == 0:
            for _ in assoc_frame:
                matrix["none"]["none"] += 1
            joint_labels.add("none")
            continue

        n_gt = len(gt_frame)
        n_assoc = len(assoc_frame)

        # Truncate extra detections
        if n_assoc > n_gt:
            assoc_frame = assoc_frame[:n_gt]

        # Pad missing detections
        if n_assoc < n_gt:
            assoc_frame = assoc_frame + [None] * (n_gt - n_assoc)

        # Now assoc_frame and gt_frame have same length
        for gt_idx, raw_det_idx in enumerate(assoc_frame):

            gt_type = gt_frame[gt_idx].get("type", "none")
            det_idx = _normalize_det_index(raw_det_idx)

            if det_idx is None:
                matrix[gt_type]["none"] += 1
                joint_labels.add("none")
                continue

            try:
                det_point = det_traj[det_idx][frame_idx]
            except Exception:
                matrix[gt_type]["none"] += 1
                joint_labels.add("none")
                continue

            joint_type = extract_joint_type(det_point)
            matrix[gt_type][joint_type] += 1
            joint_labels.add(joint_type)

    return {
        "matrix": matrix,
        "labels": sorted(joint_labels),
    }


def print_type_confusion_matrix(result):
    matrix = result["matrix"]
    labels = result["labels"]

    header = "GT \\ Det | " + " | ".join(f"{lab:>12}" for lab in labels)
    print(header)
    print("-" * len(header))

    for gt_type, row in matrix.items():
        row_str = f"{gt_type:>8} | " + " | ".join(f"{row.get(lab, 0):>12}" for lab in labels)
        print(row_str)



