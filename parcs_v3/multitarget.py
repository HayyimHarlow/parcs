import numpy as np

def build_multi_target_confusion(gt_traj: list, assoc_seq: list) -> dict:
    """
    Build a simple multi-target confusion matrix over time.
    Rows: ground-truth types
    Cols: detected types (inferred from association index)
    """
    # Collect all GT types
    gt_types = sorted({t["type"] for frame in gt_traj for t in frame})
    det_types = gt_types + ["decoy", "none"]

    type_to_idx_gt = {t: i for i, t in enumerate(gt_types)}
    type_to_idx_det = {t: i for i, t in enumerate(det_types)}

    cm = np.zeros((len(gt_types), len(det_types)), dtype=int)

    for k, assoc in enumerate(assoc_seq):
        gt_frame = gt_traj[k]
        matches = assoc["matches"]
        matched_gt = {m["gt"]["centroid"]: m["det"] for m in matches}

        # For each GT target, see what it was matched to
        for gt in gt_frame:
            gt_type = gt["type"]
            i = type_to_idx_gt[gt_type]

            det = matched_gt.get(gt["centroid"], None)
            if det is None:
                j = type_to_idx_det["none"]
            else:
                # crude heuristic: decoy vs non-decoy by strength
                j_type = "decoy" if gt_type != "decoy" and det.get("strength", 0) < 0.6 else gt_type
                j = type_to_idx_det.get(j_type, type_to_idx_det["none"])

            cm[i, j] += 1

    return {
        "gt_types": gt_types,
        "det_types": det_types,
        "matrix": cm
    }


def print_confusion_matrix(cm_dict: dict):
    gt_types = cm_dict["gt_types"]
    det_types = cm_dict["det_types"]
    cm = cm_dict["matrix"]

    header = "GT \\ Det | " + " | ".join(f"{t:>8}" for t in det_types)
    print(header)
    print("-" * len(header))

    for i, gt in enumerate(gt_types):
        row_vals = " | ".join(f"{cm[i, j]:>8}" for j in range(len(det_types)))
        print(f"{gt:>9} | {row_vals}")
