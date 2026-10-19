import numpy as np

# ============================================================
# PARCS v4 — Localization Error
# ============================================================

def compute_localization_error(results):
    """
    PARCS v4 Localization Error

    Computes the mean centroid distance between inferred object
    trajectories and ground-truth trajectories across all frames.

    This metric quantifies geometric stability under adversarial
    perturbation.

    Returns:
        float: mean localization error (>= 0)
    """

    total_error = 0.0
    count = 0

    for frame in results.frames:
        gt_frame = frame.gt_objects
        det_frame = frame.detections

        for gt_obj in gt_frame:
            # nearest detection match
            best_det = None
            best_dist = float("inf")

            for det in det_frame:
                dist = gt_obj.centroid_distance(det)
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            # accumulate error
            if best_det is not None:
                total_error += best_dist
                count += 1

    if count == 0:
        return 0.0

    return total_error / count


# ============================================================
# PARCS v4 — Confusion Matrix (Operator-Level)
# ============================================================

def compute_confusion_matrix(results):
    """
    PARCS v4 Confusion Matrix (Operator-Level)

    This confusion matrix measures semantic robustness using the
    operator's own inferred semantic types, not GT-index matching.

    Rows = ground-truth semantic class
    Columns = inferred semantic class

    Semantic taxonomy:
        0 = missile
        1 = drone
        2 = decoy
        3 = none

    This operator-level confusion matrix is distinct from the
    TypeConfusionEvaluator (which performs nearest-neighbor matching).
    """

    # 4 semantic classes
    cm = np.zeros((4, 4), dtype=int)

    for frame in results.frames:
        gt_frame = frame.gt_objects
        det_frame = frame.detections

        for gt_obj in gt_frame:
            gt_type = _encode_semantic(gt_obj.semantic_type)

            # operator-level inference: find best detection
            best_det = None
            best_dist = float("inf")

            for det in det_frame:
                dist = gt_obj.centroid_distance(det)
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            if best_det is None:
                inf_type = 3  # none
            else:
                inf_type = _encode_semantic(best_det.semantic_type)

            cm[gt_type, inf_type] += 1

    return cm


def _encode_semantic(sem_type):
    """
    Shared semantic encoder for confusion-matrix metrics.
    """
    if sem_type == "missile":
        return 0
    if sem_type == "drone":
        return 1
    if sem_type == "decoy":
        return 2
    return 3  # none / unknown / dropout


# ============================================================
# PARCS v4 — Identity Stability
# ============================================================

def compute_identity_stability(results):
    """
    Formal identity-stability metric for PARCS v4.

    Definition:
        Identity stability is defined as the fraction of frames in which
        the assigned identity for each tracked object remains consistent
        with its previous assignment, excluding frames where the object
        is not detected.

    Mathematically:
        stability = 1 - (identity_switches / valid_frames)

    Where:
        identity_switches = number of frames where an object's identity
                            changes relative to the previous frame.
        valid_frames = number of frames where the object is detected
                       and identity assignment is possible.

    This metric is explicit, reproducible, and does not rely on
    variance-to-percentage conversion.
    """

    identity_switches = 0
    valid_frames = 0

    for obj_id, traj in results.tracks.items():
        prev_identity = None
        for frame in traj:
            if frame.identity is None:
                continue
            valid_frames += 1
            if prev_identity is not None and frame.identity != prev_identity:
                identity_switches += 1
            prev_identity = frame.identity

    if valid_frames == 0:
        return 0.0

    return 1.0 - (identity_switches / valid_frames)

# ============================================================
# PARCS v4 — Detection Metrics
# ============================================================

def compute_detection_metrics(results):
    """
    PARCS v4 Detection Metrics

    Computes:
        TP = true positives
        FP = false positives
        FN = false negatives

    Definitions:
        - A true positive occurs when a detection is matched to a GT object
          via nearest-neighbor centroid distance.

        - A false positive occurs when a detection has no GT object within
          the association radius.

        - A false negative occurs when a GT object has no detection within
          the association radius.

    This metric is operator-level and consistent with the v4 evidence boundary.
    """

    TP = 0
    FP = 0
    FN = 0

    max_dist = getattr(results.cfg, "max_assoc_distance", 50.0)

    for frame in results.frames:
        gt_frame = frame.gt_objects
        det_frame = frame.detections

        # Track which detections have been matched
        matched_dets = set()

        # True positives + false negatives
        for gt_obj in gt_frame:
            best_det = None
            best_dist = float("inf")

            for det_idx, det in enumerate(det_frame):
                dist = gt_obj.centroid_distance(det)
                if dist < best_dist:
                    best_dist = dist
                    best_det = det_idx

            if best_det is not None and best_dist <= max_dist:
                TP += 1
                matched_dets.add(best_det)
            else:
                FN += 1

        # False positives
        for det_idx, det in enumerate(det_frame):
            if det_idx not in matched_dets:
                # check if any GT is close enough
                nearest_gt_dist = float("inf")
                for gt_obj in gt_frame:
                    dist = gt_obj.centroid_distance(det)
                    nearest_gt_dist = min(nearest_gt_dist, dist)

                if nearest_gt_dist > max_dist:
                    FP += 1

    return TP, FP, FN


