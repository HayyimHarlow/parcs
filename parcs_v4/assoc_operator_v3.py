# parcs_v3/assoc_operator_v3.py

import numpy as np

def run_association_operator_v3(gt_traj, assoc_dict_seq):
    """
    Rewritten v3 operator that consumes dict-style association frames
    and produces deterministic integer association lists.

    Input:
        gt_traj[f]          : list of GT objects for frame f
        assoc_dict_seq[f]   : dict with keys:
                              'matches', 'missed', 'false_positives',
                              'localization_errors', 'reconstruction_errors'

    Output:
        assoc_seq[f]        : list of length len(gt_traj[f]),
                              each entry is either an integer det index or None
    """

    assoc_seq = []

    for f_idx, (gt_frame, assoc_dict) in enumerate(zip(gt_traj, assoc_dict_seq)):
        n_gt = len(gt_frame)

        # Output frame: one entry per GT
        assoc = [None] * n_gt

        # Extract detections from dict-style matches
        # Each match is: {'gt': {...}, 'det': {...}}
        dets = []
        for m in assoc_dict.get("matches", []):
            dets.append(m["det"])

        n_det = len(dets)

        # Build cost matrix (centroid distance)
        cost = np.full((n_gt, n_det), np.inf, dtype=float)

        for i, gt in enumerate(gt_frame):
            gx, gy = gt["centroid"]
            for j, det in enumerate(dets):
                dx, dy = det["centroid"]
                cost[i, j] = np.hypot(gx - dx, gy - dy)

        # Deterministic greedy matching
        used_det = set()
        for i in range(n_gt):
            best_j = None
            best_cost = float("inf")

            for j in range(n_det):
                if j in used_det:
                    continue
                if cost[i, j] < best_cost:
                    best_cost = cost[i, j]
                    best_j = j

            # Threshold: association only if centroid distance is reasonable
            if best_j is not None and best_cost < 10.0:
                assoc[i] = best_j
                used_det.add(best_j)

        assoc_seq.append(assoc)

    return assoc_seq

# ============================================================
# v3 DICT-BASED ASSOCIATION OPERATOR (ABSTRACT, TRACE-READY)
# ============================================================

def run_association_operator_v3_dict(
    gt_traj,
    traj_temporal,
    *,
    match_policy,
    error_metrics=None,
    invariant_checks=None,
    perturbation=None,
    trace=None,
    rng=None,
):
    """
    Abstract dict-based association operator for v3.

    This operator:
      - emits full association trace (if trace is provided)
      - supports operator invariants (via invariant_checks)
      - supports perturbation stability tests (via perturbation)
      - never exposes physics or sensor internals
      - matches your existing assoc_seq dict structure exactly

    Parameters
    ----------
    gt_traj : opaque GT trajectory container
    traj_temporal : opaque detection trajectory container

    match_policy : required strategy object
        Must expose:
            associate(gt_objs, det_objs, *, rng=None) -> (matches, missed, false_pos)

    error_metrics : optional strategy object
        Must expose:
            localization(match) -> float
            reconstruction(match) -> float

    invariant_checks : optional strategy object
        Must expose:
            check_frame(frame_idx, frame_dict) -> None

    perturbation : optional strategy object
        Must expose:
            apply(gt_objs, det_objs, *, rng=None) -> (gt_objs_p, det_objs_p)

    trace : optional trace collector
        Must expose:
            record(frame_idx, gt_objs, det_objs, matches, missed, false_pos, loc_errs, rec_errs)

    rng : optional randomness source

    Returns
    -------
    assoc_seq : list[dict]
        One dict per frame, matching your existing assoc_seq structure.
    """

    n_frames = len(traj_temporal)
    assoc_seq = []

    for f_idx in range(n_frames):

        # ------------------------------------------------------------
        # 1. Abstract access to GT and detections
        # ------------------------------------------------------------
        gt_objs = gt_traj.get_frame_objects(f_idx)
        det_objs = traj_temporal.get_frame_detections(f_idx)

        # ------------------------------------------------------------
        # 2. Optional perturbation (stability testing)
        # ------------------------------------------------------------
        if perturbation is not None:
            gt_objs, det_objs = perturbation.apply(gt_objs, det_objs, rng=rng)

        # ------------------------------------------------------------
        # 3. Abstract association (match_policy)
        # ------------------------------------------------------------
        matches, missed, false_pos = match_policy.associate(
            gt_objs, det_objs, rng=rng
        )

        # ------------------------------------------------------------
        # 4. Error metrics (optional)
        # ------------------------------------------------------------
        loc_errs = []
        rec_errs = []
        if error_metrics is not None:
            for m in matches:
                loc_errs.append(error_metrics.localization(m))
                rec_errs.append(error_metrics.reconstruction(m))

        # ------------------------------------------------------------
        # 5. Build dict-frame (your exact structure)
        # ------------------------------------------------------------
        frame_dict = {
            "matches": [
                {
                    "gt": {
                        "id": m.gt.id,
                        "type": m.gt.type,
                        "centroid": m.gt.centroid,
                        "radius": m.gt.radius,
                    },
                    "det": {
                        "id": m.det.id,
                        "centroid": m.det.centroid,
                        "radius": m.det.radius,
                        "strength": m.det.strength,
                        "mask": m.det.mask,
                    },
                }
                for m in matches
            ],
            "missed": [
                {
                    "gt": {
                        "id": g.id,
                        "type": g.type,
                        "centroid": g.centroid,
                        "radius": g.radius,
                    }
                }
                for g in missed
            ],
            "false_positives": [
                {
                    "det": {
                        "id": d.id,
                        "centroid": d.centroid,
                        "radius": d.radius,
                        "strength": d.strength,
                        "mask": d.mask,
                    }
                }
                for d in false_pos
            ],
            "localization_errors": loc_errs,
            "reconstruction_errors": rec_errs,
        }

        # ------------------------------------------------------------
        # 6. Operator invariants (optional)
        # ------------------------------------------------------------
        if invariant_checks is not None:
            invariant_checks.check_frame(f_idx, frame_dict)

        # ------------------------------------------------------------
        # 7. Full trace emission (optional)
        # ------------------------------------------------------------
        if trace is not None:
            trace.record(
                frame_idx=f_idx,
                gt_objs=gt_objs,
                det_objs=det_objs,
                matches=matches,
                missed=missed,
                false_pos=false_pos,
                loc_errs=loc_errs,
                rec_errs=rec_errs,
            )

        assoc_seq.append(frame_dict)

    return assoc_seq


# ------------------------------------------------------------
# Minimal match container (abstract)
# ------------------------------------------------------------
class SimpleMatch:
    def __init__(self, gt, det):
        self.gt = gt
        self.det = det
        

