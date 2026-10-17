"""
PARCS Synthetic Validation v2
Ground-truth-independent detection, explicit association,
noise trials, and quantitative metrics.

This module is designed to be imported both by the v2 notebook
and by the v2 unit test suite.
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_opening, binary_closing


# ============================================================
# Synthetic ΔS generator (v2 baseline)
# ============================================================

def generate_synthetic_scene_v2(seed: int,
                                noise_level: float,
                                grid_size: int = 64,
                                targets: list = None) -> tuple:
    """
    Generate synthetic ΔS frame and ground-truth targets.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    noise_level : float
        Noise amplitude.
    grid_size : int
        Size of the square grid.
    targets : list of dict or None
        Optional explicit target list:
            {"centroid": (x, y), "radius": r, "amplitude": A}

    Returns
    -------
    deltaS_frame : np.ndarray
    ground_truth : list of dict
    """
    rng = np.random.default_rng(seed)
    deltaS = np.zeros((grid_size, grid_size), dtype=float)

    if targets is None:
        targets = [
            {"centroid": (20, 20), "radius": 4.0, "amplitude": 1.0},
            {"centroid": (40, 30), "radius": 3.0, "amplitude": 0.8},
            {"centroid": (30, 45), "radius": 5.0, "amplitude": 1.2},
        ]

    ys, xs = np.indices(deltaS.shape)

    for t in targets:
        cx, cy = t["centroid"]
        r = t["radius"]
        A = t["amplitude"]

        dist2 = (xs - cx)**2 + (ys - cy)**2
        deltaS += A * np.exp(-dist2 / (2 * r**2))

    deltaS += noise_level * rng.normal(size=deltaS.shape)

    ground_truth = [
        {"centroid": t["centroid"], "radius": t["radius"]}
        for t in targets
    ]

    return deltaS, ground_truth


# ============================================================
# D operator (gradient magnitude)
# ============================================================

def compute_D(deltaS_frame: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude D from ΔS.
    """
    gy, gx = np.gradient(deltaS_frame)
    D = np.sqrt(gx**2 + gy**2)
    return D


# ============================================================
# A operator v2 (ground-truth-independent detection)
# ============================================================

def operator_A_v2(deltaS_frame: np.ndarray,
                  D_frame: np.ndarray,
                  d_threshold: float,
                  min_component_size: int,
                  max_component_size: int) -> list:
    """
    Ground-truth-independent admissibility operator.
    """
    mask = (D_frame >= d_threshold)

    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = binary_closing(mask, structure=np.ones((3, 3)))

    labeled, num_labels = ndimage.label(mask)

    components = []

    for label in range(1, num_labels + 1):
        comp_mask = (labeled == label)
        size = comp_mask.sum()

        if size < min_component_size or size > max_component_size:
            continue

        ys, xs = np.where(comp_mask)
        centroid = (xs.mean(), ys.mean())
        radius = np.sqrt(size / np.pi)
        strength = deltaS_frame[comp_mask].mean()

        components.append({
            "centroid": centroid,
            "radius": radius,
            "strength": strength,
            "mask": comp_mask
        })

    return components


# ============================================================
# Association cost matrix + matching
# ============================================================

def build_cost_matrix(ground_truth: list,
                      detected: list,
                      max_match_distance: float) -> np.ndarray:
    """
    Construct cost matrix for association.
    """
    n_gt = len(ground_truth)
    n_det = len(detected)
    cost = np.zeros((n_gt, n_det))

    for i, gt in enumerate(ground_truth):
        gx, gy = gt["centroid"]
        for j, det in enumerate(detected):
            dx, dy = det["centroid"]
            dist = np.sqrt((gx - dx)**2 + (gy - dy)**2)
            cost[i, j] = dist if dist <= max_match_distance else 1e6

    return cost


def associate_targets_v2(ground_truth: list,
                         detected: list,
                         max_match_distance: float) -> dict:
    """
    One-to-one association between ground-truth targets and detected components.
    """
    cost = build_cost_matrix(ground_truth, detected, max_match_distance)

    if cost.size == 0:
        return {
            "matches": [],
            "missed": ground_truth.copy(),
            "false_positives": detected.copy(),
            "localization_errors": [],
            "reconstruction_errors": []
        }

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    missed = []
    false_positives = []

    localization_errors = []
    reconstruction_errors = []

    matched_gt_indices = set()
    matched_det_indices = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= 1e6:
            continue

        gt = ground_truth[i]
        det = detected[j]

        gx, gy = gt["centroid"]
        dx, dy = det["centroid"]
        loc_err = np.sqrt((gx - dx)**2 + (gy - dy)**2)
        rec_err = abs(det["radius"] - gt["radius"])

        matches.append({"gt": gt, "det": det})
        localization_errors.append(loc_err)
        reconstruction_errors.append(rec_err)

        matched_gt_indices.add(i)
        matched_det_indices.add(j)

    for i, gt in enumerate(ground_truth):
        if i not in matched_gt_indices:
            missed.append(gt)

    for j, det in enumerate(detected):
        if j not in matched_det_indices:
            false_positives.append(det)

    return {
        "matches": matches,
        "missed": missed,
        "false_positives": false_positives,
        "localization_errors": localization_errors,
        "reconstruction_errors": reconstruction_errors
    }


# ============================================================
# Metric computation
# ============================================================

def compute_metrics(assoc_result: dict,
                    n_ground_truth: int,
                    n_detected: int) -> dict:
    """
    Compute detection and association metrics.
    """
    tp = len(assoc_result["matches"])
    fn = len(assoc_result["missed"])
    fp = len(assoc_result["false_positives"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    assoc_acc = tp / max(n_ground_truth, n_detected) if max(n_ground_truth, n_detected) > 0 else 0.0

    loc_err_mean = (np.mean(assoc_result["localization_errors"])
                    if assoc_result["localization_errors"] else 0.0)
    rec_err_mean = (np.mean(assoc_result["reconstruction_errors"])
                    if assoc_result["reconstruction_errors"] else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "association_accuracy": assoc_acc,
        "mean_localization_error": loc_err_mean,
        "mean_reconstruction_error": rec_err_mean
    }


# ============================================================
# Noise trial harness
# ============================================================

def run_noise_trials_v2(scene_generator_fn,
                        *,
                        noise_levels: list,
                        seeds: list,
                        d_threshold: float,
                        min_component_size: int,
                        max_component_size: int,
                        max_match_distance: float) -> dict:
    """
    Execute independent noise trials for robustness evaluation.
    """
    results = {}

    for nl in noise_levels:
        results[nl] = {}
        for seed in seeds:
            deltaS, ground_truth = scene_generator_fn(seed, nl)
            D = compute_D(deltaS)
            detected = operator_A_v2(deltaS, D,
                                     d_threshold,
                                     min_component_size,
                                     max_component_size)
            assoc = associate_targets_v2(ground_truth, detected, max_match_distance)
            metrics = compute_metrics(assoc, len(ground_truth), len(detected))

            results[nl][seed] = {
                **metrics,
                "localization_errors": assoc["localization_errors"],
                "reconstruction_errors": assoc["reconstruction_errors"]
            }

    return results
