import numpy as np
import pytest

from parcs_v2 import (
    generate_synthetic_scene_v2,
    compute_D,
    operator_A_v2,
    build_cost_matrix,
    associate_targets_v2,
    compute_metrics,
    run_noise_trials_v2
)


# ------------------------------------------------------------
# Synthetic scene tests
# ------------------------------------------------------------

def test_generate_synthetic_scene_v2_shapes():
    deltaS, gt = generate_synthetic_scene_v2(seed=0, noise_level=0.1, grid_size=64)
    assert deltaS.shape == (64, 64)
    assert isinstance(gt, list)
    assert len(gt) > 0
    for t in gt:
        assert "centroid" in t and "radius" in t


# ------------------------------------------------------------
# D operator tests
# ------------------------------------------------------------

def test_compute_D_nonnegative():
    deltaS = np.random.rand(32, 32)
    D = compute_D(deltaS)
    assert D.shape == deltaS.shape
    assert np.all(D >= 0.0)


# ------------------------------------------------------------
# A operator tests
# ------------------------------------------------------------

def test_operator_A_v2_basic_detection():
    deltaS, gt = generate_synthetic_scene_v2(seed=1, noise_level=0.0, grid_size=64)
    D = compute_D(deltaS)
    components = operator_A_v2(deltaS, D,
                               d_threshold=0.05,
                               min_component_size=5,
                               max_component_size=1000)
    assert isinstance(components, list)
    assert len(components) > 0
    for c in components:
        assert "centroid" in c
        assert "radius" in c
        assert "strength" in c
        assert "mask" in c
        assert c["mask"].dtype == bool


# ------------------------------------------------------------
# Association tests
# ------------------------------------------------------------

def test_build_cost_matrix_dimensions():
    gt = [{"centroid": (10.0, 10.0), "radius": 3.0},
          {"centroid": (20.0, 20.0), "radius": 4.0}]
    det = [{"centroid": (11.0, 10.5), "radius": 3.2, "strength": 1.0, "mask": None},
           {"centroid": (19.5, 19.0), "radius": 4.1, "strength": 0.9, "mask": None}]
    cost = build_cost_matrix(gt, det, max_match_distance=5.0)
    assert cost.shape == (2, 2)
    assert np.all(cost >= 0.0)


def test_associate_targets_v2_matches_and_errors():
    gt = [{"centroid": (10.0, 10.0), "radius": 3.0}]
    det = [{"centroid": (10.5, 9.5), "radius": 3.2, "strength": 1.0, "mask": None}]
    result = associate_targets_v2(gt, det, max_match_distance=5.0)
    assert len(result["matches"]) == 1
    assert len(result["missed"]) == 0
    assert len(result["false_positives"]) == 0
    assert len(result["localization_errors"]) == 1
    assert len(result["reconstruction_errors"]) == 1


# ------------------------------------------------------------
# Metrics tests
# ------------------------------------------------------------

def test_compute_metrics_consistency():
    gt = [{"centroid": (10.0, 10.0), "radius": 3.0}]
    det = [{"centroid": (10.5, 9.5), "radius": 3.2, "strength": 1.0, "mask": None}]
    assoc = associate_targets_v2(gt, det, max_match_distance=5.0)
    metrics = compute_metrics(assoc, n_ground_truth=len(gt), n_detected=len(det))
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["mean_localization_error"] >= 0.0
    assert metrics["mean_reconstruction_error"] >= 0.0


# ------------------------------------------------------------
# Noise trial harness tests
# ------------------------------------------------------------

def test_run_noise_trials_v2_structure():
    noise_levels = [0.0, 0.1]
    seeds = [1, 2]

    results = run_noise_trials_v2(
        scene_generator_fn=generate_synthetic_scene_v2,
        noise_levels=noise_levels,
        seeds=seeds,
        d_threshold=0.05,
        min_component_size=5,
        max_component_size=1000,
        max_match_distance=5.0
    )

    assert set(results.keys()) == set(noise_levels)
    for nl in noise_levels:
        assert set(results[nl].keys()) == set(seeds)
        for seed in seeds:
            r = results[nl][seed]
            for key in ["precision", "recall", "f1",
                        "association_accuracy",
                        "mean_localization_error",
                        "mean_reconstruction_error"]:
                assert key in r
            assert isinstance(r["localization_errors"], list)
            assert isinstance(r["reconstruction_errors"], list)
