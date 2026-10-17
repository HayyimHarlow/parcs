import numpy as np
import pytest

from parcs_v3.forward_model import (
    scene_to_radiance_field,
    radiance_to_deltaS,
    generate_deltaS_sequence
)

from parcs_v3.temporal_tracking import (
    run_v3_pipeline
)

from parcs_v3.metrics import compute_trajectory_errors
from parcs_v3.configs import SENSOR_CONFIG, SCENE_DEFAULT


def test_forward_model_shapes():
    field = scene_to_radiance_field(t=0.0)
    assert field.shape == (SENSOR_CONFIG["grid_size"], SENSOR_CONFIG["grid_size"])

    deltaS = radiance_to_deltaS(field)
    assert deltaS.shape == field.shape


def test_generate_deltaS_sequence():
    deltaS_seq, gt_traj = generate_deltaS_sequence(5, 0.1)
    assert len(deltaS_seq) == 5
    assert len(gt_traj) == 5
    assert isinstance(gt_traj[0], list)


def test_run_v3_pipeline_structure():
    deltaS_seq, gt_traj = generate_deltaS_sequence(5, 0.1)
    results = run_v3_pipeline(
        deltaS_seq=deltaS_seq,
        gt_traj=gt_traj,
        d_threshold=0.05,
        min_component_size=5,
        max_component_size=1000,
        max_match_distance=5.0,
        max_spatial_jump=8.0
    )

    assert "detections_seq" in results
    assert "assoc_seq" in results
    assert "trajectories" in results

    assert len(results["detections_seq"]) == 5
    assert len(results["assoc_seq"]) == 5
    assert len(results["trajectories"]) > 0


def test_trajectory_error_metrics():
    deltaS_seq, gt_traj = generate_deltaS_sequence(5, 0.1)
    results = run_v3_pipeline(
        deltaS_seq=deltaS_seq,
        gt_traj=gt_traj,
        d_threshold=0.05,
        min_component_size=5,
        max_component_size=1000,
        max_match_distance=5.0,
        max_spatial_jump=8.0
    )

    traj_errors = compute_trajectory_errors(results["trajectories"], gt_traj)
    assert "mean_position_error" in traj_errors
    assert "track_continuity" in traj_errors
    assert traj_errors["mean_position_error"] >= 0.0
