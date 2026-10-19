import numpy as np
from parcs_v3.forward_model import generate_deltaS_sequence
from parcs_v3.adversarial import make_adversarial_scene
from parcs_v3.temporal_tracking import run_v3_pipeline
from parcs_v3.metrics import compute_trajectory_errors
from parcs_v3.configs import SENSOR_CONFIG, SCENE_DEFAULT

def run_ablation(
    *,
    vary_d_threshold: list | None = None,
    vary_min_component_size: list | None = None,
    vary_max_component_size: list | None = None,
    n_frames: int = 25,
    dt: float = 0.1
) -> dict:
    """
    Systematically vary operator parameters and record trajectory metrics.
    """
    adv_scene = make_adversarial_scene(base_scene=SCENE_DEFAULT)
    deltaS_seq_adv, gt_traj_adv = generate_deltaS_sequence(
        n_frames=n_frames,
        dt=dt,
        scene_config=adv_scene,
        sensor_config=SENSOR_CONFIG
    )

    results = {
        "d_threshold": [],
        "min_component_size": [],
        "max_component_size": [],
        "mean_position_error": [],
        "track_continuity": [],
    }

    if vary_d_threshold is None:
        vary_d_threshold = [0.03, 0.05, 0.07]
    if vary_min_component_size is None:
        vary_min_component_size = [3, 5, 8]
    if vary_max_component_size is None:
        vary_max_component_size = [800, 1000, 1200]

    for d_th in vary_d_threshold:
        for min_cs in vary_min_component_size:
            for max_cs in vary_max_component_size:
                res = run_v3_pipeline(
                    deltaS_seq=deltaS_seq_adv,
                    gt_traj=gt_traj_adv,
                    d_threshold=d_th,
                    min_component_size=min_cs,
                    max_component_size=max_cs,
                    max_match_distance=5.0,
                    max_spatial_jump=8.0
                )

                traj_errors = compute_trajectory_errors(res["trajectories"], gt_traj_adv)

                results["d_threshold"].append(d_th)
                results["min_component_size"].append(min_cs)
                results["max_component_size"].append(max_cs)
                results["mean_position_error"].append(traj_errors["mean_position_error"])
                results["track_continuity"].append(traj_errors["track_continuity"])

    return results
