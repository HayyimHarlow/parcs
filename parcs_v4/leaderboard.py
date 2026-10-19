import numpy as np
from parcs_v3.forward_model import generate_deltaS_sequence
from parcs_v3.adversarial import make_adversarial_scene
from parcs_v3.temporal_tracking import run_v3_pipeline
from parcs_v3.metrics import compute_trajectory_errors
from parcs_v3.configs import SENSOR_CONFIG, SCENE_DEFAULT

def evaluate_operator(
    operator_name: str,
    *,
    d_threshold: float,
    min_component_size: int,
    max_component_size: int,
    max_match_distance: float,
    max_spatial_jump: float,
    n_frames: int = 25,
    dt: float = 0.1
) -> dict:
    """
    Evaluate a given operator configuration on an adversarial scene.
    """
    adv_scene = make_adversarial_scene(base_scene=SCENE_DEFAULT)
    deltaS_seq_adv, gt_traj_adv = generate_deltaS_sequence(
        n_frames=n_frames,
        dt=dt,
        scene_config=adv_scene,
        sensor_config=SENSOR_CONFIG
    )

    results_adv = run_v3_pipeline(
        deltaS_seq=deltaS_seq_adv,
        gt_traj=gt_traj_adv,
        d_threshold=d_threshold,
        min_component_size=min_component_size,
        max_component_size=max_component_size,
        max_match_distance=max_match_distance,
        max_spatial_jump=max_spatial_jump
    )

    traj_errors_adv = compute_trajectory_errors(results_adv["trajectories"], gt_traj_adv)

    return {
        "operator": operator_name,
        "mean_position_error": traj_errors_adv["mean_position_error"],
        "track_continuity": traj_errors_adv["track_continuity"],
        "n_trajectories": len(results_adv["trajectories"])
    }


def build_adversarial_leaderboard(configs: list) -> list:
    """
    configs: list of dicts, each with keys:
        - name
        - d_threshold
        - min_component_size
        - max_component_size
        - max_match_distance
        - max_spatial_jump
    Returns: sorted list of results (best continuity, lowest error).
    """
    results = []
    for cfg in configs:
        r = evaluate_operator(
            operator_name=cfg["name"],
            d_threshold=cfg["d_threshold"],
            min_component_size=cfg["min_component_size"],
            max_component_size=cfg["max_component_size"],
            max_match_distance=cfg["max_match_distance"],
            max_spatial_jump=cfg["max_spatial_jump"],
        )
        results.append(r)

    # sort by failure mode: highest continuity, lowest position error
    results_sorted = sorted(
        results,
        key=lambda x: (-x["track_continuity"], x["mean_position_error"])
    )
    return results_sorted


def print_leaderboard(results_sorted: list):
    print("Adversarial Leaderboard (v3)")
    print("Rank | Operator        | MeanPosErr | Continuity | #Traj")
    print("--------------------------------------------------------")
    for i, r in enumerate(results_sorted, start=1):
        print(f"{i:>4} | {r['operator']:<14} | {r['mean_position_error']:>10.3f} | "
              f"{r['track_continuity']:>10.3f} | {r['n_trajectories']:>5}")
