# parcs_v3/operator_fusion.py

from parcs_v3.temporal_tracking import run_v3_pipeline
from parcs_v3.type_inference import label_detections_with_types

def run_fused_operator(
    deltaS_seq,
    gt_traj,
    d_threshold: float = 0.05,
    min_component_size: int = 5,
    max_component_size: int = 1000,
    max_match_distance: float = 5.0,
    max_spatial_jump: float = 8.0,
    decoy_threshold: float = 0.6,
):
    """
    Fused v3 operator:
    - runs detection + association + tracking
    - applies type inference to detections
    Returns dict with:
        - detections_seq
        - labeled_detections_seq
        - assoc_seq
        - trajectories
    """
    results = run_v3_pipeline(
        deltaS_seq=deltaS_seq,
        gt_traj=gt_traj,
        d_threshold=d_threshold,
        min_component_size=min_component_size,
        max_component_size=max_component_size,
        max_match_distance=max_match_distance,
        max_spatial_jump=max_spatial_jump,
    )

    detections_seq = results["detections_seq"]
    labeled_detections_seq = label_detections_with_types(
        detections_seq, decoy_threshold=decoy_threshold
    )

    return {
        "detections_seq": detections_seq,
        "labeled_detections_seq": labeled_detections_seq,
        "assoc_seq": results["assoc_seq"],
        "trajectories": results["trajectories"],
    }
