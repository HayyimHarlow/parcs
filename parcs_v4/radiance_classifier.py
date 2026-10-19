# parcs_v3/radiance_classifier.py

import numpy as np

def classify_component_radiance(
    strength: float,
    radius: float,
    missile_strength_threshold: float = 0.8,
    drone_strength_threshold: float = 0.5,
    compact_radius_threshold: float = 3.0,
) -> str:
    """
    True semantic classifier (radiance-only, single frame).
    Returns: 'missile', 'drone', or 'decoy'.
    """
    if strength >= missile_strength_threshold and radius <= compact_radius_threshold:
        return "missile"
    elif strength >= drone_strength_threshold:
        return "drone"
    else:
        return "decoy"


def classify_detections_frame(dets: list) -> list:
    """
    Classify all components in a single frame.
    Expects each component to have 'strength' and 'radius'.
    """
    labeled = []
    for c in dets:
        cc = dict(c)
        s = cc.get("strength", 0.0)
        r = cc.get("radius", 0.0)
        cc["semantic_type"] = classify_component_radiance(s, r)
        labeled.append(cc)
    return labeled


def classify_detections_sequence(detections_seq: list) -> list:
    """
    Classify all components across a sequence.
    """
    return [classify_detections_frame(dets) for dets in detections_seq]
