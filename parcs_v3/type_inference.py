# type_inference.py
import numpy as np

def infer_target_type_from_radiance(component: dict,
                                    decoy_threshold: float = 0.6) -> str:
    strength = component.get("strength", 0.0)
    radius = component.get("radius", 0.0)

    if strength >= decoy_threshold and radius <= 3.0:
        return "missile"
    elif strength >= decoy_threshold and radius > 3.0:
        return "drone"
    else:
        return "decoy"


def label_detections_with_types(detections_seq: list,
                                decoy_threshold: float = 0.6) -> list:
    labeled_seq = []
    for dets in detections_seq:
        frame = []
        for c in dets:
            cc = dict(c)
            cc["type"] = infer_target_type_from_radiance(cc, decoy_threshold)
            frame.append(cc)
        labeled_seq.append(frame)
    return labeled_seq

