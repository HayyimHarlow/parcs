# parcs_v3/radiance_warmstart.py

import numpy as np
from parcs_v3.forward_model import scene_to_radiance_field
from parcs_v3.configs import SENSOR_CONFIG, SCENE_DEFAULT

def generate_warmstart_fields(
    n_warm_frames: int = 5,
    dt: float = 0.1,
    scene_config: dict = SCENE_DEFAULT,
    sensor_config: dict = SENSOR_CONFIG,
) -> list:
    """
    Generate a short sequence of radiance fields to warm-start the system.
    """
    fields = []
    for k in range(n_warm_frames):
        t = - (n_warm_frames - k) * dt  # negative time: pre-roll
        f = scene_to_radiance_field(t=t, scene_config=scene_config, sensor_config=sensor_config)
        fields.append(f)
    return fields


def stabilize_radiance_sequence(fields_seq: list, n_warm_frames: int = 5) -> list:
    """
    Prepend warm-start fields and drop the first raw frame to eliminate initialization anomalies.
    """
    warm = generate_warmstart_fields(n_warm_frames=n_warm_frames)
    stabilized = warm + fields_seq[1:]  # drop original frame 0
    return stabilized
