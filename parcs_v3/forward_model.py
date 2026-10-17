# parcsv3/forward_model.py

import numpy as np
from .configs import SENSOR_CONFIG, SCENE_DEFAULT

def scene_to_radiance_field(t: float,
                            scene_config: dict = SCENE_DEFAULT,
                            sensor_config: dict = SENSOR_CONFIG) -> np.ndarray:
    """
    Map physical scene at time t to a radiance-channel field.
    """
    grid_size = sensor_config["grid_size"]
    field = np.zeros((grid_size, grid_size), dtype=float)

    ys, xs = np.indices(field.shape)
    field += scene_config["background_level"]

    for tgt in scene_config["targets"]:
        x0, y0 = tgt["pos0"]
        vx, vy = tgt["vel"]
        cx = x0 + vx * t
        cy = y0 + vy * t

        cx_pix = cx * grid_size
        cy_pix = cy * grid_size

        r = tgt["radius"]
        A = tgt["gain"]

        dist2 = (xs - cx_pix)**2 + (ys - cy_pix)**2
        field += A * np.exp(-dist2 / (2 * r**2))

    return field


def radiance_to_deltaS(field: np.ndarray,
                       sensor_config: dict = SENSOR_CONFIG,
                       noise_std: float | None = None) -> np.ndarray:
    """
    Convert radiance field to ΔS with detector noise.
    """
    if noise_std is None:
        noise_std = sensor_config["noise_std"]

    rng = np.random.default_rng()
    noise = noise_std * rng.normal(size=field.shape)
    deltaS = field + noise
    return deltaS


def generate_deltaS_sequence(n_frames: int,
                             dt: float,
                             scene_config: dict = SCENE_DEFAULT,
                             sensor_config: dict = SENSOR_CONFIG) -> tuple:
    """
    Generate a time sequence of ΔS frames and ground-truth trajectories.
    """
    deltaS_seq = []
    gt_traj = []

    for k in range(n_frames):
        t = k * dt
        field = scene_to_radiance_field(t, scene_config, sensor_config)
        deltaS = radiance_to_deltaS(field, sensor_config)
        deltaS_seq.append(deltaS)

        # ground-truth positions at time t
        gt_traj.append([
            {
                "type": tgt["type"],
                "centroid": (
                    (tgt["pos0"][0] + tgt["vel"][0] * t) * sensor_config["grid_size"],
                    (tgt["pos0"][1] + tgt["vel"][1] * t) * sensor_config["grid_size"],
                ),
                "radius": tgt["radius"]
            }
            for tgt in scene_config["targets"]
        ])

    return deltaS_seq, gt_traj
