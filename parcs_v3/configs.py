# parcsv3/configs.py

SENSOR_CONFIG = {
    "fov_deg": 30.0,
    "grid_size": 128,
    "integration_time": 0.01,
    "noise_std": 0.02,
}

SCENE_DEFAULT = {
    "targets": [
        {"type": "missile", "pos0": (0.2, 0.1), "vel": (0.01, 0.02), "radius": 3.0, "gain": 1.0},
        {"type": "drone",   "pos0": (0.7, 0.6), "vel": (-0.01, 0.0), "radius": 2.0, "gain": 0.7},
    ],
    "background_level": 0.1,
}
