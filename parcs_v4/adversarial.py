import copy

from .configs import SCENE_DEFAULT

def make_adversarial_scene(
    base_scene: dict = SCENE_DEFAULT,
    *,
    n_decoys: int = 3,
    decoy_gain: float = 0.5,
    decoy_radius: float = 2.0,
    crossing: bool = True
) -> dict:
    """
    Generate an adversarial scene:
    - extra decoy targets
    - optional crossing trajectories
    """
    scene = copy.deepcopy(base_scene)
    targets = scene["targets"]

    # Add decoys
    for k in range(n_decoys):
        targets.append({
            "type": "decoy",
            "pos0": (0.1 + 0.2 * k, 0.8),
            "vel": (0.02, -0.03),
            "radius": decoy_radius,
            "gain": decoy_gain,
        })

    # Optional crossing: invert one target's velocity
    if crossing and len(targets) >= 2:
        targets[0]["vel"] = (targets[0]["vel"][0], -targets[0]["vel"][1])

    scene["targets"] = targets
    return scene
