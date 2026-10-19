# parcsv4/forward_model.py

import numpy as np

class ForwardSceneGenerator:
    """
    PARCS v4 Forward Scene Generator

    Produces a synthetic scene consisting of frames, each containing
    ground-truth objects with geometric and semantic attributes.

    This is the first stage of the v4 pipeline.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_frames = getattr(cfg, "num_frames", 50)
        self.objects_per_frame = getattr(cfg, "objects_per_frame", 5)

    def generate_scene(self):
        frames = []

        for frame_idx in range(self.num_frames):
            gt_objects = []
            detections = []

            # Generate GT objects
            for obj_idx in range(self.objects_per_frame):
                obj = self._generate_gt_object(frame_idx, obj_idx)
                gt_objects.append(obj)

            # Initial detections = GT objects (before adversarial perturbation)
            detections = list(gt_objects)

            frame = self.cfg.FrameClass(
                frame_idx=frame_idx,
                gt_objects=gt_objects,
                detections=detections
            )
            frames.append(frame)

        return self.cfg.SceneClass(frames)

    def _generate_gt_object(self, frame_idx, obj_idx):
        """
        Generate a single ground-truth object with centroid, radius,
        semantic type, and identity.
        """

        centroid = np.array([
            np.random.uniform(0, 1000),
            np.random.uniform(0, 1000)
        ])

        radius = np.random.uniform(5, 20)

        semantic_type = np.random.choice(["missile", "drone", "decoy"])

        identity = f"obj_{obj_idx}"

        return self.cfg.ObjectClass(
            frame_idx=frame_idx,
            centroid=centroid,
            radius=radius,
            semantic_type=semantic_type,
            identity=identity
        )

