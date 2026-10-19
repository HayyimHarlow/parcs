# /content/parcs_v4/adversarial_v4.py
import copy
import numpy as np

class FullAdversarialGenerator:
    """
    PARCS v4 Full Adversarial Generator

    Applies the complete adversarial envelope to a synthetic scene:
        - centroid jitter
        - radius corruption
        - semantic flips
        - dropout
        - false-positive bursts

    This is the frozen adversarial path used in PARCS v4 evaluation.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Adversarial parameters
        self.jitter_sigma = getattr(cfg, "adv_jitter_sigma", 15.0)
        self.radius_scale = getattr(cfg, "adv_radius_scale", 0.25)
        self.semantic_flip_prob = getattr(cfg, "adv_semantic_flip_prob", 0.10)
        self.dropout_prob = getattr(cfg, "adv_dropout_prob", 0.05)
        self.false_positive_rate = getattr(cfg, "adv_false_positive_rate", 0.10)

    def apply_full_adversarial_pipeline(self, scene):
        """
        Apply all adversarial perturbations to the scene.
        Returns a new SceneClass instance.
        """

        frames_out = []

        for frame in scene.frames:
            new_gt = copy.deepcopy(frame.gt_objects)
            new_det = copy.deepcopy(frame.detections)

            # 1. Centroid jitter
            for obj in new_det:
                jitter = np.random.normal(0, self.jitter_sigma, size=2)
                obj.centroid = obj.centroid + jitter

            # 2. Radius corruption
            for obj in new_det:
                obj.radius = obj.radius * (1.0 + np.random.uniform(-self.radius_scale,
                                                                   self.radius_scale))

            # 3. Semantic flips
            for obj in new_det:
                if np.random.rand() < self.semantic_flip_prob:
                    obj.semantic_type = np.random.choice(["missile", "drone", "decoy"])

            # 4. Dropout
            new_det = [obj for obj in new_det if np.random.rand() > self.dropout_prob]

            # 5. False-positive bursts
            num_fp = int(len(new_det) * self.false_positive_rate)
            for _ in range(num_fp):
                fp = self._generate_false_positive(frame.frame_idx)
                new_det.append(fp)

            # Build new frame
            new_frame = self.cfg.FrameClass(
                frame_idx=frame.frame_idx,
                gt_objects=new_gt,
                detections=new_det
            )

            frames_out.append(new_frame)

        return self.cfg.SceneClass(frames_out)

    def _generate_false_positive(self, frame_idx):
        """
        Generate a synthetic false-positive detection.
        """

        centroid = np.array([
            np.random.uniform(0, 1000),
            np.random.uniform(0, 1000)
        ])

        radius = np.random.uniform(5, 20)
        semantic_type = np.random.choice(["missile", "drone", "decoy"])

        return self.cfg.ObjectClass(
            frame_idx=frame_idx,
            centroid=centroid,
            radius=radius,
            semantic_type=semantic_type,
            identity=None
        )

