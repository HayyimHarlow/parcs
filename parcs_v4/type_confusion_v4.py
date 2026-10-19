# parcs_v4/type_confusion_v4.py

import numpy as np
from collections import defaultdict

import numpy as np

class TypeConfusionEvaluator:
    """
    PARCS v4 Type-Confusion Evaluator

    This evaluator measures semantic-class robustness by comparing
    ground-truth semantic types against inferred semantic types
    across all frames.

    Indexing is corrected to:
        det_traj[frame_idx][det_idx]

    Semantic classes are fixed to the PARCS v4 taxonomy:
        0 = missile
        1 = drone
        2 = decoy
        3 = none (no detection / dropout)

    Output is a DARPA-ready confusion matrix suitable for inclusion
    in technical appendices.
    """

    SEM_CLASSES = ["missile", "drone", "decoy", "none"]
    NUM_CLASSES = 4

    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, results):
        """
        Build confusion matrix:
            rows = ground truth
            cols = inferred semantic type
        """

        cm = np.zeros((self.NUM_CLASSES, self.NUM_CLASSES), dtype=int)

        for frame in results.frames:
            gt_frame = frame.gt_objects
            det_frame = frame.detections

            for gt_obj in gt_frame:
                gt_type = self._encode(gt_obj.semantic_type)

                # find best detection match for this GT object
                det = self._match_detection(gt_obj, det_frame)

                if det is None:
                    inf_type = self._encode("none")
                else:
                    inf_type = self._encode(det.semantic_type)

                cm[gt_type, inf_type] += 1

        return cm

    def _match_detection(self, gt_obj, det_frame):
        """
        Simple nearest-neighbor match for confusion evaluation.
        This is NOT the association operator; it is only for
        semantic robustness measurement.
        """

        best_det = None
        best_dist = float("inf")

        for det in det_frame:
            dist = gt_obj.centroid_distance(det)
            if dist < best_dist:
                best_dist = dist
                best_det = det

        return best_det

    def _encode(self, sem_type):
        """
        Convert semantic type string → class index.
        """

        if sem_type == "missile":
            return 0
        if sem_type == "drone":
            return 1
        if sem_type == "decoy":
            return 2
        return 3  # none / dropout / unknown

    def pretty_print(self, cm):
        """
        DARPA-style formatted confusion matrix.
        """

        header = "          missile   drone   decoy   none"
        print(header)
        print("------------------------------------------------")

        for i, row in enumerate(cm):
            label = self.SEM_CLASSES[i].ljust(10)
            counts = "   ".join([str(x).rjust(6) for x in row])
            print(f"{label}{counts}")




