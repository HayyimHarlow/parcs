import numpy as np

class TieredMatchPolicy:
    """
    PARCS v4 TieredMatchPolicy

    Association priority:
        1. Geometric proximity (centroid distance)
        2. Temporal consistency (frame continuity)
        3. Semantic compatibility (type/class)

    No GT index matching is used. This is the evaluated association path.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_distance = cfg.max_assoc_distance
        self.temporal_tolerance = cfg.temporal_tolerance
        self.semantic_mismatch_penalty = getattr(
            cfg, "semantic_mismatch_penalty", 0.0
        )

    def associate(self, tracks, detections, frame_idx):
        associations = []
        used_dets = set()

        for track in tracks:
            best_det = None
            best_score = float("inf")

            for det_idx, det in enumerate(detections):
                if det_idx in used_dets:
                    continue

                # geometric distance
                dist = track.centroid_distance(det)
                if dist > self.max_distance:
                    continue

                # temporal consistency
                dt = abs(det.frame_idx - frame_idx)
                if dt > self.temporal_tolerance:
                    continue

                # semantic compatibility
                sem_penalty = 0.0
                if track.semantic_type != det.semantic_type:
                    sem_penalty = self.semantic_mismatch_penalty

                score = dist + sem_penalty

                if score < best_score:
                    best_score = score
                    best_det = det_idx

            if best_det is not None:
                associations.append((track.id, best_det))
                used_dets.add(best_det)

        return associations


class Track:
    """
    Minimal v4 track object used by PARCSv4Operator.

    Fields:
        id             : int
        semantic_type  : str
        identity       : str or None
        centroid       : np.array([x, y])
        last_frame_idx : int
        history        : list[ObjectClass]
    """

    _next_id = 0

    def __init__(self, det):
        self.id = Track._next_id
        Track._next_id += 1

        self.semantic_type = det.semantic_type
        self.identity = getattr(det, "identity", None)
        self.centroid = det.centroid
        self.last_frame_idx = det.frame_idx
        self.history = [det]

    def centroid_distance(self, other):
        return np.linalg.norm(self.centroid - other.centroid)

    def update(self, det):
        self.history.append(det)
        self.centroid = det.centroid
        self.last_frame_idx = det.frame_idx


class PARCSv4Operator:
    """
    PARCS v4 operator using TieredMatchPolicy as the evaluated path.
    """

    def __init__(self, cfg, policy: TieredMatchPolicy):
        self.cfg = cfg
        self.policy = policy

    def run(self, scene):
        # scene: adversarially perturbed forward model output
        tracks = []
        frames_out = []

        for frame_idx, frame in enumerate(scene.frames):
            detections = frame.detections  # <-- FIX: iterate over detections list

            # update existing tracks
            associations = self.policy.associate(tracks, detections, frame_idx)

            # apply associations
            for track_id, det_idx in associations:
                det = detections[det_idx]
                track = next(t for t in tracks if t.id == track_id)
                track.update(det)

            # create new tracks for unassociated detections
            associated_dets = {d for _, d in associations}
            for det_idx, det in enumerate(detections):
                if det_idx in associated_dets:
                    continue
                new_track = Track(det)
                tracks.append(new_track)

            # attach tracks to frame for plotting/temporal diagnostics
            frame.tracks = list(tracks)
            frames_out.append(frame)

        # build results.tracks dict: track_id -> history list
        tracks_dict = {t.id: list(t.history) for t in tracks}

        return self.cfg.ResultsClass(frames_out, tracks_dict, self.cfg)

