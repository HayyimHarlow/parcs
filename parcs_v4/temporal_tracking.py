# parcs_v4/temporal_tracking.py

import numpy as np

class TemporalTracker:
    """
    PARCS v4 Temporal Tracker

    Provides temporal smoothing for track histories and rebuilds
    a smoothed ResultsClass object compatible with all v4 metrics
    and plotting modules.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha = getattr(cfg, "temporal_smoothing_factor", 0.25)

    # ------------------------------------------------------------
    # Exponential smoothing for a single trajectory
    # ------------------------------------------------------------
    def smooth_trajectory(self, traj):
        if len(traj) < 2:
            return traj

        smoothed = []
        prev = traj[0].centroid.copy()

        for frame_obj in traj:
            new_centroid = (
                self.alpha * frame_obj.centroid +
                (1 - self.alpha) * prev
            )

            # clone frame_obj
            new_obj = self.cfg.ObjectClass(
                frame_idx=frame_obj.frame_idx,
                centroid=new_centroid,
                radius=frame_obj.radius,
                semantic_type=frame_obj.semantic_type,
                identity=frame_obj.identity
            )

            smoothed.append(new_obj)
            prev = new_centroid

        return smoothed

    # ------------------------------------------------------------
    # Full smoothing pass over results.tracks
    # ------------------------------------------------------------
    def smooth(self, results):
        """
        Apply smoothing to all tracks and rebuild a new ResultsClass.

        results.tracks is:
            dict[track_id] -> list[ObjectClass]
        """

        # 1. Smooth each track history
        smoothed_tracks = {}
        for track_id, traj in results.tracks.items():
            smoothed_tracks[track_id] = self.smooth_trajectory(traj)

        # 2. Rebuild frames with smoothed objects
        new_frames = []
        for frame in results.frames:
            f_idx = frame.frame_idx

            # smoothed detections: take objects whose frame_idx matches
            smoothed_dets = []
            for track_id, traj in smoothed_tracks.items():
                for obj in traj:
                    if obj.frame_idx == f_idx:
                        smoothed_dets.append(obj)

            # GT objects remain unchanged
            new_frame = self.cfg.FrameClass(
                frame_idx=f_idx,
                gt_objects=frame.gt_objects,
                detections=smoothed_dets
            )

            # attach track list for plotting
            new_frame.tracks = smoothed_dets
            new_frames.append(new_frame)

        # 3. Return new ResultsClass
        return self.cfg.ResultsClass(new_frames, smoothed_tracks, self.cfg)

