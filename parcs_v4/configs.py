# /content/parcs_v4/configs.py

import numpy as np

# ============================================================
# Core Data Classes for PARCS v4
# ============================================================

class ObjectClass:
    """
    PARCS v4 Object Representation

    Fields:
        frame_idx      : int
        centroid       : np.array([x, y])
        radius         : float
        semantic_type  : str ("missile", "drone", "decoy")
        identity       : str or None
    """

    def __init__(self, frame_idx, centroid, radius, semantic_type, identity=None):
        self.frame_idx = frame_idx
        self.centroid = np.array(centroid)
        self.radius = radius
        self.semantic_type = semantic_type
        self.identity = identity

    def centroid_distance(self, other):
        return np.linalg.norm(self.centroid - other.centroid)


class FrameClass:
    """
    PARCS v4 Frame Representation

    Fields:
        frame_idx   : int
        gt_objects  : list[ObjectClass]
        detections  : list[ObjectClass]
        tracks      : list[ObjectClass] (populated by PARCSv4Operator)
    """

    def __init__(self, frame_idx, gt_objects, detections):
        self.frame_idx = frame_idx
        self.gt_objects = gt_objects
        self.detections = detections
        self.tracks = []


class SceneClass:
    """
    PARCS v4 Scene Representation

    Fields:
        frames : list[FrameClass]
    """

    def __init__(self, frames):
        self.frames = frames


class ResultsClass:
    """
    PARCS v4 Results Representation

    Fields:
        frames : list[FrameClass]
        tracks : dict[obj_id -> list[ObjectClass]]
        cfg    : PARCSConfig
    """

    def __init__(self, frames, tracks, cfg):
        self.frames = frames
        self.tracks = tracks
        self.cfg = cfg


# ============================================================
# PARCS v4 Configuration Container
# ============================================================

class PARCSConfig:
    """
    Unified configuration for PARCS v4.

    Contains:
        - data model classes
        - operator parameters
        - adversarial parameters
        - temporal smoothing parameters
        - plotting parameters
    """

    # Data model classes
    SceneClass = SceneClass
    FrameClass = FrameClass
    ObjectClass = ObjectClass
    ResultsClass = ResultsClass

    # Operator parameters
    max_assoc_distance = 70.0
    temporal_tolerance = 4
    
    # Forward model parameters
    num_frames = 50
    objects_per_frame = 5

    # Semantic mismatch penalty
    semantic_mismatch_penalty = 40.0

    # Temporal tracking parameters
    temporal_smoothing_factor = 0.35

    def __init__(self):
        pass
