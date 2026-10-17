# parcs_v3/assoc_trace_v3.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameTrace:
    frame_idx: int
    gt_count: int
    det_count: int
    match_count: int
    missed_count: int
    false_pos_count: int
    localization_errors: List[float] = field(default_factory=list)
    reconstruction_errors: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssocTraceCollectorV3:
    """
    Association trace collector (v3).

    Collects per-frame summaries without exposing physics or sensor internals.
    """

    def __init__(self):
        self._frames: List[FrameTrace] = []

    def record_frame(self, frame_idx: int, frame_dict: Dict[str, Any]) -> None:
        matches = frame_dict.get("matches", [])
        missed = frame_dict.get("missed", [])
        false_pos = frame_dict.get("false_positives", [])
        loc_errs = frame_dict.get("localization_errors", [])
        rec_errs = frame_dict.get("reconstruction_errors", [])

        gt_count = len(matches) + len(missed)
        det_count = len(matches) + len(false_pos)

        trace = FrameTrace(
            frame_idx=frame_idx,
            gt_count=gt_count,
            det_count=det_count,
            match_count=len(matches),
            missed_count=len(missed),
            false_pos_count=len(false_pos),
            localization_errors=list(loc_errs),
            reconstruction_errors=list(rec_errs),
        )
        self._frames.append(trace)

    def get_frames(self) -> List[FrameTrace]:
        return list(self._frames)

    def summary(self) -> Dict[str, Any]:
        """
        High-level summary suitable for sharing with reviewers.
        """
        total_frames = len(self._frames)
        total_matches = sum(f.match_count for f in self._frames)
        total_missed = sum(f.missed_count for f in self._frames)
        total_false_pos = sum(f.false_pos_count for f in self._frames)

        return {
            "n_frames": total_frames,
            "total_matches": total_matches,
            "total_missed": total_missed,
            "total_false_positives": total_false_pos,
        }
