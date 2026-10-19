# parcs_v3/assoc_invariants_v3.py

from typing import Any, Dict, List


class AssocOperatorInvariantsV3:
    """
    Formal operator invariants for dict-based association.

    All checks are structural and metric-level only; no physics is exposed.
    """

    def __init__(self, *, max_localization_error: float = 1e3):
        self.max_localization_error = max_localization_error

    def check_frame(self, frame_idx: int, frame_dict: Dict[str, Any]) -> None:
        matches = frame_dict.get("matches", [])
        missed = frame_dict.get("missed", [])
        false_pos = frame_dict.get("false_positives", [])
        loc_errs: List[float] = frame_dict.get("localization_errors", [])
        rec_errs: List[float] = frame_dict.get("reconstruction_errors", [])

        # Invariant 1: counts are consistent
        gt_count = len(matches) + len(missed)
        det_count = len(matches) + len(false_pos)
        assert gt_count >= 0, f"[f={frame_idx}] negative GT count"
        assert det_count >= 0, f"[f={frame_idx}] negative Det count"

        # Invariant 2: error vectors align with matches
        assert len(loc_errs) == len(matches), (
            f"[f={frame_idx}] localization_errors length mismatch: "
            f"{len(loc_errs)} vs {len(matches)}"
        )
        assert len(rec_errs) == len(matches), (
            f"[f={frame_idx}] reconstruction_errors length mismatch: "
            f"{len(rec_errs)} vs {len(matches)}"
        )

        # Invariant 3: errors are finite and bounded
        for e in loc_errs:
            assert e >= 0.0, f"[f={frame_idx}] negative localization error"
            assert e <= self.max_localization_error, (
                f"[f={frame_idx}] localization error too large: {e}"
            )
        for e in rec_errs:
            assert e >= 0.0, f"[f={frame_idx}] negative reconstruction error"

        # Invariant 4: match objects are structurally complete
        for m in matches:
            assert "gt" in m and "det" in m, f"[f={frame_idx}] malformed match"
            gt = m["gt"]
            det = m["det"]
            assert "centroid" in gt and "radius" in gt, "[f={frame_idx}] GT missing fields"
            assert "centroid" in det and "radius" in det, "[f={frame_idx}] Det missing fields"
