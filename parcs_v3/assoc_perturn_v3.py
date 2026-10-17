# parcs_v3/assoc_perturb_v3.py

import copy
import numpy as np
from typing import Any, Callable, Dict, List, Optional


class AssocPerturbationStabilityV3:
    """
    Perturbation stability tester for dict-based association.

    Applies controlled, abstract perturbations to detections and evaluates
    stability of the association output. No physics or sensor internals
    are exposed—only structural and metric-level behavior.
    """

    def __init__(
        self,
        run_operator: Callable[..., List[Dict[str, Any]]],
        *,
        rng: Optional[np.random.Generator] = None,
        max_perturbation: float = 0.5,
        n_trials: int = 8,
    ):
        self.run_operator = run_operator
        self.rng = rng or np.random.default_rng()
        self.max_perturbation = max_perturbation
        self.n_trials = n_trials

    def _perturb_traj_temporal(self, traj_temporal) -> Any:
        """
        Abstract perturbation: shallow copy of traj_temporal with
        centroid jitter only. Assumes traj_temporal exposes
        get_frame_detections(f_idx) and set_frame_detections(f_idx, dets).
        """
        perturbed = copy.deepcopy(traj_temporal)

        for f_idx in range(len(traj_temporal)):
            dets = perturbed.get_frame_detections(f_idx)
            for d in dets:
                cx, cy = d.centroid
                dx = self.rng.uniform(-self.max_perturbation, self.max_perturbation)
                dy = self.rng.uniform(-self.max_perturbation, self.max_perturbation)
                d.centroid = (cx + dx, cy + dy)
            perturbed.set_frame_detections(f_idx, dets)

        return perturbed

    def evaluate(
        self,
        gt_traj,
        traj_temporal,
        *,
        match_policy=None,
        error_metrics=None,
        invariant_checks=None,
    ) -> Dict[str, Any]:
        """
        Run multiple perturbation trials and compute stability metrics
        on association outputs (e.g., match counts per frame).
        """

        baseline_assoc = self.run_operator(
            gt_traj,
            traj_temporal,
            match_policy=match_policy,
            error_metrics=error_metrics,
            invariant_checks=invariant_checks,
        )

        baseline_match_counts = [
            len(frame.get("matches", [])) for frame in baseline_assoc
        ]

        trial_match_counts: List[List[int]] = []

        for _ in range(self.n_trials):
            perturbed_traj = self._perturb_traj_temporal(traj_temporal)
            assoc = self.run_operator(
                gt_traj,
                perturbed_traj,
                match_policy=match_policy,
                error_metrics=error_metrics,
                invariant_checks=invariant_checks,
            )
            trial_match_counts.append(
                [len(frame.get("matches", [])) for frame in assoc]
            )

        # Simple stability metric: per-frame variance of match counts
        variances = []
        for f_idx in range(len(baseline_match_counts)):
            samples = [trial[f_idx] for trial in trial_match_counts]
            variances.append(float(np.var(samples)))

        return {
            "baseline_match_counts": baseline_match_counts,
            "match_count_variances": variances,
            "mean_variance": float(np.mean(variances)),
        }
