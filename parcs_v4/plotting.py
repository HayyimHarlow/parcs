# /content/parcs_v4/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from parcs_v4.metrics_v4 import (
    compute_confusion_matrix,
    compute_identity_stability,
)
from parcs_v4.metrics_v4 import (
    compute_confusion_matrix,
    compute_identity_stability,
)
class PlotSuite:
    """
    PARCS v4 Plotting Suite — DARPA‑grade

    Adds:
        - semantic color‑coding
        - motion arrows
        - jitter ellipses
        - velocity vectors
        - uncertainty envelopes
        - multi‑column legends
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Semantic color map
        self.semantic_colors = {
            "missile": "red",
            "drone": "blue",
            "decoy": "orange",
            "unknown": "gray"
        }

    # ------------------------------------------------------------
    # Utility: get semantic color
    # ------------------------------------------------------------
    def _color(self, semantic_type):
        return self.semantic_colors.get(semantic_type, "black")

    # ------------------------------------------------------------
    # Frame‑level visualization
    # ------------------------------------------------------------
    def plot_frame(self, frame):
        plt.figure(figsize=(8, 8))

        # GT objects
        for obj in frame.gt_objects:
            x, y = obj.centroid
            c = self._color(obj.semantic_type)
            plt.scatter(x, y, c=c, s=80, edgecolor="black")
            plt.text(x + 5, y + 5, obj.semantic_type, color=c)

            # Uncertainty envelope (radius)
            ell = Ellipse((x, y), obj.radius*2, obj.radius*2,
                          edgecolor=c, facecolor="none", linestyle="--")
            plt.gca().add_patch(ell)

        # Detections
        for det in frame.detections:
            x, y = det.centroid
            c = self._color(det.semantic_type)
            plt.scatter(x, y, c=c, s=80, marker="x")
            plt.text(x + 5, y + 5, det.semantic_type, color=c)

            # Jitter ellipse (adversarial jitter visualization)
            jitter = getattr(det, "jitter", 15)
            ell = Ellipse((x, y), jitter*2, jitter*2,
                          edgecolor=c, facecolor="none", linestyle=":")
            plt.gca().add_patch(ell)

        plt.title(f"Frame {frame.frame_idx}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Full‑scene visualization
    # ------------------------------------------------------------
    def plot_results(self, results):
        for frame in results.frames:
            self.plot_frame(frame)

    # ------------------------------------------------------------
    # Trajectory visualization
    # ------------------------------------------------------------
    def plot_trajectories(self, results):
        plt.figure(figsize=(12, 12))

        for track_id, traj in results.tracks.items():
            xs = [f.centroid[0] for f in traj]
            ys = [f.centroid[1] for f in traj]
            sem = traj[0].semantic_type
            c = self._color(sem)

            # Trajectory line
            plt.plot(xs, ys, marker="o", color=c, label=f"{sem} — Track {track_id}")

            # Motion arrows
            for i in range(1, len(xs)):
                plt.arrow(xs[i-1], ys[i-1],
                          xs[i] - xs[i-1],
                          ys[i] - ys[i-1],
                          color=c, length_includes_head=True,
                          head_width=8, alpha=0.6)

            # Velocity vectors
            for i in range(1, len(xs)):
                vx = xs[i] - xs[i-1]
                vy = ys[i] - ys[i-1]
                plt.quiver(xs[i-1], ys[i-1], vx, vy,
                           angles='xy', scale_units='xy', scale=1,
                           color=c, alpha=0.5)

        plt.title("PARCS v4 Temporal Trajectories — DARPA‑Grade")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Multi‑column legend below chart
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=6,
            fontsize="small",
            frameon=False
        )

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Identity continuity visualization
    # ------------------------------------------------------------
    def plot_identity_continuity(self, results):
        plt.figure(figsize=(12, 8))

        for track_id, traj in results.tracks.items():
            xs = [obj.frame_idx for obj in traj]
            ys = [1.0 for _ in traj]
            sem = traj[0].semantic_type
            c = self._color(sem)
            plt.plot(xs, ys, color=c, label=f"{sem} — Track {track_id}")

        plt.title("Identity Continuity (1 = valid, 0 = missing)")
        plt.xlabel("Frame")
        plt.ylabel("Continuity")
        plt.ylim(0, 1.05)

        # Multi‑column legend below chart
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=6,
            fontsize="small",
            frameon=False
        )

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Fusion-aware semantic purity visualization
    # ------------------------------------------------------------
    def plot_semantic_purity(self, results, title_suffix=""):
        """
        Plot per-class semantic purity (diagonal of confusion matrix).
        Expects results from PARCS or PARCS+Fusion.
        """
        conf = compute_confusion_matrix(results)

        classes = ["missile", "drone", "decoy"]
        totals = conf.sum(axis=1)
        diag = np.diag(conf)

        purity = []
        for i in range(3):
            purity.append(diag[i] / totals[i] if totals[i] > 0 else 0.0)

        plt.figure(figsize=(6, 4))
        plt.bar(classes, purity, color=["red", "blue", "orange"])
        plt.ylim(0, 1.0)
        plt.ylabel("Semantic Purity")
        plt.title(f"Semantic Purity {title_suffix}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Fusion-aware identity continuity visualization
    # ------------------------------------------------------------
    def plot_identity_continuity_overlay(self, base_results, fused_results):
        """
        Overlay identity continuity for PARCS v4 vs PARCS v4 + Fusion.
        """
        base_id = compute_identity_stability(base_results)
        fused_id = compute_identity_stability(fused_results)

        labels = ["PARCS v4", "PARCS v4 + Fusion"]
        values = [base_id, fused_id]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=["gray", "purple"])
        plt.ylim(0, 1.0)
        plt.ylabel("Identity Stability")
        plt.title("Identity Stability Comparison")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

