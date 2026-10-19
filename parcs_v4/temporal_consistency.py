# temporal_consistency.py
import numpy as np
import matplotlib.pyplot as plt

def build_temporal_presence_matrix(trajectories: list, n_frames: int) -> np.ndarray:
    """
    Build a matrix M[traj_index, frame] = 1 if trajectory has a detection at that frame, else 0.
    """
    M = np.zeros((len(trajectories), n_frames), dtype=int)
    for i, traj in enumerate(trajectories):
        for p in traj:
            t = p["t"]
            if 0 <= t < n_frames:
                M[i, t] = 1
    return M


def plot_temporal_consistency_heatmap(trajectories: list, n_frames: int, title: str = "Temporal Consistency"):
    """
    Visualize temporal presence of trajectories as a heatmap.
    """
    M = build_temporal_presence_matrix(trajectories, n_frames)

    plt.figure(figsize=(8, 4))
    plt.imshow(M, aspect="auto", cmap="Greens", interpolation="nearest")
    plt.colorbar(label="Presence (1=detected)")
    plt.xlabel("Frame index")
    plt.ylabel("Trajectory index")
    plt.title(title)
    plt.tight_layout()
    plt.show()
