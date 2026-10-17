import numpy as np
import matplotlib.pyplot as plt

def plot_radiance_field(field: np.ndarray, title: str = "Radiance Field"):
    plt.figure(figsize=(5, 5))
    plt.imshow(field, cmap="inferno", origin="lower")
    plt.colorbar(label="Radiance")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_deltaS_frame(deltaS: np.ndarray, title: str = "ΔS Frame"):
    plt.figure(figsize=(5, 5))
    plt.imshow(deltaS, cmap="viridis", origin="lower")
    plt.colorbar(label="ΔS")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_trajectories(trajectories: list, title: str = "Detected Trajectories"):
    plt.figure(figsize=(6, 6))
    for traj in trajectories:
        xs = [p["centroid"][0] for p in traj]
        ys = [p["centroid"][1] for p in traj]
        plt.plot(xs, ys, "-o")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trajectory_errors(traj_errors: dict):
    labels = list(traj_errors.keys())
    values = [traj_errors[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Value")
    plt.title("Trajectory Error Metrics")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()
