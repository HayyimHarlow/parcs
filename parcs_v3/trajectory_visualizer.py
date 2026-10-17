# parcs_v3/trajectory_visualizer.py

import matplotlib.pyplot as plt

def plot_multi_target_trajectories(trajectories: list,
                                   gt_traj: list | None = None,
                                   title: str = "Multi-Target Trajectories"):
    plt.figure(figsize=(6,6))

    for i, traj in enumerate(trajectories):
        xs = [p["centroid"][0] for p in traj]
        ys = [p["centroid"][1] for p in traj]
        plt.plot(xs, ys, "-o", label=f"Det {i}")

    if gt_traj is not None:
        # transpose GT list-of-lists
        zipped = list(zip(*gt_traj))
        for j, gt in enumerate(zipped):
            xs_gt = [g["centroid"][0] for g in gt]
            ys_gt = [g["centroid"][1] for g in gt]
            plt.plot(xs_gt, ys_gt, "--", linewidth=1.5, label=f"GT {j}")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()

