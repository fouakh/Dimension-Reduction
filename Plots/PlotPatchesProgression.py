import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class PlotPatchesProgression:

    def __init__(self, shapes, ordered_centers_list, patches_list, grid=(1, 1)):
        self.shapes = shapes
        self.ordered_centers_list = ordered_centers_list
        self.patches_list = patches_list
        self.grid = grid

    def plot(self, figsize=(10, 6), point_size=10):

        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for ax, shape, ordered_centers, patches in zip(
            axes, self.shapes, self.ordered_centers_list, self.patches_list
        ):

            X = shape.samples()
            n_steps = len(ordered_centers)

            cmap = plt.cm.get_cmap("viridis")

            for step, center in enumerate(ordered_centers):

                patch_dict = patches[center]
                patch_nodes = patch_dict[center]

                if len(patch_nodes) < 3:
                    continue

                pts = X[patch_nodes]

                try:
                    hull = ConvexHull(pts)
                except Exception:
                    continue

                hull_pts = pts[hull.vertices]

                t = step / max(1, n_steps - 1)
                color = cmap(t)
                alpha = 0.15 + 0.6 * t

                ax.fill(
                    hull_pts[:, 0],
                    hull_pts[:, 1],
                    color=color,
                    alpha=alpha,
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=2
                )

                ax.scatter(
                    X[center, 0],
                    X[center, 1],
                    color=color,
                    edgecolor="black",
                    s=60,
                    zorder=3
                )

            ax.scatter(
                X[:, 0],
                X[:, 1],
                color="black",
                s=point_size,
                alpha=0.15,
                zorder=1
            )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.shapes):]:
            ax.axis("off")

        plt.tight_layout()
        return fig, axes
