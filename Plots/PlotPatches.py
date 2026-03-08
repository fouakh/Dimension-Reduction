import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class PlotPatches:

    def __init__(self, shapes, patches, centers, grid=(1, 1), save_name="patches"):

        if not (len(shapes) == len(patches) == len(centers)):
            raise ValueError("Shapes, patches, and centers must have same length.")

        n_rows, n_cols = grid
        if n_rows * n_cols < len(shapes):
            raise ValueError("Grid too small for number of patches.")

        self.shapes = shapes
        self.patches = patches
        self.centers = centers
        self.grid = grid

        base_dir = "figs"
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.save_dir = os.path.join(
            base_dir,
            f"{save_name}_{timestamp}"
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, figsize=(8, 6), point_size=15):

        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for ax, shape, patch, center in zip(
            axes, self.shapes, self.patches, self.centers
        ):
            X = shape.samples()
            patch_set = set(patch)

            mask = np.array([i in patch_set for i in range(len(X))])

            ax.scatter(
                X[~mask, 0],
                X[~mask, 1],
                s=point_size,
                color="lightgray"
            )

            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=point_size,
                color="blue"
            )

            ax.scatter(
                X[center, 0],
                X[center, 1],
                s=point_size * 3,
                color="red"
            )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.shapes):]:
            ax.axis("off")

        fig.tight_layout()

        fig.savefig(
            os.path.join(self.save_dir, "patches.png"),
            dpi=200
        )

        plt.close(fig)