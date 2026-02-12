import matplotlib.pyplot as plt
import numpy as np


class PlotPatches:
    def __init__(self, shapes, patches, centers, grid=(1, 1)):
        self.shapes = shapes
        self.patches = patches
        self.centers = centers
        self.grid = grid

        if not (len(shapes) == len(patches) == len(centers)):
            raise ValueError("Shapes, patches, and centers must have same length.")

        n_rows, n_cols = self.grid
        if n_rows * n_cols < len(self.shapes):
            raise ValueError("Grid too small for number of patches.")

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

        plt.tight_layout()
        return fig, axes
