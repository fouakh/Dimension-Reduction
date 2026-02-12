import numpy as np
import matplotlib.pyplot as plt


class PlotPatchesProgression:
    def __init__(self, shapes, ordered_centers_list, patches_list, grid=(1, 1)):
        self.shapes = shapes
        self.ordered_centers_list = ordered_centers_list
        self.patches_list = patches_list
        self.grid = grid

        if not (
            len(shapes) == len(ordered_centers_list) == len(patches_list)
        ):
            raise ValueError("Shapes, centers and patches must have same length.")

        n_rows, n_cols = grid
        if n_rows * n_cols < len(shapes):
            raise ValueError("Grid too small for number of shapes.")

    def plot(self, figsize=(10, 6), point_size=10):

        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for ax, shape, ordered_centers, patches in zip(
            axes, self.shapes, self.ordered_centers_list, self.patches_list
        ):

            X = shape.samples()
            n_points = X.shape[0]
            
            coverage_step = np.full(n_points, -1)

            for step, center in enumerate(ordered_centers):
                patch_nodes, _ = patches[center]

                for node in patch_nodes:
                    if coverage_step[node] == -1:
                        coverage_step[node] = step

            max_step = max(coverage_step)
            normalized = coverage_step / max(1, max_step)

            cmap = plt.cm.get_cmap("viridis")

            ax.scatter(
                X[:, 0],
                X[:, 1],
                c=normalized,
                cmap=cmap,
                s=point_size
            )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.shapes):]:
            ax.axis("off")

        plt.tight_layout()
        return fig, axes
 