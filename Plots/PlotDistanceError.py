import numpy as np
import matplotlib.pyplot as plt


class PlotDistanceError:
    def __init__(self, shapes, D_true_list, D_hat_list, grid=(1, 1)):
        self.shapes = shapes
        self.D_true_list = D_true_list
        self.D_hat_list = D_hat_list
        self.grid = grid

        if len(shapes) != len(D_true_list) or len(shapes) != len(D_hat_list):
            raise ValueError("Shapes and distance lists must have same length.")

        n_rows, n_cols = self.grid
        if n_rows * n_cols < len(self.shapes):
            raise ValueError("Grid too small for number of shapes.")

    def _compute_point_error(self, D_true, D_hat):
        diff = D_hat - D_true
        rms = np.sqrt(np.mean(diff**2, axis=1))
        return rms

    def plot(self, figsize=(8, 6), point_size=15):
        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        cmap = plt.cm.get_cmap("RdYlGn_r")

        all_errors = []
        for D_true, D_hat in zip(self.D_true_list, self.D_hat_list):
            all_errors.append(self._compute_point_error(D_true, D_hat))

        global_min = min(err.min() for err in all_errors)
        global_max = max(err.max() for err in all_errors)

        for ax, shape, error in zip(axes, self.shapes, all_errors):
            X = shape.samples()

            sc = ax.scatter(
                X[:, 0],
                X[:, 1],
                c=error,
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
                s=point_size
            )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.shapes):]:
            ax.axis("off")

        return fig, axes

