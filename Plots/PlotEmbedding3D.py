import matplotlib.pyplot as plt
import numpy as np


class PlotEmbeddingComparison3Dwith2D:
    def __init__(self, shapes, embeddings_2d, colors=None, grid=(1, 1), z_mode="min"):
       
        self.shapes = shapes
        self.embeddings_2d = embeddings_2d
        self.colors = colors
        self.grid = grid
        self.z_mode = z_mode

        if len(self.shapes) != len(self.embeddings_2d):
            raise ValueError("Shapes and embeddings must have same length.")

        if self.colors is not None and len(self.colors) != len(self.shapes):
            raise ValueError("Number of colors must match number of shapes.")

        n_rows, n_cols = self.grid
        if n_rows * n_cols < len(self.shapes):
            raise ValueError("Grid too small for number of shapes.")

    def plot(self, figsize=(9, 7), point_size=10, elev=20, azim=-60):
        n_rows, n_cols = self.grid
        fig = plt.figure(figsize=figsize)

        axes = []
        total_slots = n_rows * n_cols
        for idx in range(total_slots):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
            axes.append(ax)

        if self.colors is None:
            cmap = plt.cm.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(self.shapes))]
        else:
            colors = self.colors

        for ax, shape, X_new2d, color in zip(axes, self.shapes, self.embeddings_2d, colors):
            X_old = shape.samples()

            if X_old.ndim != 2 or X_old.shape[1] != 3:
                raise ValueError(f"Expected shape.samples() -> (n, 3). Got {X_old.shape}.")
            if X_new2d.ndim != 2 or X_new2d.shape[1] != 2:
                raise ValueError(f"Expected embedding -> (n, 2). Got {X_new2d.shape}.")

            # Choose the z plane for the 2D embedding
            z_old = X_old[:, 2]
            if self.z_mode == "zero":
                z0 = 0.0
            elif self.z_mode == "mean":
                z0 = float(np.mean(z_old))
            elif self.z_mode == "min":
                z0 = float(np.min(z_old))
            elif self.z_mode == "max":
                z0 = float(np.max(z_old))
            else:
                raise ValueError(f"Unknown z_mode: {self.z_mode}")

            X_new3d = np.column_stack([X_new2d[:, 0], X_new2d[:, 1], np.full(len(X_new2d), z0)])

            # Old 3D points
            ax.scatter(
                X_old[:, 0], X_old[:, 1], X_old[:, 2],
                s=point_size, color="lightgray", alpha=0.35
            )

            # New embedding (2D lifted into 3D plane)
            ax.scatter(
                X_new3d[:, 0], X_new3d[:, 1], X_new3d[:, 2],
                s=point_size, color=color
            )

            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()

            # scale using both sets (best effort)
            self._set_equal_aspect_3d(ax, np.vstack([X_old, X_new3d]))

        for ax in axes[len(self.shapes):]:
            ax.set_axis_off()

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def _set_equal_aspect_3d(ax, X):
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        z_min, z_max = X[:, 2].min(), X[:, 2].max()

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range == 0:
            max_range = 1.0

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        half = max_range / 2
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)