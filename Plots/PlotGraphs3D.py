import matplotlib.pyplot as plt
import numpy as np


class PlotGraphs3D:
    def __init__(
        self,
        graphs,
        grid=(1, 1),
        edge_color="green",
        edge_alpha=0.3,
        edge_width=0.8,
        show_points=True,
        point_size=8,
        point_color="black",
    ):
        self.graphs = graphs
        self.grid = grid
        self.edge_color = edge_color
        self.edge_alpha = edge_alpha
        self.edge_width = edge_width
        self.show_points = show_points
        self.point_size = point_size
        self.point_color = point_color

        n_rows, n_cols = grid
        if n_rows * n_cols < len(graphs):
            raise ValueError("Grid too small for number of graphs.")

    def plot(self, figsize=(10, 8), elev=20, azim=-60):
        n_rows, n_cols = self.grid

        fig = plt.figure(figsize=figsize)

        axes = []
        total_slots = n_rows * n_cols

        for idx in range(total_slots):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
            axes.append(ax)

        for ax, graph in zip(axes, self.graphs):
            X = graph.X
            if X is None:
                raise ValueError("graph.X is None. Build the graph from X first.")
            if X.ndim != 2 or X.shape[1] != 3:
                raise ValueError(f"Expected X of shape (n, 3). Got {X.shape}.")

            # Draw edges
            for i, j in graph.edges():
                xi, yi, zi = X[i]
                xj, yj, zj = X[j]
                ax.plot(
                    [xi, xj],
                    [yi, yj],
                    [zi, zj],
                    color=self.edge_color,
                    alpha=self.edge_alpha,
                    linewidth=self.edge_width,
                )

            # Draw points
            if self.show_points:
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2],
                    s=self.point_size,
                    color=self.point_color,
                )

            # View + cosmetics
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()

            # Optional: make scaling look less distorted
            # (Matplotlib 3D doesn't truly do equal aspect by default)
            self._set_equal_aspect_3d(ax, X)

        # Turn off unused axes
        for ax in axes[len(self.graphs):]:
            ax.set_axis_off()

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def _set_equal_aspect_3d(ax, X):
        """Best-effort equal scaling for 3D plots."""
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