import matplotlib.pyplot as plt
import numpy as np


class PlotGraphs:
    def __init__(
        self,
        graphs,
        grid=(1, 1),
        edge_color="green",
        edge_alpha=0.3,
        edge_width=0.5,
        show_points=True,
        point_size=5,
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

    def plot(self, figsize=(10, 8)):
        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for ax, graph in zip(axes, self.graphs):
            X = graph.X

            for i, j in graph.edges():
                xi, yi = X[i]
                xj, yj = X[j]
                ax.plot(
                    [xi, xj],
                    [yi, yj],
                    color=self.edge_color,
                    alpha=self.edge_alpha,
                    linewidth=self.edge_width
                )

            if self.show_points:
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    s=self.point_size,
                    color=self.point_color
                )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.graphs):]:
            ax.axis("off")

        plt.tight_layout()
        return fig, axes
