import os
import matplotlib.pyplot as plt
from datetime import datetime


class PlotEmbeddingComparison:

    def __init__(self, shapes, embeddings, colors=None, grid=(1, 1)):

        if len(shapes) != len(embeddings):
            raise ValueError("Shapes and embeddings must have same length.")

        if colors is not None and len(colors) != len(shapes):
            raise ValueError("Number of colors must match number of shapes.")

        n_rows, n_cols = grid
        if n_rows * n_cols < len(shapes):
            raise ValueError("Grid too small for number of shapes.")

        self.shapes = shapes
        self.embeddings = embeddings
        self.colors = colors
        self.grid = grid

        base_dir = "figs"
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        class_name = self.__class__.__name__.lower()

        self.save_dir = os.path.join(
            base_dir,
            f"{class_name}_{timestamp}"
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, figsize=(8, 6), point_size=10):

        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        if self.colors is None:
            cmap = plt.cm.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(self.shapes))]
        else:
            colors = self.colors

        for ax, shape, X_new, color in zip(
            axes, self.shapes, self.embeddings, colors
        ):
            X_old = shape.samples()

            ax.scatter(
                X_old[:, 0],
                X_old[:, 1],
                s=point_size,
                color="lightgray",
                alpha=0.5
            )

            ax.scatter(
                X_new[:, 0],
                X_new[:, 1],
                s=point_size,
                color=color
            )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.shapes):]:
            ax.axis("off")

        fig.tight_layout()

        fig.savefig(
            os.path.join(self.save_dir, "embedding_comparison.png"),
            dpi=200
        )

        plt.close(fig)