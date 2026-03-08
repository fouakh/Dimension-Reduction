import os
import matplotlib.pyplot as plt
from datetime import datetime


class PlotEmbeddingComparison:

    def __init__(self, shapes, embeddings, grid=(1, 1), save_name="embedding_comparison"):

        if len(shapes) != len(embeddings):
            raise ValueError("Shapes and embeddings must have same length.")

        n_rows, n_cols = grid
        if n_rows * n_cols < len(shapes):
            raise ValueError("Grid too small for number of shapes.")

        self.shapes = shapes
        self.embeddings = embeddings
        self.grid = grid

        base_dir = "figs"
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.save_dir = os.path.join(
            base_dir,
            f"{save_name}_{timestamp}"
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, figsize=(8, 6), point_size=10):

        n_rows, n_cols = self.grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for ax, shape, X_new in zip(
            axes, self.shapes, self.embeddings
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
                color="blue"
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