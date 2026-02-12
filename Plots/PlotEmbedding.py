import matplotlib.pyplot as plt


class PlotEmbeddingComparison:
    def __init__(self, shapes, embeddings, colors=None, grid=(1, 1)):
        self.shapes = shapes
        self.embeddings = embeddings
        self.colors = colors
        self.grid = grid

        if len(self.shapes) != len(self.embeddings):
            raise ValueError("Shapes and embeddings must have same length.")

        if self.colors is not None:
            if len(self.colors) != len(self.shapes):
                raise ValueError("Number of colors must match number of shapes.")

        n_rows, n_cols = self.grid
        if n_rows * n_cols < len(self.shapes):
            raise ValueError("Grid too small for number of shapes.")

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

        plt.tight_layout()
        return fig, axes
