import matplotlib.pyplot as plt

class PlotPoints:
    def __init__(self, shapes, colors=None, grid=(1, 1)):
        self.shapes = shapes
        self.colors = colors
        self.grid = grid

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

        for ax, shape, color in zip(axes, self.shapes, colors):
            X = shape.samples()
            ax.scatter(
                X[:, 0],
                X[:, 1],
                s=point_size,
                color=color
            )
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        for ax in axes[len(self.shapes):]:
            ax.axis("off")

        plt.tight_layout()
        return fig, axes
