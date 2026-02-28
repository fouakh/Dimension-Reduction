import matplotlib.pyplot as plt

class PlotPoints3D:
    def __init__(self, shapes, colors=None, grid=(1, 1)):
        self.shapes = shapes
        self.colors = colors
        self.grid = grid

        if self.colors is not None and len(self.colors) != len(self.shapes):
            raise ValueError("Number of colors must match number of shapes.")

        n_rows, n_cols = self.grid
        if n_rows * n_cols < len(self.shapes):
            raise ValueError("Grid too small for number of shapes.")

    def plot(self, figsize=(8, 6), point_size=10, elev=10, azim=-60):
        n_rows, n_cols = self.grid
        fig = plt.figure(figsize=figsize)

        # Create a list of 3D axes
        axes = []
        for i in range(n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            axes.append(ax)

        if self.colors is None:
            cmap = plt.cm.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(self.shapes))]
        else:
            colors = self.colors

        for ax, shape, color in zip(axes, self.shapes, colors):
            X = shape.samples()
            if X.shape[1] != 3:
                raise ValueError(f"Expected (N,3) points for 3D plotting, got {X.shape}")
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=point_size, color=color)
            ranges = X.max(axis=0) - X.min(axis=0)
            ax.set_box_aspect(ranges)   # makes the 3 axes use comparable scaling

            # Same spirit as your 2D version: no axes/ticks
            ax.set_axis_off()

        for ax in axes[len(self.shapes):]:
            ax.set_axis_off()

        plt.tight_layout()
        return fig, axes