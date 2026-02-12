import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.HollowRectangle import HollowRectangle
from Shapes.CShape import CShape
from Shapes.HShape import HShape

from Graphs.Graphs import Graph
from Plots.PlotPatches import PlotPatches

from Algorithms.mdsmapp import build_patch


def main():
    shapes = [
        Rectangle(nx=20, ny=20, jitter=0.05, seed=0),
        HollowRectangle(nx=20, ny=20, inner_margin_x=7, inner_margin_y=7, jitter=0.05, seed=1),
        CShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=5, jitter=0.1, seed=2),
        HShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=7, jitter=0.1, seed=3)
    ]

    graphs = []
    patches = []
    centers = []

    h = 1
    rng = np.random.default_rng(42)

    for shape in shapes:
        X = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=5,
            sigma=0.04,
            seed=0
        )

        graphs.append(graph)

        center = rng.integers(0, graph.n_nodes)
        patch = build_patch(graph, center=center, h=h)

        centers.append(center)
        patches.append(patch)

    patch_plotter = PlotPatches(
        shapes=shapes,
        patches=patches,
        centers=centers,
        grid=(1, 4)
    )

    patch_plotter.plot(figsize=(14, 4), point_size=15)
    plt.show()


if __name__ == "__main__":
    main()
