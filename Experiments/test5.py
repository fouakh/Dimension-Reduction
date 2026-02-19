import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.CShape import CShape

from Graphs.Graphs import Graph

from Algorithms.mdsmapp import (
    build_csr,
    build_patch,
    list_pivots
)

from Plots.PlotPatchesProgression import PlotPatchesProgression


def main() -> None:

    shapes = [
        Rectangle(nx=40, ny=40, jitter=0.05, seed=0),
        CShape(nx=40, ny=40, inner_margin_x=10, inner_margin_y=10, jitter=0.1, seed=2)
    ]

    ordered_centers_list: list[list[int]] = []
    patches_list: list[dict[int, dict[int, list[int]]]] = []

    h: int = 5

    for shape in shapes:

        X: np.ndarray = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=5,
            sigma=0.04,
            seed=0
        )

        A = build_csr(graph)

        n = A.shape[0]
        patches = {}

        for v in range(n):
            patches[v] = build_patch(A, v, h)

        ordered_centers = list_pivots(A, h)

        ordered_centers_list.append(ordered_centers)
        patches_list.append(patches)

    plotter = PlotPatchesProgression(
        shapes=shapes,
        ordered_centers_list=ordered_centers_list,
        patches_list=patches_list,
        grid=(1, 2)
    )

    plotter.plot(figsize=(12, 5), point_size=32)
    plt.show()


if __name__ == "__main__":
    main()


