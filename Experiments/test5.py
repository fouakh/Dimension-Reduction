import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.CShape import CShape

from Graphs.Graphs import Graph

from Algorithms.mdsmapp import (
    compute_local_embeddings,
    select_pivot_patches
)

from Plots.PlotPatchesProgression import PlotPatchesProgression


def main():

    shapes = [
        Rectangle(nx=40, ny=40, jitter=0.05, seed=0),
        CShape(nx=40, ny=40, inner_margin_x=10, inner_margin_y=10, jitter=0.1, seed=2)
    ]

    ordered_centers_list = []
    patches_list = []

    h = 5
    dim = 2

    for shape in shapes:

        X = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=5,
            sigma=0.04,
            seed=0
        )

        local_embeddings = compute_local_embeddings(
            graph=graph,
            h=h,
            dim=dim,
            use_smacof=False
        )

        ordered_centers = select_pivot_patches(local_embeddings)

        ordered_centers_list.append(ordered_centers)
        patches_list.append(local_embeddings)

    plotter = PlotPatchesProgression(
        shapes=shapes,
        ordered_centers_list=ordered_centers_list,
        patches_list=patches_list,
        grid=(1, 2)
    )

    plotter.plot(figsize=(12, 5), point_size=12)
    plt.show()


if __name__ == "__main__":
    main()
