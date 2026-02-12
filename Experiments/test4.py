import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.HollowRectangle import HollowRectangle
from Shapes.CShape import CShape
from Shapes.HShape import HShape

from Graphs.Graphs import Graph

from Algorithms.mdsmapp import mds_d, classical_scaling, smacof
from Plots.PlotEmbedding import PlotEmbeddingComparison


def main():
    shapes = [
        Rectangle(nx=20, ny=20, jitter=0.05, seed=0),
        HollowRectangle(nx=20, ny=20, inner_margin_x=7, inner_margin_y=7, jitter=0.05, seed=1),
        CShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=5, jitter=0.1, seed=2),
        HShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=7, jitter=0.1, seed=3)
    ]

    embeddings = []

    for shape in shapes:
        X = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=5,
            sigma=0.04,
            seed=0
        )

        D = mds_d(graph, list(graph.nodes()))

        X_init = classical_scaling(D, dim=2)

        X_refined = smacof(D, X_init)

        embeddings.append(X_refined)

    plotter = PlotEmbeddingComparison(
        shapes=shapes,
        embeddings=embeddings,
        grid=(1, 4)
    )

    plotter.plot(figsize=(14, 4), point_size=15)
    plt.show()


if __name__ == "__main__":
    main()
