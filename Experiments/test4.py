import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.HollowRectangle import HollowRectangle
from Shapes.CShape import CShape
from Shapes.HShape import HShape

from Graphs.Graphs import Graph

from Algorithms.mdsmapp import (
    build_csr,
    mds_d,
    classical_scaling,
    smacof
)

from Plots.PlotEmbedding import PlotEmbeddingComparison


def main() -> None:

    shapes = [
        Rectangle(nx=20, ny=20, jitter=0.05, seed=0),
        HollowRectangle(nx=20, ny=20, inner_margin_x=7, inner_margin_y=7, jitter=0.05, seed=1),
        CShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=5, jitter=0.1, seed=2),
        HShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=7, jitter=0.1, seed=3)
    ]

    embeddings: list[np.ndarray] = []

    dim: int = 2

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

        D: np.ndarray = mds_d(A, list(graph.nodes()))

        X_init: np.ndarray = classical_scaling(D, dim=dim)

        X_refined: np.ndarray = smacof(D, X_init)

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
