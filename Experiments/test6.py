import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.CShape import CShape

from Graphs.Graphs import Graph

from Algorithms.mdsmapp import mds_mapp, build_csr
from Plots.PlotEmbedding import PlotEmbeddingComparison


def main() -> None:

    shapes = [
        Rectangle(nx=30, ny=30, jitter=0.05, seed=0),
        CShape(nx=30, ny=30, inner_margin_x=10, inner_margin_y=10, jitter=0.1, seed=1),
    ]

    embeddings: list[np.ndarray] = []

    h: int = 5
    dim: int = 2

    for shape in shapes:

        X: np.ndarray = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=10,
            sigma=0.04,
            seed=0
        )

        A = build_csr(graph)

        X_emb = mds_mapp(A, h=h, dim=dim)

        embeddings.append(X_emb)

    plotter = PlotEmbeddingComparison(
        shapes=shapes,
        embeddings=embeddings,
        grid=(1, 2)
    )

    plotter.plot(figsize=(10, 4), point_size=15)
    plt.show()


if __name__ == "__main__":
    main()
