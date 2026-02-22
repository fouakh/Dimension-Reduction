import numpy as np
import matplotlib.pyplot as plt

from Shapes.CShape import CShape
from Graphs.Graphs import Graph

from Algorithms.mdsmapp import mds_mapp, build_csr
from Plots.PlotEmbedding import PlotEmbeddingComparison


def main() -> None:
    dim = 2
    h_values = [2, 3, 5, 7]  

    shape = CShape(
        nx=30,
        ny=30,
        inner_margin_x=10,
        inner_margin_y=10,
        jitter=0.1,
        seed=1,
    )

    X = shape.samples()

    graph = Graph()
    graph.build_synth_graph(
        X=X,
        k=10,
        sigma=0.04,
        seed=0
    )

    A = build_csr(graph)

    embeddings = []
    shapes = []

    for h in h_values:

        print(f"Computing embedding for h = {h}")

        X_emb = mds_mapp(A, h=h, dim=dim)

        embeddings.append(X_emb)
        shapes.append(shape)

    plotter = PlotEmbeddingComparison(
        shapes=shapes,
        embeddings=embeddings,
        grid=(1, len(h_values)) 
    )

    plotter.plot(figsize=(4 * len(h_values), 4), point_size=12)
    plt.show()


if __name__ == "__main__":
    main()