import numpy as np

from Shapes.HShape import HShape
from Shapes.CShape import CShape
from Graphs.Graphs import Graph

from Algorithms.mdsmapp import (
    mds_mapp,
    build_csr,
    compute_s_stress
)

from Plots.PlotEmbeddingStress import PlotEmbeddingStress


def run_for_shape(shape, h_values, dim=2):

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
    stress_values = []

    for h in h_values:

        X_emb = mds_mapp(A, h=h, dim=dim)

        stress = compute_s_stress(X, X_emb)

        embeddings.append(X_emb)
        stress_values.append(stress)

    plotter = PlotEmbeddingStress(
        shape=shape,
        h_values=h_values,
        embeddings=embeddings,
        stress_values=stress_values,
        save_name="test_10"
    )

    plotter.plot()


def main() -> None:

    dim = 2
    h_values = [1, 3, 6, 9, 12, 15]

    shapes = [
        HShape(
            nx=50,
            ny=50,
            inner_margin_x=15,
            inner_margin_y=17,
            jitter=0.05,
            seed=0
        )
    ]

    for shape in shapes:
        run_for_shape(shape, h_values, dim)


if __name__ == "__main__":
    main()