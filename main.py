import numpy as np
import matplotlib.pyplot as plt

from Shapes.HShape import HShape
from Shapes.CShape import CShape
from Graphs.Graphs import Graph

from Algorithms.mdsmapp import (
    mds_mapp,
    build_csr,
    compute_s_stress,
    compute_real_s_stress
)

from Plots.PlotStressVsH import PlotStressVsH


def main() -> None:

    dim = 2
    h_values = [1, 3, 5, 7, 10, 15, 20]

    shapes = [
        HShape(nx=50, ny=50, inner_margin_x=15, inner_margin_y=17, jitter=0.05, seed=0),
        CShape(nx=50, ny=50, inner_margin_x=15, inner_margin_y=15, jitter=0.1, seed=1),
    ]

    h_values_list = []
    stress_values_list = []

    for shape in shapes:

        X = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=10,
            sigma=0.6,
            seed=0
        )

        A = build_csr(graph)

        stress_values = []

        for h in h_values:

            X_emb = mds_mapp(A, h=h, dim=dim)

            stress = compute_real_s_stress(A, X_emb)

            stress_values.append(np.log(stress))

        h_values_list.append(h_values)
        stress_values_list.append(stress_values)

    plotter = PlotStressVsH(
        shapes=shapes,
        h_values_list=h_values_list,
        stress_values_list=stress_values_list
    )

    plotter.plot(figsize=(7, 5))


if __name__ == "__main__":
    main()