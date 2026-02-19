import numpy as np
import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.HollowRectangle import HollowRectangle
from Shapes.CShape import CShape
from Shapes.HShape import HShape

from Graphs.Graphs import Graph
from Plots.PlotDistanceError import PlotDistanceError

from Algorithms.mdsmapp import build_csr, mds_d


def true_distance_matrix(X):
    return np.linalg.norm(
        X[:, None, :] - X[None, :, :],
        axis=2
    )


def main():
    shapes = [
        Rectangle(nx=20, ny=20, jitter=0.05, seed=0),
        HollowRectangle(nx=20, ny=20, inner_margin_x=7, inner_margin_y=7, jitter=0.05, seed=1),
        CShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=5, jitter=0.1, seed=2),
        HShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=7, jitter=0.1, seed=3)
    ]

    graphs = []
    D_true_list = []
    D_hat_list = []

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

        A = build_csr(graph)

        D_true: np.ndarray = true_distance_matrix(X)
        D_hat: np.ndarray = mds_d(A, list(graph.nodes()))

        D_true_list.append(D_true)
        D_hat_list.append(D_hat)

    plotter = PlotDistanceError(
        shapes=shapes,
        D_true_list=D_true_list,
        D_hat_list=D_hat_list,
        grid=(1, 4)
    )

    plotter.plot(figsize=(14, 4), point_size=10)
    plt.show()


if __name__ == "__main__":
    main()
