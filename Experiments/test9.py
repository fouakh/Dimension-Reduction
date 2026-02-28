import matplotlib.pyplot as plt
import numpy as np

from Shapes.SManifoldShape import SManifoldShape
from Shapes.HollowRectangle import HollowRectangle
from Shapes.SwissRollShape import SwissRollShape
from Shapes.RectangleShape import Rectangle


from Plots.PlotPoints import PlotPoints
from Plots.PlotEmbedding3D import PlotEmbeddingComparison3Dwith2D
from Graphs.Graphs import Graph
from Algorithms.mdsmapp import (
    build_csr,
    mds_mapp
)


def main():

    base = HollowRectangle(nx=60, ny=60, inner_margin_x=18, inner_margin_y=22, jitter=0.5, seed=1)
    # base = Rectangle(nx=60, ny=60, jitter=0.5, seed=0)
    s = SManifoldShape(base_shape=base, alpha=4.75)
    # s = SwissRollShape(base, alpha=10, seed=42)

    X = s.samples()

    graph = Graph()
    graph.build_synth_graph(
        X=X,
        k=15,
        sigma=0,
        seed=0
    )

    A = build_csr(graph)
    output = mds_mapp(A, h=10, dim = 2)

    plotter = PlotEmbeddingComparison3Dwith2D([s], [output], grid=(1,1))
    fig, axes = plotter.plot(point_size=5)
    plt.show()


    


    


if __name__ == "__main__":
    main()
