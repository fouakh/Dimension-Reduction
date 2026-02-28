import matplotlib.pyplot as plt
import numpy as np

from Shapes.SManifoldShape import SManifoldShape
from Shapes.HollowRectangle import HollowRectangle
from Shapes.SwissRollShape import SwissRollShape


from Plots.PlotPoints3D import PlotPoints3D
from Plots.PlotGraphs3D import PlotGraphs3D
from Graphs.Graphs import Graph


def main():

    base = HollowRectangle(nx=60, ny=60, inner_margin_x=18, inner_margin_y=22, jitter=0.5, seed=1)
    shapes = [
          SManifoldShape(base_shape=base, alpha=4.75),
          SwissRollShape(base, alpha=10, seed=42)
        ]
    

    # plotter = PlotPoints3D(shapes, grid=(1,2))
    # fig, axes = plotter.plot(point_size=5)
    # plt.show()

    graphs = []
    for shape in shapes:
        X = shape.samples()

        graph = Graph()
        graph.build_synth_graph(
            X=X,
            k=15,
            sigma=0,
            seed=0
        )

        graphs.append(graph)

    graph_plotter = PlotGraphs3D(
        graphs=graphs,
        grid=(1, 2),
        edge_alpha=0.3,
        edge_width=0.5,
        show_points=True,
        point_size=1
    )

    graph_plotter.plot(figsize=(10, 8))
    plt.show()


    


if __name__ == "__main__":
    main()
