import matplotlib.pyplot as plt

from Shapes.RectangleShape import Rectangle
from Shapes.HollowRectangle import HollowRectangle
from Shapes.CShape import CShape
from Shapes.HShape import HShape

from Plots.PlotPoints import PlotPoints
from Plots.PlotGraphs import PlotGraphs
from Graphs.Graphs import Graph


def main():
    shapes = [
        Rectangle(nx=20, ny=20, jitter=0.05, seed=0),
        HollowRectangle(nx=20, ny=20, inner_margin_x=7, inner_margin_y=7, jitter=0.05, seed=1),
        CShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=5, jitter=0.1, seed=2),
        HShape(nx=20, ny=20, inner_margin_x=5, inner_margin_y=7, jitter=0.1, seed=3)
    ]

    # plotter = PlotPoints(
    #     shapes=shapes,
    #     colors=["black", "red", "blue", "green"],
    #     grid=(2, 2)
    # )

    # plotter.plot(figsize=(10, 8), point_size=8)
    # plt.show()

    graphs = []
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

    graph_plotter = PlotGraphs(
        graphs=graphs,
        grid=(1, 4),
        edge_alpha=0.3,
        edge_width=0.5,
        show_points=True,
        point_size=1
    )

    graph_plotter.plot(figsize=(10, 8))
    plt.show()


if __name__ == "__main__":
    main()
