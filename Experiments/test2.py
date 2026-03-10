# import numpy as np
# import matplotlib.pyplot as plt

# from Shapes.HShape import HShape
# from Shapes.CShape import CShape
# from Graphs.Graphs import Graph

# from Algorithms.mdsmapp import (
#     mds_mapp,
#     build_csr,
#     compute_s_stress,
#     compute_real_s_stress
# )


# def main() -> None:

#     dim      = 2
#     h_values = [1, 3, 4, 5, 7, 10]

#     shapes = [
#         HShape(nx=30, ny=30, inner_margin_x=10, inner_margin_y=10, jitter=0.05, seed=0),
#         CShape(nx=30, ny=30, inner_margin_x=10, inner_margin_y=10, jitter=0.1,  seed=1),
#     ]

#     COLORS = {
#         "HShape": {"real": "#1E3A8A", "obs": "#60A5FA"},
#         "CShape": {"real": "#FF4500", "obs": "#FFA500"},
#     }

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     for ax, shape in zip(axes, shapes):

#         name = shape.__class__.__name__
#         c    = COLORS[name]

#         X = shape.samples()

#         graph = Graph()
#         graph.build_synth_graph(X=X, k=10, sigma=0.04, seed=0)

#         A = build_csr(graph)

#         real_stress_values = []
#         obs_stress_values  = []

#         for h in h_values:
#             X_emb = mds_mapp(A, h=h, dim=dim)
#             real_stress_values.append(compute_real_s_stress(A, X_emb))
#             obs_stress_values.append(compute_s_stress(X, X_emb))

#         ax.semilogy(h_values, real_stress_values,
#                     color=c["real"], linewidth=2,
#                     marker="o", markersize=6,
#                     label="Observed s-stress $\\mathcal{S}(Y_h)$")

#         ax.semilogy(h_values, obs_stress_values,
#                     color=c["obs"], linewidth=2,
#                     linestyle="--", marker="s", markersize=6,
#                     label="Real s-stress $\\mathcal{S}^*(Y_h)$")

#         ax.set_xlabel("Number of hops $h$", fontsize=12)
#         ax.set_ylabel("S-stress (log scale)", fontsize=12)
#         ax.set_title(name.replace("Shape", "-shape"), fontsize=13)
#         ax.legend(fontsize=10)
#         ax.grid(True, which="both", alpha=0.3)

#     plt.suptitle("Real vs Observed S-stress", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("stress_comparison.png", dpi=150)
#     plt.show()


# if __name__ == "__main__":
#     main()

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


def main() -> None:

    dim      = 2
    h_values = [1, 2, 3, 4, 5, 7, 10]
    noise_levels = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    shape_configs = [
        HShape(nx=30, ny=30, inner_margin_x=10, inner_margin_y=10, jitter=0.05, seed=0),
        CShape(nx=30, ny=30, inner_margin_x=10, inner_margin_y=10, jitter=0.1,  seed=1),
    ]

    COLORS = {
        "HShape": {"stress": "#1E3A8A", "real": "#60A5FA"},
        "CShape": {"stress": "#FF4500", "real": "#FFA500"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, shape_template in zip(axes, shape_configs):

        name = shape_template.__class__.__name__
        c    = COLORS[name]

        h_opt_stress = []
        h_opt_real   = []

        for sigma in noise_levels:

            shape = shape_template.__class__(
                nx=30, ny=30,
                inner_margin_x=10, inner_margin_y=10,
                jitter=0.05, seed=0
            )

            X = shape.samples()

            graph = Graph()
            graph.build_synth_graph(X=X, k=10, sigma=sigma, seed=0)

            A = build_csr(graph)

            stress_values = []
            real_stress_values = []

            for h in h_values:
                X_emb = mds_mapp(A, h=h, dim=dim)
                stress_values.append(compute_real_s_stress(A, X_emb))
                real_stress_values.append(compute_s_stress(X, X_emb))

            h_opt_stress.append(h_values[np.argmin(stress_values)])
            h_opt_real.append(h_values[np.argmin(real_stress_values)])

        ax.plot(noise_levels, h_opt_stress,
                color=c["stress"], linewidth=2,
                marker="o", markersize=6,
                label="$h^*$ by observed s-stress")

        ax.plot(noise_levels, h_opt_real,
                color=c["real"], linewidth=2,
                linestyle="--", marker="s", markersize=6,
                label="$h^*$ by real s-stress")

        ax.set_xlabel("Noise level $\\sigma$", fontsize=12)
        ax.set_ylabel("Optimal $h^*$", fontsize=12)
        ax.set_title(name.replace("Shape", "-shape"), fontsize=13)
        ax.set_yticks(h_values)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Optimal $h^*$ vs noise level", fontsize=14)
    plt.tight_layout()
    plt.savefig("optimal_h_vs_noise.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()