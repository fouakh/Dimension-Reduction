import numpy as np

from Shapes.CShape import CShape
from Graphs.Graphs import Graph

from Algorithms.mdsmapp import (
    build_csr,
    build_patch,
    list_pivots,
    mds_d,
    classical_scaling,
    smacof,
    procrustes
)

from Plots.PlotMDSMAPPDebug import PlotMDSMAPPDebug


def main() -> None:

    h = 3
    dim = 2

    shape = CShape(
        nx=50,
        ny=50,
        inner_margin_x=15,
        inner_margin_y=15,
        jitter=0.1,
        seed=1,
    )

    X_background = shape.samples()

    graph = Graph()
    graph.build_synth_graph(
        X=X_background,
        k=10,
        sigma=0.04,
        seed=0
    )

    A = build_csr(graph)

    n = A.shape[0]

    patches = {v: build_patch(A, v, h)[v] for v in range(n)}
    ordered_pivots = list_pivots(A, h)

    debug = PlotMDSMAPPDebug(X_background)

    global_X = np.zeros((n, dim))
    embedded = set()

    # First patch
    first = ordered_pivots[0]
    patch_nodes = patches[first]

    debug.first_patch(patch_nodes)

    D = mds_d(A, patch_nodes)
    X_init = classical_scaling(D, dim)
    X_local = smacof(D, X_init)

    debug.first_patch_embedding(X_local)

    for i, node in enumerate(patch_nodes):
        global_X[node] = X_local[i]

    embedded.update(patch_nodes)

    # Remaining pivots
    for pivot in ordered_pivots[1:]:

        patch_nodes = patches[pivot]

        overlap = [v for v in patch_nodes if v in embedded]

        if len(overlap) < dim:
            continue

        debug.new_patch_initial_position(
            global_X,
            patch_nodes,
            overlap
        )

        D = mds_d(A, patch_nodes)
        X_init = classical_scaling(D, dim)
        X_local = smacof(D, X_init)

        idx_local = [patch_nodes.index(v) for v in overlap]

        debug.new_patch_after_mds(
            global_X,
            X_local,
            overlap,
            idx_local
        )

        X_ref = global_X[overlap]
        X_overlap = X_local[idx_local]

        R, t = procrustes(X_ref, X_overlap)
        X_aligned = X_local @ R + t

        debug.after_procrustes(
            global_X,
            X_aligned,
            overlap,
            idx_local
        )

        for i, node in enumerate(patch_nodes):
            if node not in embedded:
                global_X[node] = X_aligned[i]

        embedded.update(patch_nodes)


if __name__ == "__main__":
    main()
