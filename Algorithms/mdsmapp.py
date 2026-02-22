from typing import List, Dict, Set, Tuple

import numpy as np
from sklearn.manifold import MDS
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import breadth_first_order


def build_patch(A: csr_matrix, center: int, h: int) -> Dict[int, List[int]]:
    order, predecessors = breadth_first_order(
        csgraph=A,
        i_start=center,
        directed=False,
        return_predecessors=True
    )

    depth = np.full(A.shape[0], -1)
    depth[center] = 0

    for node in order[1:]:
        parent = predecessors[node]
        depth[node] = depth[parent] + 1

    nodes = [node for node in order if depth[node] <= h]

    return {center: nodes}


def build_csr(graph: "Graph") -> csr_matrix:
    n: int = graph.n_nodes

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for (i, j), w in graph._weights.items():
        rows.append(i)
        cols.append(j)
        data.append(w)

    return csr_matrix((data, (rows, cols)), shape=(n, n))


def mds_d(A: csr_matrix, patch: List[int]) -> np.ndarray:
    D_sub: np.ndarray = dijkstra(
        csgraph=A,
        directed=False,
        indices=patch
    )

    D_patch: np.ndarray = D_sub[:, patch]

    return 0.5 * (D_patch + D_patch.T)



def classical_scaling(D: np.ndarray, dim: int) -> np.ndarray:
    n = D.shape[0]

    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    eigvals, eigvecs = eigsh(B, k=dim, which="LA")

    eigvals = np.maximum(eigvals, 0)

    X = eigvecs * np.sqrt(eigvals)

    return X


def smacof(
    D: np.ndarray,
    X_init: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> np.ndarray:

    dim = X_init.shape[1]

    mds = MDS(
        n_components=dim,
        metric=True,
        max_iter=max_iter,
        eps=tol,
        dissimilarity="precomputed",
        n_init=1
    )

    return mds.fit_transform(D, init=X_init)


def select_patch(
    patches: Dict[int, List[int]],
    mapped_nodes: Set[int],
    used_centers: Set[int],
    alpha: float = 0.4
) -> int:

    best_center = None
    best_score = -np.inf

    for center, nodes in patches.items():

        if center in used_centers:
            continue

        nodes_set = set(nodes)

        overlap = len(nodes_set & mapped_nodes)
        unmapped = len(nodes_set - mapped_nodes)

        if unmapped == 0:
            continue

        score = (
            alpha * np.log(overlap + 1) +
            (1 - alpha) * np.log(unmapped + 1)
        )

        if score > best_score:
            best_score = score
            best_center = center

    if best_center is None:
        raise ValueError("No valid patch remaining.")

    return best_center



def list_pivots(A: csr_matrix, h: int) -> List[int]:

    n = A.shape[0]
    patches: Dict[int, List[int]] = {}
    for v in range(n):
        patches[v] = build_patch(A, v, h)[v]

    pivot0 = max(patches.keys(), key=lambda v: len(patches[v]))

    pivots: List[int] = [pivot0]
    used_centers: Set[int] = {pivot0}
    mapped_nodes: Set[int] = set(patches[pivot0])

    while len(mapped_nodes) < n:

        next_pivot = select_patch(
            patches=patches,
            mapped_nodes=mapped_nodes,
            used_centers=used_centers 
        )

        pivots.append(next_pivot)
        used_centers.add(next_pivot)

        mapped_nodes.update(patches[next_pivot])

    return pivots


def procrustes(
    X_ref: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    if X_ref.shape != X.shape:
        raise ValueError("X_ref and X must have the same shape.")

    mu_ref = X_ref.mean(axis=0)
    mu = X.mean(axis=0)

    X_ref_c = X_ref - mu_ref
    X_c = X - mu

    C = X_c.T @ X_ref_c
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt

    t = mu_ref - mu @ R

    return R, t
   

def assemble_embedding(
    A: csr_matrix,
    patches: Dict[int, List[int]],
    ordered_pivots: List[int],
    dim: int = 2
) -> np.ndarray:

    n = A.shape[0]

    global_X = np.zeros((n, dim))
    embedded: Set[int] = set()

    first = ordered_pivots[0]
    patch_nodes = patches[first]

    D = mds_d(A, patch_nodes)
    X_init = classical_scaling(D, dim)
    X_local = smacof(D, X_init)

    for i, node in enumerate(patch_nodes):
        global_X[node] = X_local[i]

    embedded.update(patch_nodes)

    for pivot in ordered_pivots[1:]:

        patch_nodes = patches[pivot]

        D = mds_d(A, patch_nodes)

        X_init = classical_scaling(D, dim)
        X_local = smacof(D, X_init)

        overlap = list(set(patch_nodes) & embedded)

        if len(overlap) < dim:
            continue

        idx_local = [patch_nodes.index(v) for v in overlap]

        X_ref = global_X[overlap]
        X_overlap = X_local[idx_local]

        R, t = procrustes(X_ref, X_overlap)

        X_aligned = X_local @ R + t

        for i, node in enumerate(patch_nodes):
            if node not in embedded:
                global_X[node] = X_aligned[i]

        embedded.update(patch_nodes)

    return global_X


def mds_mapp(
    A: csr_matrix,
    h: int,
    dim: int = 2
) -> np.ndarray:

    n = A.shape[0]

    patches: Dict[int, List[int]] = {}
    for v in range(n):
        patches[v] = build_patch(A, v, h)[v]

    ordered_pivots = list_pivots(A, h)

    X = assemble_embedding(
        A=A,
        patches=patches,
        ordered_pivots=ordered_pivots,
        dim=dim
    )

    D_full = dijkstra(
        csgraph=A,
        directed=False
    )

    D_full = 0.5 * (D_full + D_full.T)

    X_refined = smacof(
        D_full,
        X,
        max_iter=1,
        tol=1e-4
    )

    # return X
    return X_refined
