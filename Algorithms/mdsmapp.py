import heapq
from collections import deque
from typing import List, Dict, Tuple

import numpy as np
from sklearn.manifold import MDS


def build_patch(graph, center: int, h: int) -> List[int]:
    visited = {center}
    queue = deque([(center, 0)])

    while queue:
        node, depth = queue.popleft()

        if depth == h:
            continue

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return list(visited)


def mds_d(graph, patch: List[int]) -> np.ndarray:
    p = len(patch)
    index = {node: i for i, node in enumerate(patch)}

    D_hat = np.full((p, p), np.inf)
    np.fill_diagonal(D_hat, 0.0)

    patch_set = set(patch)

    for src in patch:
        src_idx = index[src]

        dist = {src: 0.0}
        visited = set()
        heap = [(0.0, src)]

        while heap:
            d_u, u = heapq.heappop(heap)

            if u in visited:
                continue
            visited.add(u)

            for v in graph.neighbors(u):
                if v not in patch_set:
                    continue

                w = graph.weight(u, v)
                new_d = d_u + w

                if v not in dist or new_d < dist[v]:
                    dist[v] = new_d
                    heapq.heappush(heap, (new_d, v))

        for v, d in dist.items():
            j = index[v]
            D_hat[src_idx, j] = d

    # D_hat = 0.5 * (D_hat + D_hat.T)
    return D_hat



def classical_scaling(D: np.ndarray, dim: int) -> np.ndarray:
    n = D.shape[0]

    D2 = D ** 2

    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    eigvals, eigvecs = np.linalg.eigh(B)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.maximum(eigvals[:dim], 0)
    eigvecs = eigvecs[:, :dim]

    X = eigvecs * np.sqrt(eigvals)

    return X


def smacof(
    D: np.ndarray,
    X_init: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
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


def compute_local_embeddings(
    graph,
    h: int,
    dim: int,
    use_smacof: bool = True,
) -> Dict[int, Tuple[List[int], np.ndarray]]:

    local_embeddings = {}

    for center in graph.nodes():

        patch = build_patch(graph, center=center, h=h)

        if len(patch) < dim + 1:
            continue

        D_patch = mds_d(graph, patch)

        X_init = classical_scaling(D_patch, dim=dim)

        if use_smacof:
            X_local = smacof(D_patch, X_init)
        else:
            X_local = X_init

        local_embeddings[center] = (patch, X_local)

    return local_embeddings


def select_pivot_patches(
    patches: Dict[int, Tuple[List[int], object]]
) -> List[int]:

    if not patches:
        return []

    remaining = set(patches.keys())
    ordered_pivots = []

    patch_sets = {
        c: set(patches[c][0])
        for c in patches
    }

    first = max(
        remaining,
        key=lambda c: len(patch_sets[c])
    )

    ordered_pivots.append(first)
    remaining.remove(first)

    covered_nodes = set(patch_sets[first])

    while remaining:

        best_center = max(
            remaining,
            key=lambda c: len(patch_sets[c] & covered_nodes)
        )

        overlap = len(patch_sets[best_center] & covered_nodes)

        if overlap == 0:
            best_center = max(
                remaining,
                key=lambda c: len(patch_sets[c])
            )

        ordered_pivots.append(best_center)
        remaining.remove(best_center)
        covered_nodes.update(patch_sets[best_center])

    return ordered_pivots


def procrustes(
    X_ref: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if X_ref.shape != X.shape:
        raise ValueError("X_ref and X must have the same shape.")

    mu_ref = X_ref.mean(axis=0)
    mu = X.mean(axis=0)

    X_ref_c = X_ref - mu_ref
    X_c = X - mu

    C = X_c.T @ X_ref_c
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = mu_ref - mu @ R

    X_aligned = X @ R + t

    return X_aligned, R, t



def align_patches(
    local_embeddings: Dict[int, np.ndarray],
    patches: Dict[int, List[int]],
    pivots: List[int],
) -> Dict[int, np.ndarray]:
    pass


def assemble_global_embedding(
    aligned_embeddings: Dict[int, np.ndarray],
    patches: Dict[int, List[int]],
    n_nodes: int,
) -> np.ndarray:
    pass


def compute_stress(graph, X: np.ndarray) -> float:
    pass
