import numpy as np
from typing import Optional, Iterable, Tuple
from collections import defaultdict


class Graph:
    def __init__(self):
        self.X = None
        self.n = 0

        self._adj = defaultdict(list)
        self._weights = {}

    # ---------- Basic graph API ----------

    @property
    def n_nodes(self) -> int:
        return self.n

    def nodes(self):
        return range(self.n)

    def neighbors(self, i: int):
        return self._adj[i]

    def weight(self, i: int, j: int) -> float:
        return self._weights[(i, j)]

    def edges(self) -> Iterable[Tuple[int, int]]:
        return self._weights.keys()

    # ---------- Graph construction methods ----------

    def add_edge(self, i: int, j: int, w: float):
        if i == j:
            return

        if (i, j) in self._weights:
            return

        self._adj[i].append(j)
        self._adj[j].append(i)

        self._weights[(i, j)] = w
        self._weights[(j, i)] = w

    def build_synth_graph(
        self,
        X: np.ndarray,
        k: int,
        sigma: float = 0.0,
        seed: Optional[int] = None,
    ):
        if X.ndim != 2:
            raise ValueError("X must be of shape (n, dim).")

        if k <= 0:
            raise ValueError("k must be positive.")

        if sigma < 0:
            raise ValueError("sigma must be non-negative.")

        self.X = X
        self.n = X.shape[0]

        rng = np.random.default_rng(seed)

        D = np.linalg.norm(
            X[:, None, :] - X[None, :, :],
            axis=2
        )

        for i in range(self.n):
            nn_idx = np.argsort(D[i])[1:k + 1]
            for j in nn_idx:
                if (i, j) in self._weights or (j, i) in self._weights:
                    continue

                eta = 0.0
                if sigma > 0.0:
                    eta = rng.uniform(-sigma, sigma)

                dij = (1.0 + eta) * D[i, j]
                self.add_edge(i, j, dij)

    def build_from_edges(
        self,
        n_nodes: int,
        edges: Iterable[Tuple[int, int, float]],
    ):
        self.n = n_nodes

        for i, j, w in edges:
            self.add_edge(i, j, w)
