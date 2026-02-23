import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from Graphs.Graphs import Graph
from Algorithms.mdsmapp import (
    mds_mapp,
    build_csr,
    compute_s_stress
)


def main() -> None:

    dim = 2
    k = 100
    h = 10

    print("Loading Digits dataset...")
    digits = load_digits()

    X = digits.data.astype(np.float64)
    y = digits.target

    print("Dataset shape:", X.shape)

    print("Standardizing...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Building graph...")
    graph = Graph()
    graph.build_synth_graph(
        X=X,
        k=k,
        sigma=0.0,
        seed=0
    )

    A = build_csr(graph)

    print("Running MDS-MAP...")
    X_emb = mds_mapp(A, h=h, dim=dim)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_emb[:, 0],
        X_emb[:, 1],
        c=y,
        cmap="tab10",
        s=25,
        alpha=0.85,
        edgecolors="none"
    )

    plt.title(f"Digits - MDS-MAP (h = {h})", fontsize=14)
    plt.axis("equal")
    plt.axis("off")
    plt.colorbar(scatter, ticks=range(10))
    plt.show()


if __name__ == "__main__":
    main()