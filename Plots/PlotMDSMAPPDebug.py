import os
import numpy as np
import matplotlib.pyplot as plt


class PlotMDSMAPPDebug:

    def __init__(self, X_background: np.ndarray):
        self.X_background = X_background
        self.save_dir = "figs"
        os.makedirs(self.save_dir, exist_ok=True)
        self.step = 0

    def _save(self, fig, name: str):
        fig.savefig(
            os.path.join(self.save_dir, f"{self.step:03d}_{name}.png"),
            dpi=200
        )
        plt.close(fig)
        self.step += 1

    def _base_ax(self):
        fig, ax = plt.subplots()
        ax.scatter(
            self.X_background[:, 0],
            self.X_background[:, 1],
            color="lightgray",
            alpha=0.35,
            s=10
        )
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    # 1
    def first_patch(self, patch_nodes):
        fig, ax = self._base_ax()

        ax.scatter(
            self.X_background[patch_nodes, 0],
            self.X_background[patch_nodes, 1],
            color="blue",
            alpha=0.65,
            s=10
        )

        self._save(fig, "01_first_patch_highlight")

    # 2
    def first_patch_embedding(self, X_local):
        fig, ax = self._base_ax()

        ax.scatter(
            X_local[:, 0],
            X_local[:, 1],
            color="blue",
            alpha=0.65,
            s=10
        )

        self._save(fig, "02_first_patch_embedding")

    # 3 — new patch in ORIGINAL position
    def new_patch_initial_position(
        self,
        X_global,
        patch_nodes,
        overlap_nodes
    ):
        fig, ax = self._base_ax()

        # already embedded
        ax.scatter(
            X_global[:, 0],
            X_global[:, 1],
            color="blue",
            alpha=0.65,
            s=10
        )

        # new patch in original space
        X_new = self.X_background[patch_nodes]
        ax.scatter(
            X_new[:, 0],
            X_new[:, 1],
            color="red",
            alpha=0.65,
            s=10
        )

        # links
        for node in overlap_nodes:
            g = X_global[node]
            l = self.X_background[node]
            ax.plot([g[0], l[0]], [g[1], l[1]], color="black", linewidth=0.5)

        self._save(fig, "03_new_patch_original_position")

    # 4 — after MDS (not aligned)
    def new_patch_after_mds(
        self,
        X_global,
        X_local,
        overlap_nodes,
        idx_local
    ):
        fig, ax = self._base_ax()

        ax.scatter(
            X_global[:, 0],
            X_global[:, 1],
            color="blue",
            alpha=0.65,
            s=10
        )

        ax.scatter(
            X_local[:, 0],
            X_local[:, 1],
            color="red",
            alpha=0.65,
            s=10
        )

        for k, node in enumerate(overlap_nodes):
            g = X_global[node]
            l = X_local[idx_local[k]]
            ax.plot([g[0], l[0]], [g[1], l[1]], color="black", linewidth=0.5)

        self._save(fig, "04_new_patch_after_mds")

    # 5 — after Procrustes
    def after_procrustes(
        self,
        X_global,
        X_aligned,
        overlap_nodes,
        idx_local
    ):
        fig, ax = self._base_ax()

        ax.scatter(
            X_global[:, 0],
            X_global[:, 1],
            color="blue",
            alpha=0.65,
            s=10
        )

        ax.scatter(
            X_aligned[:, 0],
            X_aligned[:, 1],
            color="green",
            alpha=0.65,
            s=10
        )

        for k, node in enumerate(overlap_nodes):
            g = X_global[node]
            l = X_aligned[idx_local[k]]
            ax.plot([g[0], l[0]], [g[1], l[1]], color="black", linewidth=0.5)

        self._save(fig, "05_after_procrustes")
