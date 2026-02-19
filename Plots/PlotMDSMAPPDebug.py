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
            s=15
        )
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    # ----------------------------------------------------
    # 1 - First patch highlighted
    # ----------------------------------------------------
    def first_patch(self, patch_nodes):
        fig, ax = self._base_ax()

        ax.scatter(
            self.X_background[patch_nodes, 0],
            self.X_background[patch_nodes, 1],
            color="red",
            s=25
        )

        self._save(fig, "01_first_patch_highlight")

    # ----------------------------------------------------
    # 2 - First patch embedded (MDS+SMACOF result)
    # ----------------------------------------------------
    def first_patch_embedding(self, X_local):
        fig, ax = self._base_ax()

        ax.scatter(
            X_local[:, 0],
            X_local[:, 1],
            color="blue",
            s=25
        )

        self._save(fig, "02_first_patch_embedding")

    # ----------------------------------------------------
    # 3 - New patch before alignment
    # ----------------------------------------------------
    def new_patch_before_alignment(
        self,
        X_global,
        X_local,
        overlap_global,
        overlap_local
    ):
        fig, ax = self._base_ax()

        # previous embedded
        ax.scatter(
            X_global[:, 0],
            X_global[:, 1],
            color="blue",
            s=25
        )

        # new local patch
        ax.scatter(
            X_local[:, 0],
            X_local[:, 1],
            color="orange",
            s=25
        )

        # overlap highlighted
        ax.scatter(
            overlap_global[:, 0],
            overlap_global[:, 1],
            color="green",
            s=35
        )

        # links
        for g, l in zip(overlap_global, overlap_local):
            ax.plot(
                [g[0], l[0]],
                [g[1], l[1]],
                color="black",
                linewidth=1
            )

        self._save(fig, "03_new_patch_before_alignment")

    # ----------------------------------------------------
    # 4 - After local MDS+SMACOF (still unaligned)
    # ----------------------------------------------------
    def new_patch_after_mds(self, *args):
        self.new_patch_before_alignment(*args)

    # ----------------------------------------------------
    # 5 - After Procrustes alignment
    # ----------------------------------------------------
    def after_alignment(
        self,
        X_global,
        X_aligned,
        overlap_global,
        overlap_aligned
    ):
        fig, ax = self._base_ax()

        ax.scatter(
            X_global[:, 0],
            X_global[:, 1],
            color="blue",
            s=25
        )

        ax.scatter(
            X_aligned[:, 0],
            X_aligned[:, 1],
            color="purple",
            s=25
        )

        ax.scatter(
            overlap_global[:, 0],
            overlap_global[:, 1],
            color="green",
            s=35
        )

        for g, l in zip(overlap_global, overlap_aligned):
            ax.plot(
                [g[0], l[0]],
                [g[1], l[1]],
                color="black",
                linewidth=1
            )

        self._save(fig, "04_after_procrustes")
