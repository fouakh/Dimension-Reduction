import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class PlotMDSMAPPDebug:

    def __init__(self, X_background: np.ndarray, save_name="mdsmapp_debug"):
        self.X_background = X_background

        base_dir = "figs"
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.save_dir = os.path.join(
            base_dir,
            f"{save_name}_{timestamp}"
        )

        os.makedirs(self.save_dir, exist_ok=True)

        self.step = 0

    def _save(self, fig, name: str):
        fig.savefig(
            os.path.join(self.save_dir, f"{self.step:03d}_{name}.png"),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.02
        )
        plt.close(fig)
        self.step += 1


    def _base_ax(self, margin_ratio=1.2, shift=(-13.0, -15.0)):

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(
            self.X_background[:, 0],
            self.X_background[:, 1],
            color="lightgray",
            alpha=0.35,
            s=10
        )

        x_min, x_max = self.X_background[:, 0].min(), self.X_background[:, 0].max()
        y_min, y_max = self.X_background[:, 1].min(), self.X_background[:, 1].max()

        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        dx, dy = shift
        x_center += dx
        y_center += dy

        base_radius = max(x_max - x_min, y_max - y_min) * 0.5
        radius = base_radius * (1 + margin_ratio)

        ax.set_xlim(x_center - radius, x_center + radius)
        ax.set_ylim(y_center - radius, y_center + radius)

        ax.set_aspect("equal", adjustable="box")

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1
        )

        return fig, ax

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

    def new_patch_initial_position(
        self,
        X_global,
        patch_nodes,
        overlap_nodes
    ):
        fig, ax = self._base_ax()

        ax.scatter(
            X_global[:, 0],
            X_global[:, 1],
            color="blue",
            alpha=0.65,
            s=10
        )

        X_new = self.X_background[patch_nodes]
        ax.scatter(
            X_new[:, 0],
            X_new[:, 1],
            color="red",
            alpha=0.65,
            s=10
        )

        for node in overlap_nodes:
            g = X_global[node]
            l = self.X_background[node]
            ax.plot([g[0], l[0]], [g[1], l[1]], color="black", linewidth=0.5)

        self._save(fig, "03_new_patch_original_position")

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