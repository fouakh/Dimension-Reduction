import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class PlotEmbeddingStress:

    def __init__(
        self,
        shape,
        h_values,
        embeddings,
        stress_values,
        save_name="embedding_stress"
    ):

        if not (
            len(h_values) ==
            len(embeddings) ==
            len(stress_values)
        ):
            raise ValueError("Inputs must have same length.")

        self.shape = shape
        self.h_values = np.array(h_values)
        self.embeddings = embeddings
        self.stress_values = np.array(stress_values)

        base_dir = "figs"
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.save_dir = os.path.join(
            base_dir,
            f"{save_name}_{timestamp}"
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, figsize_per_h=3, point_size=10, interp_resolution=1000):

        n_h = len(self.h_values)

        fig = plt.figure(
            figsize=(figsize_per_h * n_h, 5)
        )

        gs = fig.add_gridspec(
            2,
            n_h,
            height_ratios=[4, 1]
        )

        X_true = self.shape.samples()
        stress = self.stress_values

        # Normalisation pour heatmap
        stress_norm = (stress - stress.min()) / (
            stress.max() - stress.min() + 1e-12
        )

        idx_opt = np.argmin(stress)

        # ---------------- Embeddings ----------------
        for i in range(n_h):

            ax = fig.add_subplot(gs[0, i])
            X_new = self.embeddings[i]

            ax.scatter(
                X_true[:, 0],
                X_true[:, 1],
                s=point_size,
                color="lightgray",
                alpha=0.4
            )

            ax.scatter(
                X_new[:, 0],
                X_new[:, 1],
                s=point_size,
                color="blue"
            )

            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            ax.set_title(
                f"h={self.h_values[i]}\nS={stress[i]:.4f}",
                fontsize=10
            )

        # ---------------- Heat band aligned ----------------

        ax_band = fig.add_subplot(gs[1, :])

        # Colonnes discrètes
        x_positions = np.arange(n_h)

        # Interpolation sur indices
        x_dense = np.linspace(0, n_h - 1, interp_resolution)
        stress_dense = np.interp(
            x_dense,
            x_positions,
            stress_norm
        )

        heat_band = np.tile(stress_dense, (30, 1))

        ax_band.imshow(
            heat_band,
            aspect="auto",
            cmap="viridis",
            extent=[0, n_h - 1, 0, 1],
            origin="lower"
        )

        ax_band.set_xlim(0, n_h - 1)
        ax_band.set_ylim(0, 1)
        ax_band.set_yticks([])

        # Ticks parfaitement alignés
        ax_band.set_xticks(x_positions)
        ax_band.set_xticklabels(self.h_values)
        ax_band.set_xlabel("h")

        # Marqueur optimal
        ax_band.scatter(
            idx_opt,
            0.5,
            color="red",
            marker="x",
            s=120,
            zorder=5
        )

        for spine in ax_band.spines.values():
            spine.set_visible(False)

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.save_dir,
                "embedding_stress.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)