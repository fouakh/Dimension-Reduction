import numpy as np
from typing import Optional
from .SyntheticShape import SyntheticShape


class HShape(SyntheticShape):
    def __init__(
        self,
        nx: int,
        ny: int,
        inner_margin_x: int,
        inner_margin_y: int,
        spacing: float = 1.0,
        jitter: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(dim=2, seed=seed)

        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers.")

        if inner_margin_x < 0 or inner_margin_y < 0:
            raise ValueError("Inner margins must be non-negative.")

        if 2 * inner_margin_x >= nx or 2 * inner_margin_y >= ny:
            raise ValueError("Inner margins too large.")

        if spacing <= 0:
            raise ValueError("spacing must be positive.")

        if jitter < 0:
            raise ValueError("jitter must be non-negative.")

        self.nx = nx
        self.ny = ny
        self.inner_margin_x = inner_margin_x
        self.inner_margin_y = inner_margin_y
        self.spacing = spacing
        self.jitter = jitter

    def name(self) -> str:
        return "HShape"

    def samples(self) -> np.ndarray:
        xs, ys = np.meshgrid(
            np.arange(self.nx),
            np.arange(self.ny),
            indexing="ij"
        )
        X = np.stack([xs, ys], axis=-1).reshape(-1, 2)

        left_bar = X[:, 0] < self.inner_margin_x
        right_bar = X[:, 0] >= self.nx - self.inner_margin_x

        middle_bar = (
            (X[:, 1] >= self.inner_margin_y) &
            (X[:, 1] < self.ny - self.inner_margin_y)
        )

        X = X[left_bar | right_bar | middle_bar]
        X = X * self.spacing

        if self.jitter > 0.0:
            X = X + self.rng.normal(
                loc=0.0,
                scale=self.jitter,
                size=X.shape
            )

        return X
