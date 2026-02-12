import numpy as np
from typing import Optional
from .SyntheticShape import SyntheticShape


class Rectangle(SyntheticShape):
    def __init__(
        self,
        nx: int,
        ny: int,
        spacing: float = 1.0,
        jitter: float = 0.0,
        seed: Optional[int] = None,
    ):
    
        super().__init__(dim=2, seed=seed)

        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers.")

        if spacing <= 0:
            raise ValueError("spacing must be positive.")

        if jitter < 0:
            raise ValueError("jitter must be non-negative.")

        self.nx = nx
        self.ny = ny
        self.spacing = spacing
        self.jitter = jitter

    def name(self) -> str:
        return "Rectangle"

    def samples(self) -> np.ndarray:
        xs, ys = np.meshgrid(
            np.arange(self.nx),
            np.arange(self.ny),
            indexing="ij"
        )
        X = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        X = X * self.spacing

        if self.jitter > 0.0:
            X = X + self.rng.normal(
                loc=0.0,
                scale=self.jitter,
                size=X.shape
            )
        return X