import numpy as np
from typing import Optional
from abc import ABC, abstractmethod

class SyntheticShape(ABC):
    def __init__(self, dim: int, seed: Optional[int] = None):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def samples(self) -> np.ndarray:
        pass

    def n_points(self) -> int:
        return self.noiseless_samples().shape[0]
