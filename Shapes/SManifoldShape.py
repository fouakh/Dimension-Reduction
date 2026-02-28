import numpy as np
from typing import Optional
from .SyntheticShape import SyntheticShape

class SManifoldShape(SyntheticShape):
    """
    S-shaped cylindrical surface obtained by embedding (u,v) in [0,1]^2 into R^3:
        x = sin(alpha*v)/alpha
        y = u
        z = (cos(alpha*v) - 1)/alpha
    """

    def __init__(self, 
                 base_shape : SyntheticShape, 
                 alpha=10.0, 
                 jitter: float = 0.0, 
                 seed: Optional[int] =None
                 ):
        
        super().__init__(dim=3, seed=seed)
        self.base_shape = base_shape
        self.alpha = float(alpha)
        self.jitter = float(jitter)

    def name(self):
        return f"SManifold(alpha={self.alpha})"

    def samples(self):
        X2 = self.base_shape.samples()  
        x = X2[:, 0]
        y = X2[:, 1]


        eps = 1e-12
        u = (x - x.min()) / max(x.max() - x.min(), eps)
        v = (y - y.min()) / max(y.max() - y.min(), eps)

        a = self.alpha

        t = a * (2.0 * v - 1.0)

        
        X3 = np.empty((X2.shape[0], 3), dtype=np.float64)
        X3[:, 0] = np.sin(t) * 2
        X3[:, 1] = u * 2
        X3[:, 2] = np.sign(t) * (np.cos(t) - 1.0) 


        if self.jitter > 0:
            X3 += self.rng.normal(0.0, self.jitter, size=X3.shape)

        return X3

    