import numpy as np
from scipy import integrate, optimize
from typing import Optional
from .SyntheticShape import SyntheticShape


class SwissRollShape(SyntheticShape):
    """
    Swiss roll surface via equation (3.2) of the paper:
        phi_Swiss,alpha(u, v) = ( s(v)*cos(alpha*s(v)),  u,  s(v)*sin(alpha*s(v)) )

    where s(v) is the solution to:
        integral from 0 to s of sqrt(1 + (alpha*t)^2) dt = v
    """

    def __init__(
        self,
        base_shape: SyntheticShape,
        alpha: float = 10.0,
        seed: Optional[int] = None,
    ):
        super().__init__(dim=3, seed=seed)
        self.base_shape = base_shape
        self.alpha = float(alpha)

    def name(self) -> str:
        return f"SwissRoll(alpha={self.alpha})"

    def _arc_length(self, s: float) -> float:
        """
        Computes the arc length integral from 0 to s:
            L(s) = integral_0^s sqrt(1 + (alpha*t)^2) dt

        This has a closed form:
            L(s) = s/2 * sqrt(1 + (alpha*s)^2)
                   + sinh^{-1}(alpha*s) / (2*alpha)
        """
        a = self.alpha
        return (s / 2) * np.sqrt(1 + (a * s) ** 2) + np.arcsinh(a * s) / (2 * a)

    def _compute_s(self, v_values: np.ndarray) -> np.ndarray:
        """
        For each target arc length v, solve L(s) = v numerically to get s(v).
        Uses the closed-form arc length for speed, with brentq root finding.
        """
        # Total arc length when s=1 (upper bound for the search)
        s_max = 2.0  # generous upper bound
        L_max = self._arc_length(s_max)

        s_values = np.empty_like(v_values)
        for i, v in enumerate(v_values):
            if v <= 0:
                s_values[i] = 0.0
            else:
                # Solve: arc_length(s) - v = 0
                s_values[i] = optimize.brentq(
                    lambda s: self._arc_length(s) - v,
                    0.0, s_max,
                    xtol=1e-10
                )
        return s_values

    def samples(self) -> np.ndarray:
        X2 = self.base_shape.samples()  # (N, 2)

        eps = 1e-12
        
        u = (X2[:, 0] - X2[:, 0].min()) / max(X2[:, 0].max() - X2[:, 0].min(), eps)
        v = (X2[:, 1] - X2[:, 1].min()) / max(X2[:, 1].max() - X2[:, 1].min(), eps)

        # Rescale v to [0, L(1)] so it represents actual arc length
        # This ensures s(v) is solved over the full spiral range
        L_total = self._arc_length(1.0)
        v_scaled = v * L_total  # v_scaled ∈ [0, L(1)]

        # Solve for s(v) — the inverse arc length
        s = self._compute_s(v_scaled)  # s ∈ [0, 1]

        a = self.alpha
        X3 = np.empty((len(u), 3), dtype=np.float64)
        X3[:, 0] = s * np.cos(a * s)   # x
        X3[:, 1] = u                    # y (straight tube axis)
        X3[:, 2] = s * np.sin(a * s)   # z

        return X3