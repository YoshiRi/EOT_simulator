"""GGIW (Gamma Gaussian Inverse-Wishart) single-target state.

State representation for the GGIW distribution, which is the
conjugate prior for extended object tracking with Poisson measurements
scattered uniformly over an ellipsoidal extent.

References:
    Granström, K. & Orguner, U. (2012). A PHD filter for tracking
    multiple extended targets. IEEE Transactions on Signal Processing.

    Granström, K. & Baum, M. (2022). A Tutorial on Multiple Extended
    Object Tracking. TechRxiv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Spatial dimension (2D tracking)
_D = 2

# Minimum valid IW degrees of freedom: v > 2d + 2
_DOF_MIN = 2 * _D + 2  # = 6


@dataclass
class GGIWState:
    """State of a single GGIW target.

    The joint distribution is:
        p(x, X, γ) = N(x; m, P) · IW(X; v, V) · Gamma(γ; α, β)

    Attributes:
        m: Kinematic mean [px, py, vx, vy], shape (4,).
        P: Kinematic covariance, shape (4, 4).
        v: IW degrees of freedom. Must satisfy v > 2d+2 = 6 for d=2.
        V: IW scale matrix, shape (2, 2). Symmetric positive definite.
        alpha: Gamma shape parameter α > 0 (expected count ~ α/β per frame).
        beta:  Gamma rate  parameter β > 0.
    """

    m: np.ndarray  # (4,)
    P: np.ndarray  # (4, 4)
    v: float
    V: np.ndarray  # (2, 2)
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        self.m = np.asarray(self.m, dtype=float).ravel()
        self.P = np.asarray(self.P, dtype=float)
        self.V = np.asarray(self.V, dtype=float)
        if self.v <= _DOF_MIN:
            raise ValueError(f"IW degrees of freedom v must be > {_DOF_MIN}, got {self.v}")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Gamma parameters alpha and beta must be > 0")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def extent_mean(self) -> np.ndarray:
        """Expected extent matrix E[X] = V / (v - 2d - 2), shape (2, 2)."""
        return self.V / (self.v - _DOF_MIN)

    @property
    def rate_mean(self) -> float:
        """Expected measurement rate E[γ] = α / β."""
        return self.alpha / self.beta

    @property
    def position(self) -> np.ndarray:
        """Estimated position [px, py], shape (2,)."""
        return self.m[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity [vx, vy], shape (2,)."""
        return self.m[2:].copy()

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_position(
        cls,
        position: np.ndarray,
        *,
        pos_var: float = 1e2,
        vel_var: float = 1e4,
        extent_init: np.ndarray | None = None,
        dof_init: float = 20.0,
        alpha: float = 5.0,
        beta: float = 1.0,
    ) -> "GGIWState":
        """Create a birth state from an initial position estimate.

        Args:
            position: Initial position [px, py], shape (2,).
            pos_var:  Initial position variance (diagonal).
            vel_var:  Initial velocity variance (diagonal).
            extent_init: Initial IW scale matrix V. Defaults to identity * (dof_init - 6).
            dof_init: Initial degrees of freedom (> 6).
            alpha:    Gamma shape.
            beta:     Gamma rate.
        """
        p = np.asarray(position, dtype=float).ravel()
        m = np.array([p[0], p[1], 0.0, 0.0])
        P = np.diag([pos_var, pos_var, vel_var, vel_var])
        if extent_init is None:
            # Default: unit-circle ellipse with uncertainty
            extent_init = np.eye(2) * (dof_init - _DOF_MIN)
        return cls(m=m, P=P, v=dof_init, V=np.asarray(extent_init, dtype=float),
                   alpha=alpha, beta=beta)
