"""Temporal smoothing of OBB parameters (theta, length, width).

Two modes are provided:

* **ExponentialSmoother** — simple 1-D exponential moving average.
  O(1) state; no tuning beyond the alpha parameter.

* **OBBKalmanSmoother** — constant-value Kalman filter on the
  [theta, length, width] vector.  Handles measurement noise and process
  noise separately, and allows one-step-ahead prediction when a frame is
  missed.

Both classes follow the same interface::

    smoother = ExponentialSmoother(alpha=0.3)
    smoothed = smoother.update(obb_result)   # returns OBBResult

Design constraint (from CLAUDE.md):
    Temporal smoothing on (θ, l, w) is *mandatory* to suppress the
    partial-observation artefact where one-sided point clouds shrink
    the apparent length / width.

Notes on angle wrapping
-----------------------
theta lives in [-π/2, π/2) with period π (not 2π) because the box has
no preferred direction along its long axis.  The smoothers handle
wrapping by working with the *sine / cosine* representation and
converting back, which avoids discontinuities near ±π/2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.tracking.shape.rectangle_fitting import OBBResult


# ---------------------------------------------------------------------------
# Helpers

def _wrap_pi(angle: float) -> float:
    """Wrap *angle* to [-π/2, π/2) (period π)."""
    # Map to [-π, π) first, then fold the remaining π range
    a = (angle + np.pi / 2) % np.pi - np.pi / 2
    return float(a)


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest difference a - b in [-π/2, π/2), wrapping at ±π/2."""
    d = a - b
    return _wrap_pi(d)


# ---------------------------------------------------------------------------
# Exponential Moving Average

@dataclass
class ExponentialSmoother:
    """1-D exponential moving average over OBB parameters.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  alpha=1 means no smoothing (raw
        measurements pass through); alpha→0 gives heavy smoothing.
    """

    alpha: float = 0.3
    _initialized: bool = field(default=False, repr=False, init=False)
    _theta: float = field(default=0.0, repr=False, init=False)
    _length: float = field(default=0.0, repr=False, init=False)
    _width: float = field(default=0.0, repr=False, init=False)
    _center: np.ndarray = field(
        default_factory=lambda: np.zeros(2), repr=False, init=False
    )

    def __post_init__(self) -> None:
        if not (0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    def update(self, obb: OBBResult) -> OBBResult:
        """Incorporate a new OBB measurement and return the smoothed estimate."""
        if not self._initialized:
            self._theta = obb.theta
            self._length = obb.length
            self._width = obb.width
            self._center = obb.center.copy()
            self._initialized = True
            return obb

        # Angle: add weighted difference to handle wrapping
        self._theta = _wrap_pi(
            self._theta + self.alpha * _angle_diff(obb.theta, self._theta)
        )
        self._length = (1 - self.alpha) * self._length + self.alpha * obb.length
        self._width = (1 - self.alpha) * self._width + self.alpha * obb.width
        self._center = (1 - self.alpha) * self._center + self.alpha * obb.center

        return OBBResult(
            theta=self._theta,
            length=self._length,
            width=self._width,
            center=self._center.copy(),
        )

    def reset(self) -> None:
        self._initialized = False


# ---------------------------------------------------------------------------
# Kalman-based smoother

# State: [theta_cos, theta_sin, length, width]
# We encode theta as (cos θ, sin θ) with period-π convention to avoid
# the wrapping singularity inside the Kalman equations.

_DIM = 4   # state dimension


@dataclass
class OBBKalmanSmoother:
    """Constant-value Kalman filter for OBB parameter smoothing.

    The state vector is [cos(θ), sin(θ), l, w].  Using the circular
    encoding avoids discontinuities in the innovation term.

    Parameters
    ----------
    process_noise : float
        Diagonal process noise standard deviation applied each frame
        (models slow shape changes / orientation drift).
    meas_noise_theta : float
        Measurement noise standard deviation for the heading angle.
    meas_noise_size : float
        Measurement noise standard deviation for length and width.
    """

    process_noise: float = 0.05
    meas_noise_theta: float = 0.1
    meas_noise_size: float = 0.3

    _x: np.ndarray = field(init=False, repr=False)
    _P: np.ndarray = field(init=False, repr=False)
    _initialized: bool = field(default=False, repr=False, init=False)
    _center: np.ndarray = field(
        default_factory=lambda: np.zeros(2), repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._x = np.zeros(_DIM)
        self._P = np.eye(_DIM)

    # --- measurement noise matrix R ---
    @property
    def _R(self) -> np.ndarray:
        sig_th = self.meas_noise_theta
        sig_sz = self.meas_noise_size
        # Measurement is [cos θ, sin θ, l, w]; noise approximated as diagonal.
        # cos/sin noise ≈ sigma_theta for small theta errors.
        return np.diag([sig_th**2, sig_th**2, sig_sz**2, sig_sz**2])

    # --- process noise matrix Q ---
    @property
    def _Q(self) -> np.ndarray:
        q = self.process_noise**2
        return np.eye(_DIM) * q

    # --- encode OBBResult into measurement vector ---
    @staticmethod
    def _encode(obb: OBBResult) -> np.ndarray:
        return np.array(
            [np.cos(obb.theta), np.sin(obb.theta), obb.length, obb.width]
        )

    # --- decode state vector back to OBBResult ---
    def _decode(self, center: np.ndarray) -> OBBResult:
        theta = float(np.arctan2(self._x[1], self._x[0]))
        theta = _wrap_pi(theta)
        length = max(float(self._x[2]), 0.1)
        width = max(float(self._x[3]), 0.1)
        if width > length:
            length, width = width, length
            theta = _wrap_pi(theta + np.pi / 2)
        return OBBResult(theta=theta, length=length, width=width, center=center)

    def update(self, obb: OBBResult) -> OBBResult:
        """Run one Kalman predict + update step and return the smoothed OBB."""
        z = self._encode(obb)
        self._center = obb.center.copy()

        if not self._initialized:
            self._x = z.copy()
            self._P = self._R.copy()
            self._initialized = True
            return self._decode(self._center)

        # Predict (constant model)
        # x_pred = x  (no dynamics)
        P_pred = self._P + self._Q

        # Update (H = I, full-state observation)
        S = P_pred + self._R
        K = P_pred @ np.linalg.inv(S)
        innovation = z - self._x
        self._x = self._x + K @ innovation
        self._P = (np.eye(_DIM) - K) @ P_pred

        # Renormalise the (cos, sin) components to stay on the unit circle
        norm = np.hypot(self._x[0], self._x[1])
        if norm > 1e-6:
            self._x[0] /= norm
            self._x[1] /= norm

        return self._decode(self._center)

    def predict(self) -> OBBResult:
        """Advance one step without a measurement (missed detection).

        The state mean is unchanged; only covariance grows.
        """
        self._P += self._Q
        return self._decode(self._center)

    def reset(self) -> None:
        self._initialized = False
        self._x = np.zeros(_DIM)
        self._P = np.eye(_DIM)
