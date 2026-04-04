"""GGIW prediction and measurement update.

Implements the approximate conjugate GGIW filter equations for a single
extended target tracked with a Poisson measurement model.

The update derivation follows Granström & Orguner (2012), Theorem 2,
extended with the Gamma-Poisson mixture for the measurement count.

Notation
--------
d  = 2   (spatial dimension)
n  = number of measurements in a cell
ẑ  = centroid of measurements
Zε = scatter matrix  Σ (z_i - ẑ)(z_i - ẑ)^T
X̂  = V / (v - 2d - 2)  prior mean of extent (IW mean)
S  = H P H^T + X̂/n     innovation covariance of the centroid

References
----------
Granström, K. & Orguner, U. (2012). A PHD filter for tracking
multiple extended targets. IEEE Trans. Signal Processing.

Granström, K. & Baum, M. (2022). A Tutorial on Multiple Extended
Object Tracking. TechRxiv. https://doi.org/10.36227/techrxiv.19115858.v1
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from scipy.special import gammaln

from .ggiw_state import GGIWState, _D, _DOF_MIN

# Measurement matrix H: extracts position [px, py] from [px, py, vx, vy]
H = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Constant-velocity motion model helpers
# ---------------------------------------------------------------------------

def _cv_matrices(dt: float, sigma_q: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity state transition matrix F and process noise Q.

    State: x = [px, py, vx, vy]

    Args:
        dt:      Time step.
        sigma_q: Process noise standard deviation (acceleration noise).

    Returns:
        F: (4, 4) transition matrix.
        Q: (4, 4) process noise covariance.
    """
    F = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    q = sigma_q ** 2
    Q = q * np.array([
        [dt**4 / 4.0, 0.0,          dt**3 / 2.0, 0.0        ],
        [0.0,          dt**4 / 4.0, 0.0,          dt**3 / 2.0],
        [dt**3 / 2.0, 0.0,          dt**2,        0.0        ],
        [0.0,          dt**3 / 2.0, 0.0,          dt**2      ],
    ])
    return F, Q


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    state: GGIWState,
    dt: float,
    *,
    sigma_q: float = 1.0,
    eta_x: float = 0.9,
    eta_gamma: float = 0.9,
) -> GGIWState:
    """GGIW prediction step.

    Propagates the kinematic state via a constant-velocity model and
    allows the extent and rate distributions to "forget" old data via
    exponential forgetting factors.

    The IW mean E[X] is preserved by prediction — only the uncertainty
    (degrees of freedom) is reduced. The Gamma mean E[γ] is scaled by
    eta_gamma, reflecting survival uncertainty.

    Args:
        state:    Prior GGIWState at time k-1.
        dt:       Time step Δt.
        sigma_q:  Process noise std dev for the CV model.
        eta_x:    Extent forgetting factor ∈ (0, 1). Smaller → faster
                  spreading of extent uncertainty.
        eta_gamma: Rate forgetting factor ∈ (0, 1). Smaller → more
                   uncertainty in measurement rate.

    Returns:
        Predicted GGIWState at time k.
    """
    F, Q = _cv_matrices(dt, sigma_q)

    # --- Kinematic prediction (standard Kalman) ---
    m_pred = F @ state.m
    P_pred = F @ state.P @ F.T + Q

    # --- Extent prediction (IW forgetting) ---
    # Shrink excess DOF toward the minimum valid value.
    # This preserves E[X] = V/(v-6) while reducing certainty.
    dof_excess = state.v - _DOF_MIN
    v_pred = eta_x * dof_excess + _DOF_MIN
    # Clamp strictly above minimum to keep IW mean well-defined
    v_pred = max(v_pred, _DOF_MIN + 1e-6)
    scale = (v_pred - _DOF_MIN) / dof_excess
    V_pred = scale * state.V

    # --- Rate prediction (Gamma forgetting) ---
    # Scale both parameters equally so mean = α_pred/β_pred = η_γ * (α/β)
    alpha_pred = eta_gamma * state.alpha
    beta_pred = eta_gamma * state.beta

    return GGIWState(
        m=m_pred, P=P_pred,
        v=v_pred, V=V_pred,
        alpha=alpha_pred, beta=beta_pred,
    )


# ---------------------------------------------------------------------------
# Measurement update
# ---------------------------------------------------------------------------

def update(state: GGIWState, measurements: np.ndarray) -> GGIWState:
    """GGIW measurement update with a cell of n point measurements.

    Given a measurement cell Z = {z_1, ..., z_n} (n ≥ 1) assigned to
    this target, computes the approximate conjugate posterior.

    The update is derived under the approximate Gaussian / IW / Gamma
    assumptions from Granström & Orguner (2012), Theorem 2:

    - Kinematic:  Kalman update using the cell centroid ẑ.
    - Extent:     IW update using scatter Zε and innovation spread N.
    - Rate:       Gamma update: α += n, β += 1.

    The extent innovation term N maps the kinematic innovation ε back
    into extent space via the prior mean X̂:

        N = sqrtm(X̂) S⁻¹ εεᵀ S⁻¹ sqrtm(X̂)

    Args:
        state:        Predicted GGIWState.
        measurements: Array of shape (n, 2), at least one measurement.

    Returns:
        Updated GGIWState.

    Raises:
        ValueError: If measurements array is empty.
    """
    Z = np.atleast_2d(measurements).astype(float)
    n = len(Z)
    if n == 0:
        raise ValueError("measurements must contain at least one point; "
                         "use predict() alone for missed detection.")

    # --- Cell sufficient statistics ---
    z_bar = Z.mean(axis=0)                        # centroid (2,)
    Z_eps = (Z - z_bar).T @ (Z - z_bar)           # scatter  (2, 2)

    # --- Prior extent mean ---
    X_hat = state.extent_mean                      # (2, 2)

    # --- Innovation covariance of centroid ẑ ---
    # Cov(ẑ) = (1/n) Cov(z_i) ≈ (1/n)(H P H^T + X̂)
    # but H P H^T is the uncertainty from kinematics alone,
    # and X̂/n is from the distributed measurements.
    inn = z_bar - H @ state.m                      # (2,)
    S = H @ state.P @ H.T + X_hat / n             # (2, 2)
    S_inv = np.linalg.inv(S)

    # --- Kalman gain and kinematic update ---
    K = state.P @ H.T @ S_inv                      # (4, 2)
    m_new = state.m + K @ inn
    P_new = state.P - K @ S @ K.T
    # Symmetrise to suppress floating-point drift
    P_new = 0.5 * (P_new + P_new.T)

    # --- Rate update (Gamma conjugate) ---
    alpha_new = state.alpha + n
    beta_new = state.beta + 1.0

    # --- Extent update (IW approximate conjugate) ---
    # Innovation spread mapped into extent coordinates:
    #   N = sqrtm(X̂) S⁻¹ εεᵀ S⁻¹ sqrtm(X̂)
    X_half = np.real(sqrtm(X_hat))                # (2, 2) real SPD
    N = X_half @ S_inv @ np.outer(inn, inn) @ S_inv @ X_half

    v_new = state.v + n
    V_new = state.V + Z_eps + N
    # Symmetrise
    V_new = 0.5 * (V_new + V_new.T)

    return GGIWState(
        m=m_new, P=P_new,
        v=v_new, V=V_new,
        alpha=alpha_new, beta=beta_new,
    )


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def likelihood(state: GGIWState, measurements: np.ndarray) -> float:
    """Approximate predictive likelihood L(Z | predicted state).

    Used by the PMBM filter to compute data association weights.

    The likelihood factorises as:

        L(Z) = L_count(n) · L_spatial(ẑ)

    where:
    - L_count  is the Gamma-Poisson (Negative Binomial) marginal over γ,
      evaluating p(n | α, β) = NegBin(n; α, β/(β+1)).
    - L_spatial is a Gaussian in the centroid ẑ ~ N(H m, S) with
      S = H P H^T + X̂/n (accounts for both kinematic and extent noise).

    Args:
        state:        Predicted GGIWState.
        measurements: Array of shape (n, 2).

    Returns:
        Scalar likelihood ≥ 0.
    """
    from scipy.stats import multivariate_normal

    Z = np.atleast_2d(measurements).astype(float)
    n = len(Z)
    if n == 0:
        return 0.0

    z_bar = Z.mean(axis=0)

    # --- Count likelihood: Gamma-Poisson mixture = Negative Binomial ---
    alpha, beta = state.alpha, state.beta
    # log NegBin(n; α, p=β/(β+1))
    p = beta / (beta + 1.0)
    log_l_count = (
        gammaln(alpha + n) - gammaln(alpha) - gammaln(n + 1)
        + alpha * np.log(p)
        + n * np.log(1.0 - p)
    )
    l_count = float(np.exp(log_l_count))

    # --- Centroid spatial likelihood: N(ẑ; H m, S) ---
    X_hat = state.extent_mean
    S = H @ state.P @ H.T + X_hat / n
    l_spatial = float(multivariate_normal.pdf(z_bar, mean=H @ state.m, cov=S))

    return l_count * l_spatial
