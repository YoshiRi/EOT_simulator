"""Poisson Point Process (PPP) component for GGIW-PMBM.

The PPP models the spatial density of *undetected* targets.  Its intensity
function is represented as a finite mixture of weighted GGIW components:

    D(x) = Σ_k  w_k · GGIW(x; s_k)

where each ``PPPComponent(weight=w_k, state=s_k)`` carries a non-negative
Poisson weight (expected number of targets in that mode) and a ``GGIWState``.

Lifecycle in one PMBM cycle
---------------------------
1. **predict**       – each component is propagated by ggiw_predict; weight
                       is scaled by p_survival.
2. **add_birth**     – new components are appended for this frame's birth prior
                       (typically one per measurement cell, with weight λ_birth).
3. **update (read)** – ``detection_log_likelihood`` and ``detection_posterior``
                       are called (read-only) inside the hypothesis enumeration
                       loop to compute per-cell birth costs and new Bernoullis.
4. **undetected_update** – after all hypotheses are updated, all surviving
                       PPP weights are scaled by (1 − p_D).
5. **prune**         – components below weight_threshold are discarded.

References
----------
Granström, K. & Baum, M. (2022).  A Tutorial on Multiple Extended Object
Tracking.  TechRxiv.  https://doi.org/10.36227/techrxiv.19115858.v1

Williams, J. L. (2015).  Marginal multi-Bernoulli filters: RFS derivation of
MHT, JIPDA, and association-based MeMBer.  IEEE TAES.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.tracking.ggiw.ggiw_state import GGIWState
from src.tracking.ggiw.ggiw_update import (
    likelihood as ggiw_likelihood,
    predict as ggiw_predict,
    update as ggiw_update,
)
from src.tracking.measurement.clustering import MeasurementCell
from .bernoulli import Bernoulli


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PPPComponent:
    """One GGIW component of the PPP intensity mixture.

    Attributes
    ----------
    weight : float
        Poisson intensity weight w_k ≥ 0.  Represents the expected number
        of undetected targets with this kinematic/extent/rate state.
    state : GGIWState
        GGIW parameters for this component.
    """

    weight: float
    state: GGIWState


# ---------------------------------------------------------------------------
# PPP
# ---------------------------------------------------------------------------

@dataclass
class PPP:
    """Finite-mixture PPP intensity for undetected extended targets.

    Parameters
    ----------
    components : list[PPPComponent]
        Initial mixture.  Usually empty at filter startup.
    weight_threshold : float
        Components with weight below this value are pruned (default 1e-5).
    """

    components: list[PPPComponent] = field(default_factory=list)
    weight_threshold: float = 1e-5

    # ------------------------------------------------------------------
    # Lifecycle operations (mutating)
    # ------------------------------------------------------------------

    def predict(
        self,
        dt: float,
        p_survival: float = 0.99,
        **ggiw_kwargs,
    ) -> None:
        """In-place: propagate surviving undetected targets.

        Scales each weight by ``p_survival`` and advances each state
        via the constant-velocity GGIW predict step.  Components whose
        weight drops below ``weight_threshold`` are removed.
        """
        kept: list[PPPComponent] = []
        for c in self.components:
            w = p_survival * c.weight
            if w >= self.weight_threshold:
                kept.append(PPPComponent(
                    weight=w,
                    state=ggiw_predict(c.state, dt, **ggiw_kwargs),
                ))
        self.components = kept

    def add_birth_components(self, components: list[PPPComponent]) -> None:
        """Append new birth-prior components to the mixture.

        Called once per frame *before* the hypothesis update loop, with
        components generated from the current measurement cells (adaptive
        birth model).
        """
        self.components.extend(components)

    def undetected_update(self, p_detection: float) -> None:
        """In-place: down-weight surviving undetected components.

        Called *once* after the full hypothesis enumeration loop.  Each
        component that was not associated with a measurement retains
        weight  w_k · (1 − p_D).
        """
        surviving: list[PPPComponent] = []
        for c in self.components:
            w = (1.0 - p_detection) * c.weight
            if w >= self.weight_threshold:
                surviving.append(PPPComponent(weight=w, state=c.state))
        self.components = surviving

    def prune(self) -> None:
        """Remove components with weight < weight_threshold."""
        self.components = [
            c for c in self.components if c.weight >= self.weight_threshold
        ]

    # ------------------------------------------------------------------
    # Read-only query methods (called inside hypothesis enumeration)
    # ------------------------------------------------------------------

    def detection_log_likelihood(
        self,
        cell: MeasurementCell,
        p_detection: float,
        fallback_log_rate: float = -4.6,
    ) -> float:
        """Log detection intensity at *cell*:  log Σ_k w_k · p_D · L_k(cell).

        This replaces the flat ``birth_log_rate`` scalar used in the
        simplified filter.  When the PPP mixture is empty (e.g., at the
        very start before any birth components are added), falls back to
        ``fallback_log_rate``.

        The return value is used as the log-weight increment for a birth
        assignment in the global hypothesis enumeration.

        Parameters
        ----------
        cell : MeasurementCell
            Measurement cell to evaluate at.
        p_detection : float
            Probability of detection.
        fallback_log_rate : float
            Log-rate used when all component likelihoods are zero or the
            mixture is empty.  Should match the prior belief about new
            target density (default log(0.01) = −4.6).
        """
        if not self.components:
            return fallback_log_rate

        total = 0.0
        for c in self.components:
            L = ggiw_likelihood(c.state, cell.points)
            total += c.weight * p_detection * L

        return float(np.log(total)) if total > 0.0 else fallback_log_rate

    def detection_posterior(
        self,
        cell: MeasurementCell,
        p_detection: float,
        fallback_log_rate: float = -4.6,
    ) -> Bernoulli:
        """New Bernoulli track formed by associating *cell* with the PPP.

        Computes the posterior existence probability r_new and a
        moment-matched GGIW state from the weighted mixture of updated
        components.

        The existence probability uses the PMBM formula:

            r_new = Λ / (κ + Λ)

        where ``Λ = Σ_k w_k · p_D · L_k(cell)`` is the detection
        intensity and ``κ = exp(fallback_log_rate)`` acts as the clutter
        or residual birth density.

        Parameters
        ----------
        cell : MeasurementCell
            Measurement cell triggering the birth.
        p_detection : float
            Probability of detection.
        fallback_log_rate : float
            Log-rate for the clutter/prior term and empty-PPP fallback.
        """
        clutter = float(np.exp(fallback_log_rate))

        if not self.components:
            state = self._default_state(cell)
            r = clutter / (clutter + clutter)  # 0.5 when no info
            return Bernoulli(r=float(np.clip(r, 1e-6, 1.0 - 1e-6)), state=state)

        # Per-component detection likelihoods
        raw = np.array([
            c.weight * p_detection * ggiw_likelihood(c.state, cell.points)
            for c in self.components
        ])
        total = float(raw.sum())

        # Existence probability
        r_new = total / (clutter + total) if (clutter + total) > 0.0 else 0.1
        r_new = float(np.clip(r_new, 1e-6, 1.0 - 1e-6))

        if total <= 0.0:
            return Bernoulli(r=r_new, state=self._default_state(cell))

        norm = raw / total
        updated = [ggiw_update(c.state, cell.points) for c in self.components]

        # Kinematic moment matching
        m_bar = sum(float(w) * s.m for w, s in zip(norm, updated))
        P_bar = sum(
            float(w) * (s.P + np.outer(s.m - m_bar, s.m - m_bar))
            for w, s in zip(norm, updated)
        )
        P_bar = 0.5 * (P_bar + P_bar.T)

        # IW and Gamma: use dominant component
        dom = int(np.argmax(norm))
        s_dom = updated[dom]

        merged = GGIWState(
            m=m_bar,
            P=P_bar,
            v=s_dom.v,
            V=s_dom.V,
            alpha=s_dom.alpha,
            beta=s_dom.beta,
        )
        return Bernoulli(r=r_new, state=merged)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_state(self, cell: MeasurementCell) -> GGIWState:
        """Fallback: birth prior updated with cell measurements."""
        prior = GGIWState.from_position(cell.centroid, pos_var=1e2, vel_var=1e2)
        return ggiw_update(prior, cell.points)
