"""Bernoulli component for extended object tracking.

A Bernoulli component represents a single-target hypothesis with an
associated existence probability r and a GGIW state.

    p(target exists) = r
    p(state | exists) = GGIW(m, P, v, V, α, β)

References
----------
Granström, K. & Baum, M. (2022). A Tutorial on Multiple Extended
Object Tracking. TechRxiv. Section 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tracking.ggiw.ggiw_state import GGIWState
from src.tracking.ggiw.ggiw_update import (
    likelihood as ggiw_likelihood,
    predict as ggiw_predict,
    update as ggiw_update,
)
from src.tracking.measurement.clustering import MeasurementCell


@dataclass
class Bernoulli:
    """Single-target hypothesis: existence probability + GGIW state.

    Attributes:
        r:     Existence probability ∈ [0, 1].
        state: GGIW state conditioned on existence.
    """

    r: float
    state: GGIWState

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, dt: float, p_survival: float = 0.99, **ggiw_kwargs) -> Bernoulli:
        """Predict the Bernoulli forward by one time step.

        Args:
            dt:         Time step.
            p_survival: Probability that the target survives to the next
                        frame. Scales down r.
            **ggiw_kwargs: Forwarded to ggiw_predict (sigma_q, eta_x, …).

        Returns:
            Predicted Bernoulli.
        """
        return Bernoulli(
            r=p_survival * self.r,
            state=ggiw_predict(self.state, dt, **ggiw_kwargs),
        )

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------

    def detection_update(
        self, cell: MeasurementCell, p_detection: float
    ) -> tuple[Bernoulli, float]:
        """Update assuming this track was detected by ``cell``.

        The updated Bernoulli has r=1 (existence is certain once
        detected). The log-weight contribution is used by the parent
        GlobalHypothesis to compute the posterior hypothesis weight.

        Args:
            cell:        Measurement cell assigned to this track.
            p_detection: Probability of detection given existence.

        Returns:
            (updated_bernoulli, log_weight_contribution)
        """
        l = ggiw_likelihood(self.state, cell.points)
        log_w = np.log(max(self.r * p_detection * l, 1e-300))
        return Bernoulli(r=1.0, state=ggiw_update(self.state, cell.points)), log_w

    def missed_update(self, p_detection: float) -> tuple[Bernoulli, float]:
        """Update for missed detection (no cell assigned to this track).

        Args:
            p_detection: Probability of detection given existence.

        Returns:
            (updated_bernoulli, log_weight_contribution)
            The log-weight contribution is log(1 - r·p_D).
        """
        r_pd = self.r * p_detection
        log_w = np.log(max(1.0 - r_pd, 1e-300))
        # Posterior existence: Bayes update for the "missed" event
        r_new = self.r * (1.0 - p_detection) / (1.0 - r_pd) if r_pd < 1.0 - 1e-10 else 0.0
        return Bernoulli(r=r_new, state=self.state), log_w
