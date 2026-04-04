"""GGIW-PMBM multi-object tracking filter.

Orchestrates the Poisson Multi-Bernoulli Mixture (PMBM) filter using
GGIW single-target distributions.

Architecture
------------
The filter state at each time step consists of:

  PPP  — Poisson Point Process (undetected / birth targets)
         Approximated by a fixed BirthModel (no explicit PPP propagation).

  MBM  — Multi-Bernoulli Mixture
         A weighted list of GlobalHypothesis objects, each holding an
         ordered list of Bernoulli components.

Pipeline per frame
------------------
1. predict   — advance all MBM tracks via CV model; scale r by p_survival.
2. update    — for each old hypothesis, enumerate valid cell→track
               assignments; prune + cap the resulting hypothesis set.
3. extract   — return the MAP hypothesis tracks with r > threshold.

Notes
-----
* Tracks with r < ``prune_track_threshold`` are silently removed from all
  hypotheses after each update to limit track-list growth.
* Cells assigned to no existing track create a new Bernoulli via
  ``BirthModel.new_bernoulli()``.  Existence probability of newborns is
  controlled by ``BirthModel.r_birth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.tracking.ggiw.ggiw_state import GGIWState
from src.tracking.measurement.clustering import MeasurementCell
from src.tracking.measurement.partition import partition
from .bernoulli import Bernoulli
from .hypothesis import GlobalHypothesis, cap, normalise, prune, update_hypothesis


# ---------------------------------------------------------------------------
# Birth model
# ---------------------------------------------------------------------------

@dataclass
class BirthModel:
    """Parameters for initialising new tracks from unassigned cells.

    Attributes:
        r_birth:      Initial existence probability for a newborn track.
        dof_init:     Initial IW degrees of freedom (> 6).
        extent_init:  Initial IW scale matrix V (shape (2, 2)).
                      Defaults to identity * (dof_init − 6) if None.
        alpha_init:   Initial Gamma shape α.
        beta_init:    Initial Gamma rate β.
        pos_var:      Initial position variance (diagonal).
        vel_var:      Initial velocity variance (diagonal).
    """

    r_birth: float = 0.1
    dof_init: float = 20.0
    extent_init: np.ndarray | None = None
    alpha_init: float = 5.0
    beta_init: float = 1.0
    pos_var: float = 1e2
    vel_var: float = 1e2
    birth_log_rate: float = -4.6  # log(0.01): expected 0.01 new targets per scan

    def new_bernoulli(self, cell: MeasurementCell) -> Bernoulli:
        """Create a birth Bernoulli initialised from the cell.

        The PPP prior at the cell centroid is updated with the cell
        measurements (proper PPP posterior), giving a tight initial
        position estimate and realistic future likelihoods.
        """
        from src.tracking.ggiw.ggiw_update import update as ggiw_update
        prior = GGIWState.from_position(
            cell.centroid,
            pos_var=self.pos_var,
            vel_var=self.vel_var,
            extent_init=self.extent_init,
            dof_init=self.dof_init,
            alpha=self.alpha_init,
            beta=self.beta_init,
        )
        return Bernoulli(r=self.r_birth, state=ggiw_update(prior, cell.points))


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class PMBMFilter:
    """GGIW-PMBM multi-object filter.

    Attributes:
        p_survival:           Target survival probability per frame.
        p_detection:          Probability of detecting a target given existence.
        birth_model:          Birth model for new tracks.
        max_hypotheses:       Hard cap on the MBM hypothesis count.
        prune_log_threshold:  Prune hypotheses whose log-weight is this many
                              nats below the maximum.  (e.g. −15 ≈ e⁻¹⁵)
        prune_track_threshold: Remove Bernoulli components with r below this
                              value from all hypotheses after each update.
        ggiw_kwargs:          Extra keyword arguments forwarded to
                              ggiw_predict (sigma_q, eta_x, eta_gamma).
    """

    p_survival: float = 0.99
    p_detection: float = 0.9
    birth_model: BirthModel = field(default_factory=BirthModel)
    max_hypotheses: int = 100
    prune_log_threshold: float = -15.0
    prune_track_threshold: float = 1e-3
    ggiw_kwargs: dict = field(default_factory=dict)

    # MBM state — one GlobalHypothesis with zero tracks at startup
    _hypotheses: list[GlobalHypothesis] = field(
        default_factory=lambda: [GlobalHypothesis(log_weight=0.0, tracks=[])]
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Predict all MBM tracks forward by ``dt`` seconds."""
        self._hypotheses = [
            h.predict(dt, self.p_survival, **self.ggiw_kwargs)
            for h in self._hypotheses
        ]

    def update(self, cells: list[MeasurementCell]) -> None:
        """Update the filter with a list of measurement cells.

        For each existing global hypothesis, generates all valid
        posterior hypotheses, then prunes and caps the result.
        """
        new_hypotheses: list[GlobalHypothesis] = []

        for hyp in self._hypotheses:
            candidates = update_hypothesis(
                hyp,
                cells,
                self.p_detection,
                self.birth_model.new_bernoulli,
                birth_log_rate=self.birth_model.birth_log_rate,
                max_hypotheses=self.max_hypotheses,
                prune_log_threshold=self.prune_log_threshold,
            )
            new_hypotheses.extend(candidates)

        # Manage hypothesis set
        new_hypotheses = prune(new_hypotheses, self.prune_log_threshold)
        new_hypotheses = cap(new_hypotheses, self.max_hypotheses)
        new_hypotheses = normalise(new_hypotheses)

        # Remove near-zero-existence tracks from all hypotheses
        self._hypotheses = [self._drop_weak_tracks(h) for h in new_hypotheses]

    def update_from_points(self, points: np.ndarray, eps: float) -> None:
        """Convenience: cluster raw points then update.

        Args:
            points: Raw sensor point cloud, shape (n, 2).
            eps:    Clustering radius in metres.
        """
        cells = partition(points, eps=eps)
        self.update(cells)

    def extract_estimates(
        self, existence_threshold: float = 0.5
    ) -> list[GGIWState]:
        """Return GGIW states from the MAP hypothesis with r > threshold.

        Selects the highest-weight global hypothesis and returns its
        confirmed tracks.

        Args:
            existence_threshold: Minimum r for a track to be reported.

        Returns:
            List of GGIWState objects, one per confirmed track.
        """
        if not self._hypotheses:
            return []
        map_hyp = max(self._hypotheses, key=lambda h: h.log_weight)
        return [
            b.state
            for b in map_hyp.tracks
            if b.r >= existence_threshold
        ]

    @property
    def num_hypotheses(self) -> int:
        """Current number of global hypotheses in the MBM."""
        return len(self._hypotheses)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drop_weak_tracks(self, hyp: GlobalHypothesis) -> GlobalHypothesis:
        """Remove tracks with r < prune_track_threshold."""
        kept = [b for b in hyp.tracks if b.r >= self.prune_track_threshold]
        return GlobalHypothesis(log_weight=hyp.log_weight, tracks=kept)
