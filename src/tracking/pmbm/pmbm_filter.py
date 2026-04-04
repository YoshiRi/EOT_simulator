"""GGIW-PMBM multi-object tracking filter.

Orchestrates the Poisson Multi-Bernoulli Mixture (PMBM) filter using
GGIW single-target distributions.

Architecture
------------
The filter state at each time step consists of:

  PPP  — Poisson Point Process (undetected / birth targets)
         Represented as a finite GGIW mixture; properly propagated and
         updated each frame.

  MBM  — Multi-Bernoulli Mixture
         A weighted list of GlobalHypothesis objects, each holding an
         ordered list of Bernoulli components.

Pipeline per frame
------------------
1. predict
     a. Advance all MBM tracks via CV model; scale r by p_survival.
     b. Propagate surviving PPP components (weight × p_survival).
     c. Add birth-prior components to PPP at measurement-cell positions
        (adaptive birth model; weight = BirthModel.birth_weight).

2. update
     a. For each old hypothesis, enumerate valid cell→track assignments;
        birth cost and new-Bernoulli state come from the PPP.
     b. Prune + cap the resulting hypothesis set.
     c. Call ppp.undetected_update(p_D) to down-weight PPP components
        that survived without detection.

3. extract
     Return the MAP hypothesis tracks with r > threshold.

Notes
-----
* ``BirthModel`` is now a parameter holder for the birth *prior* (weight
  per new component, initial GGIW hyperparameters).  The actual birth
  Bernoullis and their log-likelihood contributions are computed by the
  PPP at update time.
* Tracks with r < ``prune_track_threshold`` are removed from all
  hypotheses after each update to limit track-list growth.
* ``BirthModel.birth_log_rate`` remains the *fallback* log-rate used
  when all PPP likelihoods evaluate to zero (e.g., very first frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.tracking.ggiw.ggiw_state import GGIWState
from src.tracking.measurement.clustering import MeasurementCell
from src.tracking.measurement.partition import partition
from .bernoulli import Bernoulli
from .hypothesis import GlobalHypothesis, cap, normalise, prune, update_hypothesis
from .ppp import PPP, PPPComponent


# ---------------------------------------------------------------------------
# Birth model (configuration only)
# ---------------------------------------------------------------------------

@dataclass
class BirthModel:
    """Configuration for the adaptive PPP birth model.

    Each unassigned measurement cell causes one new PPP component to be
    added to the filter with ``birth_weight`` as its Poisson intensity
    weight.

    Attributes
    ----------
    birth_weight : float
        Poisson weight (expected # new targets per cell per frame).
        Internally, ``log(birth_weight)`` is used as the fallback log-rate
        in the PPP likelihood computation.
    dof_init : float
        Initial IW degrees of freedom.
    extent_init : np.ndarray or None
        Initial IW scale matrix V.  Defaults to identity × (dof_init − 6).
    alpha_init : float
        Initial Gamma shape α.
    beta_init : float
        Initial Gamma rate β.
    pos_var : float
        Initial position variance.
    vel_var : float
        Initial velocity variance.
    """

    birth_weight: float = 0.01       # ≈ exp(-4.6)
    dof_init: float = 20.0
    extent_init: np.ndarray | None = None
    alpha_init: float = 5.0
    beta_init: float = 1.0
    pos_var: float = 1e2
    vel_var: float = 1e2

    # Backward-compat alias kept for tests that set birth_log_rate directly
    @property
    def birth_log_rate(self) -> float:
        return float(np.log(self.birth_weight))

    @birth_log_rate.setter
    def birth_log_rate(self, value: float) -> None:
        self.birth_weight = float(np.exp(value))

    def make_birth_components(
        self, cells: list[MeasurementCell]
    ) -> list[PPPComponent]:
        """Create one PPPComponent per cell, placed at the cell centroid.

        The GGIW state is the birth prior updated with the cell's points
        (measurement-informed initialisation for fast convergence).
        """
        from src.tracking.ggiw.ggiw_update import update as ggiw_update
        components = []
        for cell in cells:
            prior = GGIWState.from_position(
                cell.centroid,
                pos_var=self.pos_var,
                vel_var=self.vel_var,
                extent_init=self.extent_init,
                dof_init=self.dof_init,
                alpha=self.alpha_init,
                beta=self.beta_init,
            )
            state = ggiw_update(prior, cell.points)
            components.append(PPPComponent(weight=self.birth_weight, state=state))
        return components


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class PMBMFilter:
    """GGIW-PMBM multi-object filter with proper PPP component.

    Attributes
    ----------
    p_survival : float
        Target survival probability per frame.
    p_detection : float
        Probability of detecting a target given existence.
    birth_model : BirthModel
        Configuration for the adaptive birth PPP.
    max_hypotheses : int
        Hard cap on the MBM hypothesis count.
    prune_log_threshold : float
        Prune hypotheses whose log-weight is this many nats below the
        maximum (e.g. −15 ≈ e⁻¹⁵).
    prune_track_threshold : float
        Remove Bernoulli components with r below this value from all
        hypotheses after each update.
    ggiw_kwargs : dict
        Extra keyword arguments forwarded to ggiw_predict
        (sigma_q, eta_x, eta_gamma).
    """

    p_survival: float = 0.99
    p_detection: float = 0.9
    birth_model: BirthModel = field(default_factory=BirthModel)
    max_hypotheses: int = 100
    prune_log_threshold: float = -15.0
    prune_track_threshold: float = 1e-3
    ggiw_kwargs: dict = field(default_factory=dict)

    # MBM state
    _hypotheses: list[GlobalHypothesis] = field(
        default_factory=lambda: [GlobalHypothesis(log_weight=0.0, tracks=[])]
    )

    # PPP state — empty at startup; populated by first predict/update cycle
    _ppp: PPP = field(default_factory=PPP)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Predict MBM and PPP forward by ``dt`` seconds."""
        # MBM: advance all tracks
        self._hypotheses = [
            h.predict(dt, self.p_survival, **self.ggiw_kwargs)
            for h in self._hypotheses
        ]
        # PPP: propagate surviving undetected components
        self._ppp.predict(dt, self.p_survival, **self.ggiw_kwargs)

    def update(self, cells: list[MeasurementCell]) -> None:
        """Update the filter with a list of measurement cells.

        Steps:
          1. Add birth components from current cells to the PPP.
          2. Enumerate hypotheses (PPP provides birth costs/posteriors).
          3. Prune, cap, and normalise.
          4. Down-weight undetected PPP components.
          5. Drop weak tracks from all hypotheses.
        """
        # Step 1: Adaptive birth — add one PPP component per cell
        birth_components = self.birth_model.make_birth_components(cells)
        self._ppp.add_birth_components(birth_components)

        # Step 2: Hypothesis enumeration (PPP is read-only here)
        new_hypotheses: list[GlobalHypothesis] = []
        for hyp in self._hypotheses:
            candidates = update_hypothesis(
                hyp,
                cells,
                self.p_detection,
                self._ppp,
                max_hypotheses=self.max_hypotheses,
                prune_log_threshold=self.prune_log_threshold,
            )
            new_hypotheses.extend(candidates)

        # Step 3: Manage hypothesis set
        new_hypotheses = prune(new_hypotheses, self.prune_log_threshold)
        new_hypotheses = cap(new_hypotheses, self.max_hypotheses)
        new_hypotheses = normalise(new_hypotheses)

        # Remove near-zero-existence tracks from all hypotheses
        self._hypotheses = [self._drop_weak_tracks(h) for h in new_hypotheses]

        # Step 4: Down-weight undetected PPP components
        self._ppp.undetected_update(self.p_detection)
        self._ppp.prune()

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

    @property
    def ppp(self) -> PPP:
        """The current PPP state (read access for diagnostics)."""
        return self._ppp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drop_weak_tracks(self, hyp: GlobalHypothesis) -> GlobalHypothesis:
        """Remove tracks with r < prune_track_threshold."""
        kept = [b for b in hyp.tracks if b.r >= self.prune_track_threshold]
        return GlobalHypothesis(log_weight=hyp.log_weight, tracks=kept)
