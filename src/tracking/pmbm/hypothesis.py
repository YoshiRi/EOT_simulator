"""Global hypothesis management for PMBM.

A GlobalHypothesis is one element of the Multi-Bernoulli Mixture (MBM).
It carries a log-weight and an ordered list of Bernoulli track components.

Update strategy
---------------
Given N existing tracks and M measurement cells, the valid assignments are
all injective maps  σ: cells → tracks ∪ {birth}  (each track used ≤ once).
Unassigned tracks receive a missed-detection update.

For small scenes (N ≤ _ENUM_LIMIT and M ≤ _ENUM_LIMIT) all valid
assignments are enumerated explicitly. For larger scenes, only the MAP
(minimum-cost) assignment from scipy's linear_sum_assignment is used,
which degrades the filter to a single-hypothesis approximation but keeps
it computationally tractable.

After enumeration, hypotheses are pruned by a log-weight threshold and
capped at a maximum count to prevent combinatorial explosion.

PPP interface
-------------
The ``update_hypothesis`` function now accepts a ``PPP`` object (see
``ppp.py``) instead of a ``birth_fn / birth_log_rate`` pair.  The PPP
provides:

* ``detection_log_likelihood(cell, p_d)`` – replaces the flat
  ``birth_log_rate`` constant; computed from the full PPP mixture.
* ``detection_posterior(cell, p_d)``      – replaces ``birth_fn(cell)``;
  returns a Bernoulli with r and state derived from the PPP posterior.

Both methods are read-only – the PPP state is mutated later by
``ppp.undetected_update()``, after the enumeration loop completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bernoulli import Bernoulli
from src.tracking.measurement.clustering import MeasurementCell

if TYPE_CHECKING:
    from .ppp import PPP

# Max N or M before falling back to MAP-only
_ENUM_LIMIT = 7


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class GlobalHypothesis:
    """One element of the MBM: a joint assignment hypothesis.

    Attributes:
        log_weight: Unnormalised log-weight of this hypothesis.
        tracks:     Ordered list of Bernoulli components.
    """

    log_weight: float
    tracks: list[Bernoulli] = field(default_factory=list)

    def predict(self, dt: float, p_survival: float, **ggiw_kwargs) -> GlobalHypothesis:
        """Propagate all tracks forward by one time step."""
        return GlobalHypothesis(
            log_weight=self.log_weight,
            tracks=[t.predict(dt, p_survival, **ggiw_kwargs) for t in self.tracks],
        )


# ---------------------------------------------------------------------------
# Update: generate new hypotheses from one old hypothesis
# ---------------------------------------------------------------------------

def update_hypothesis(
    hyp: GlobalHypothesis,
    cells: list[MeasurementCell],
    p_detection: float,
    ppp: "PPP",
    max_hypotheses: int = 100,
    prune_log_threshold: float = -15.0,
) -> list[GlobalHypothesis]:
    """Generate all valid posterior hypotheses from a predicted hypothesis.

    Args:
        hyp:                 Predicted GlobalHypothesis.
        cells:               Measurement cells from this time step.
        p_detection:         Probability of detecting a target given it exists.
        ppp:                 PPP intensity (read-only during this call).
                             Provides per-cell birth likelihoods and posteriors.
        max_hypotheses:      Hard cap on the number of hypotheses returned.
        prune_log_threshold: Discard hypotheses whose log-weight is below
                             this value (relative to the current hypothesis).

    Returns:
        List of new GlobalHypothesis objects (un-normalised weights).
    """
    N = len(hyp.tracks)
    M = len(cells)

    if M == 0:
        return [_all_missed(hyp, p_detection)]

    if N <= _ENUM_LIMIT and M <= _ENUM_LIMIT:
        return _enumerate(hyp, cells, p_detection, ppp,
                          max_hypotheses, prune_log_threshold)
    return _map_only(hyp, cells, p_detection, ppp)


# ---------------------------------------------------------------------------
# Enumeration (small N, M)
# ---------------------------------------------------------------------------

def _enumerate(
    hyp: GlobalHypothesis,
    cells: list[MeasurementCell],
    p_detection: float,
    ppp: "PPP",
    max_hypotheses: int,
    prune_log_threshold: float,
) -> list[GlobalHypothesis]:
    N = len(hyp.tracks)
    M = len(cells)

    # Pre-compute all per-track / per-cell local hypothesis costs
    det_bern = [[None] * M for _ in range(N)]   # updated Bernoulli
    det_lw   = [[0.0]  * M for _ in range(N)]   # log-weight contribution
    for i, track in enumerate(hyp.tracks):
        for j, cell in enumerate(cells):
            b, lw = track.detection_update(cell, p_detection)
            det_bern[i][j] = b
            det_lw[i][j] = lw

    miss_bern = [None] * N
    miss_lw   = [0.0]  * N
    for i, track in enumerate(hyp.tracks):
        b, lw = track.missed_update(p_detection)
        miss_bern[i] = b
        miss_lw[i] = lw

    # PPP: per-cell birth Bernoullis and log-weight contributions
    birth_bern = [ppp.detection_posterior(cell, p_detection) for cell in cells]
    birth_lw   = [ppp.detection_log_likelihood(cell, p_detection) for cell in cells]

    # Threshold relative to base weight
    abs_threshold = hyp.log_weight + prune_log_threshold

    new_hypotheses: list[GlobalHypothesis] = []

    # assignment[j] ∈ {0..N-1}: cell j → track assignment[j]
    # assignment[j] = N:         cell j → birth (PPP)
    for assignment in iproduct(range(N + 1), repeat=M):
        # Validate: each track used at most once
        track_used = [False] * N
        valid = True
        for a in assignment:
            if a < N:
                if track_used[a]:
                    valid = False
                    break
                track_used[a] = True
        if not valid:
            continue

        # Accumulate log-weight
        log_w = hyp.log_weight
        for j, a in enumerate(assignment):
            if a < N:
                log_w += det_lw[a][j]
            else:  # birth — PPP detection intensity at this cell
                log_w += birth_lw[j]
        for i in range(N):
            if not track_used[i]:
                log_w += miss_lw[i]

        if log_w < abs_threshold:
            continue

        # Construct updated track list
        assigned_to: dict[int, int] = {a: j for j, a in enumerate(assignment) if a < N}
        tracks_new: list[Bernoulli] = []
        for i in range(N):
            if i in assigned_to:
                tracks_new.append(det_bern[i][assigned_to[i]])
            else:
                tracks_new.append(miss_bern[i])
        for j, a in enumerate(assignment):
            if a == N:
                tracks_new.append(birth_bern[j])

        new_hypotheses.append(GlobalHypothesis(log_weight=log_w, tracks=tracks_new))

        if len(new_hypotheses) >= max_hypotheses:
            break

    # Guarantee at least one hypothesis (all-missed fallback)
    if not new_hypotheses:
        new_hypotheses.append(_all_missed(hyp, p_detection))

    return new_hypotheses


def _all_missed(hyp: GlobalHypothesis, p_detection: float) -> GlobalHypothesis:
    """Apply missed-detection update to every track."""
    tracks_new = []
    log_w = hyp.log_weight
    for track in hyp.tracks:
        b, lw = track.missed_update(p_detection)
        tracks_new.append(b)
        log_w += lw
    return GlobalHypothesis(log_weight=log_w, tracks=tracks_new)


# ---------------------------------------------------------------------------
# MAP-only fallback (large N or M)
# ---------------------------------------------------------------------------

def _map_only(
    hyp: GlobalHypothesis,
    cells: list[MeasurementCell],
    p_detection: float,
    ppp: "PPP",
) -> list[GlobalHypothesis]:
    """Return a single MAP-assignment hypothesis via Hungarian algorithm."""
    N = len(hyp.tracks)
    M = len(cells)
    INF = 1e9

    # Cost matrix rows=tracks+birth_slots, cols=cells+missed_slots
    size = N + M
    cost = np.full((size, size), INF)

    # Detection: track i → cell j
    for i, track in enumerate(hyp.tracks):
        for j, cell in enumerate(cells):
            _, lw = track.detection_update(cell, p_detection)
            cost[i, j] = -lw

    # Missed detection: track i → dummy column
    for i, track in enumerate(hyp.tracks):
        _, lw = track.missed_update(p_detection)
        cost[i, M:M + N] = -lw

    # Birth: dummy row → cell j; cost from PPP detection log-likelihood
    for j, cell in enumerate(cells):
        birth_lw = ppp.detection_log_likelihood(cell, p_detection)
        cost[N:, j] = -birth_lw

    row_ind, col_ind = linear_sum_assignment(cost)

    tracks_new = list(hyp.tracks)
    log_w = hyp.log_weight
    assigned = [False] * N

    for r, c in zip(row_ind, col_ind):
        if r < N and c < M:            # existing track → cell
            b, lw = hyp.tracks[r].detection_update(cells[c], p_detection)
            tracks_new[r] = b
            log_w += lw
            assigned[r] = True
        elif r < N:                     # existing track → missed
            b, lw = hyp.tracks[r].missed_update(p_detection)
            tracks_new[r] = b
            log_w += lw
            assigned[r] = True
        elif c < M:                     # birth → cell (PPP posterior)
            tracks_new.append(ppp.detection_posterior(cells[c], p_detection))
            log_w += ppp.detection_log_likelihood(cells[c], p_detection)

    # Any track not yet processed
    for i in range(N):
        if not assigned[i]:
            b, lw = hyp.tracks[i].missed_update(p_detection)
            tracks_new[i] = b
            log_w += lw

    return [GlobalHypothesis(log_weight=log_w, tracks=tracks_new)]


# ---------------------------------------------------------------------------
# Hypothesis management utilities
# ---------------------------------------------------------------------------

def prune(
    hypotheses: list[GlobalHypothesis],
    log_threshold: float,
) -> list[GlobalHypothesis]:
    """Remove hypotheses whose log-weight is more than ``log_threshold``
    below the maximum. Always keeps at least one hypothesis."""
    if not hypotheses:
        return hypotheses
    max_lw = max(h.log_weight for h in hypotheses)
    kept = [h for h in hypotheses if h.log_weight >= max_lw + log_threshold]
    return kept or [max(hypotheses, key=lambda h: h.log_weight)]


def cap(hypotheses: list[GlobalHypothesis], max_count: int) -> list[GlobalHypothesis]:
    """Keep the ``max_count`` highest-weight hypotheses."""
    return sorted(hypotheses, key=lambda h: h.log_weight, reverse=True)[:max_count]


def normalise(hypotheses: list[GlobalHypothesis]) -> list[GlobalHypothesis]:
    """Normalise so that exp(log_weight) values sum to 1."""
    if not hypotheses:
        return hypotheses
    max_lw = max(h.log_weight for h in hypotheses)
    log_sum = max_lw + np.log(sum(np.exp(h.log_weight - max_lw) for h in hypotheses))
    return [
        GlobalHypothesis(log_weight=h.log_weight - log_sum, tracks=h.tracks)
        for h in hypotheses
    ]
