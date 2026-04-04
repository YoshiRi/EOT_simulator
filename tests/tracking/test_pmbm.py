"""Tests for PMBM filter components."""

import numpy as np
import pytest

from src.tracking.ggiw.ggiw_state import GGIWState, _DOF_MIN
from src.tracking.measurement.clustering import MeasurementCell
from src.tracking.pmbm.bernoulli import Bernoulli
from src.tracking.pmbm.hypothesis import (
    GlobalHypothesis,
    cap,
    normalise,
    prune,
    update_hypothesis,
)
from src.tracking.pmbm.pmbm_filter import BirthModel, PMBMFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(pos=(0.0, 0.0), v=20.0) -> GGIWState:
    return GGIWState(
        m=np.array([pos[0], pos[1], 0.0, 0.0]),
        P=np.diag([1.0, 1.0, 4.0, 4.0]),
        v=v,
        V=np.eye(2) * (v - _DOF_MIN),
        alpha=5.0,
        beta=1.0,
    )


def _make_cell(center=(0.0, 0.0), n=4, spread=0.2, seed=0) -> MeasurementCell:
    rng = np.random.default_rng(seed)
    pts = np.array(center) + rng.standard_normal((n, 2)) * spread
    return MeasurementCell(pts)


# ---------------------------------------------------------------------------
# Bernoulli
# ---------------------------------------------------------------------------

class TestBernoulli:
    def test_predict_scales_r(self):
        b = Bernoulli(r=0.8, state=_make_state())
        pred = b.predict(dt=0.1, p_survival=0.95)
        assert pred.r == pytest.approx(0.8 * 0.95)

    def test_predict_returns_bernoulli(self):
        b = Bernoulli(r=0.5, state=_make_state())
        assert isinstance(b.predict(dt=0.1), Bernoulli)

    def test_detection_update_sets_r_to_one(self):
        b = Bernoulli(r=0.6, state=_make_state())
        cell = _make_cell()
        b_new, _ = b.detection_update(cell, p_detection=0.9)
        assert b_new.r == pytest.approx(1.0)

    def test_detection_update_log_weight_finite(self):
        b = Bernoulli(r=0.8, state=_make_state())
        cell = _make_cell()
        _, lw = b.detection_update(cell, p_detection=0.9)
        assert np.isfinite(lw)

    def test_missed_update_decreases_r(self):
        b = Bernoulli(r=0.8, state=_make_state())
        b_new, _ = b.missed_update(p_detection=0.9)
        assert b_new.r < b.r

    def test_missed_update_log_weight(self):
        # log(1 - r*pD) = log(1 - 0.8*0.9) = log(0.28)
        b = Bernoulli(r=0.8, state=_make_state())
        _, lw = b.missed_update(p_detection=0.9)
        assert lw == pytest.approx(np.log(1.0 - 0.8 * 0.9), rel=1e-6)

    def test_missed_update_preserves_state(self):
        b = Bernoulli(r=0.5, state=_make_state())
        b_new, _ = b.missed_update(p_detection=0.9)
        assert np.allclose(b_new.state.m, b.state.m)

    def test_r_zero_target_stays_zero(self):
        b = Bernoulli(r=0.0, state=_make_state())
        b_new, _ = b.missed_update(p_detection=0.9)
        assert b_new.r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GlobalHypothesis
# ---------------------------------------------------------------------------

class TestGlobalHypothesis:
    def test_predict_propagates_tracks(self):
        tracks = [Bernoulli(r=0.9, state=_make_state((i * 10.0, 0.0))) for i in range(3)]
        hyp = GlobalHypothesis(log_weight=0.0, tracks=tracks)
        pred = hyp.predict(dt=0.5, p_survival=0.99)
        assert len(pred.tracks) == 3
        for orig, new in zip(tracks, pred.tracks):
            assert new.r == pytest.approx(0.99 * orig.r)

    def test_update_empty_cells_returns_all_missed(self):
        tracks = [Bernoulli(r=0.9, state=_make_state())]
        hyp = GlobalHypothesis(log_weight=0.0, tracks=tracks)
        birth = BirthModel()
        results = update_hypothesis(hyp, [], 0.9, birth.new_bernoulli)
        assert len(results) == 1
        assert results[0].tracks[0].r < 0.9  # missed → r decreased

    def test_update_single_track_single_cell(self):
        track = Bernoulli(r=0.8, state=_make_state())
        hyp = GlobalHypothesis(log_weight=0.0, tracks=[track])
        cell = _make_cell(center=(0.0, 0.0))
        birth = BirthModel()
        results = update_hypothesis(hyp, [cell], 0.9, birth.new_bernoulli)
        assert len(results) >= 2  # at least: (track+cell) and (track missed + birth)

    def test_update_no_tracks_creates_birth(self):
        hyp = GlobalHypothesis(log_weight=0.0, tracks=[])
        cell = _make_cell()
        birth = BirthModel(r_birth=0.1)
        results = update_hypothesis(hyp, [cell], 0.9, birth.new_bernoulli)
        # Only option: create a new birth track
        assert any(len(h.tracks) == 1 for h in results)

    def test_update_respects_max_hypotheses(self):
        tracks = [Bernoulli(r=0.8, state=_make_state((float(i), 0.0))) for i in range(3)]
        hyp = GlobalHypothesis(log_weight=0.0, tracks=tracks)
        cells = [_make_cell((float(j), 0.0), seed=j) for j in range(3)]
        birth = BirthModel()
        results = update_hypothesis(hyp, cells, 0.9, birth.new_bernoulli, max_hypotheses=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# Hypothesis management
# ---------------------------------------------------------------------------

class TestHypothesisManagement:
    def _hyps(self, weights):
        return [GlobalHypothesis(log_weight=w, tracks=[]) for w in weights]

    def test_prune_removes_low_weight(self):
        hyps = self._hyps([0.0, -5.0, -20.0])
        kept = prune(hyps, log_threshold=-10.0)
        assert all(h.log_weight >= -10.0 for h in kept)
        assert len(kept) == 2

    def test_prune_keeps_at_least_one(self):
        hyps = self._hyps([-100.0, -200.0])
        kept = prune(hyps, log_threshold=-10.0)
        assert len(kept) >= 1

    def test_cap_keeps_top_n(self):
        hyps = self._hyps([0.0, -1.0, -2.0, -3.0, -4.0])
        capped = cap(hyps, max_count=3)
        assert len(capped) == 3
        assert capped[0].log_weight == pytest.approx(0.0)

    def test_normalise_weights_sum_to_one(self):
        hyps = self._hyps([0.0, -1.0, -2.0])
        normed = normalise(hyps)
        total = sum(np.exp(h.log_weight) for h in normed)
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_normalise_preserves_order(self):
        hyps = self._hyps([0.0, -1.0, -2.0])
        normed = normalise(hyps)
        for orig, new in zip(hyps, normed):
            assert new.log_weight <= orig.log_weight  # all shifted down


# ---------------------------------------------------------------------------
# PMBMFilter integration
# ---------------------------------------------------------------------------

class TestPMBMFilter:
    def test_initial_state_has_one_empty_hypothesis(self):
        f = PMBMFilter()
        assert f.num_hypotheses == 1
        assert f.extract_estimates() == []

    def test_predict_does_not_crash_on_empty_filter(self):
        f = PMBMFilter()
        f.predict(dt=0.1)

    def test_update_with_one_cell_creates_birth_track(self):
        f = PMBMFilter(birth_model=BirthModel(r_birth=0.5))
        cell = _make_cell(center=(5.0, 5.0))
        f.update([cell])
        estimates = f.extract_estimates(existence_threshold=0.4)
        assert len(estimates) >= 1

    def test_single_target_tracking_scenario(self):
        """A target at (10, 10) moving with vx=1 is consistently observed.

        After enough consistent detections the detection hypothesis should
        overtake the all-births hypothesis and the MAP estimate should
        converge near the true position.
        """
        rng = np.random.default_rng(42)
        f = PMBMFilter(
            p_survival=0.99,
            p_detection=0.9,
            birth_model=BirthModel(r_birth=0.3, birth_log_rate=-4.6),
            max_hypotheses=50,
            prune_log_threshold=-30.0,
        )
        pos = np.array([10.0, 10.0])
        for step in range(15):
            pos[0] += 1.0  # move right
            pts = pos + rng.standard_normal((5, 2)) * 0.3
            f.predict(dt=1.0)
            f.update_from_points(pts, eps=1.0)

        estimates = f.extract_estimates(existence_threshold=0.5)
        assert len(estimates) >= 1
        # The estimated position should be close to the true position
        est_pos = estimates[0].position
        assert np.linalg.norm(est_pos - pos) < 5.0

    def test_num_hypotheses_bounded(self):
        rng = np.random.default_rng(0)
        f = PMBMFilter(max_hypotheses=20)
        for _ in range(5):
            pts = rng.standard_normal((8, 2)) * 3.0
            f.predict(dt=0.1)
            f.update_from_points(pts, eps=1.5)
        assert f.num_hypotheses <= 20

    def test_missed_frames_reduce_existence(self):
        """After several frames with no measurements, r should drop."""
        f = PMBMFilter(p_survival=0.99, p_detection=0.9,
                       birth_model=BirthModel(r_birth=0.5))
        # Seed a track
        cell = _make_cell(center=(5.0, 5.0))
        f.update([cell])
        r_initial = max(
            (b.r for h in f._hypotheses for b in h.tracks), default=0.0
        )
        # 10 missed frames
        for _ in range(10):
            f.predict(dt=0.1)
            f.update([])
        r_final = max(
            (b.r for h in f._hypotheses for b in h.tracks), default=0.0
        )
        assert r_final < r_initial
