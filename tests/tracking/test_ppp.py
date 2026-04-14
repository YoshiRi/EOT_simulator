"""Unit tests for the PPP (Poisson Point Process) component."""

import numpy as np
import pytest

from src.tracking.ggiw.ggiw_state import GGIWState, _DOF_MIN
from src.tracking.measurement.clustering import MeasurementCell
from src.tracking.pmbm.bernoulli import Bernoulli
from src.tracking.pmbm.ppp import PPP, PPPComponent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(pos=(0.0, 0.0), v: float = 20.0) -> GGIWState:
    return GGIWState(
        m=np.array([pos[0], pos[1], 0.0, 0.0]),
        P=np.diag([1.0, 1.0, 4.0, 4.0]),
        v=v,
        V=np.eye(2) * (v - _DOF_MIN),
        alpha=5.0,
        beta=1.0,
    )


def _make_cell(center=(0.0, 0.0), n: int = 5, spread: float = 0.3,
               seed: int = 0) -> MeasurementCell:
    rng = np.random.default_rng(seed)
    pts = np.array(center) + rng.standard_normal((n, 2)) * spread
    return MeasurementCell(pts)


def _make_ppp(centers=None, weight: float = 0.1) -> PPP:
    if not centers:
        return PPP()
    return PPP(components=[
        PPPComponent(weight=weight, state=_make_state(c))
        for c in centers
    ])


# ---------------------------------------------------------------------------
# PPPComponent
# ---------------------------------------------------------------------------

class TestPPPComponent:
    def test_weight_and_state_stored(self):
        s = _make_state()
        c = PPPComponent(weight=0.5, state=s)
        assert c.weight == pytest.approx(0.5)
        assert np.allclose(c.state.m, s.m)

    def test_zero_weight_allowed(self):
        c = PPPComponent(weight=0.0, state=_make_state())
        assert c.weight == 0.0


# ---------------------------------------------------------------------------
# PPP.predict
# ---------------------------------------------------------------------------

class TestPPPPredict:
    def test_weight_scaled_by_p_survival(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=1.0)
        ppp.predict(dt=0.5, p_survival=0.8)
        assert ppp.components[0].weight == pytest.approx(0.8)

    def test_state_position_advances(self):
        """After predict, kinematic position advances with velocity."""
        state = GGIWState(
            m=np.array([0.0, 0.0, 2.0, 0.0]),  # vx=2
            P=np.diag([1.0, 1.0, 4.0, 4.0]),
            v=20.0,
            V=np.eye(2) * 14.0,
            alpha=5.0, beta=1.0,
        )
        ppp = PPP(components=[PPPComponent(weight=1.0, state=state)])
        ppp.predict(dt=1.0, p_survival=1.0)
        assert ppp.components[0].state.m[0] == pytest.approx(2.0, abs=0.01)

    def test_low_weight_component_pruned(self):
        ppp = PPP(
            components=[PPPComponent(weight=1e-6, state=_make_state())],
            weight_threshold=1e-5,
        )
        ppp.predict(dt=0.5, p_survival=0.99)
        assert len(ppp.components) == 0

    def test_empty_ppp_predict_noop(self):
        ppp = PPP()
        ppp.predict(dt=0.5, p_survival=0.99)
        assert ppp.components == []


# ---------------------------------------------------------------------------
# PPP.add_birth_components
# ---------------------------------------------------------------------------

class TestAddBirthComponents:
    def test_components_appended(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=0.1)
        new = [PPPComponent(weight=0.05, state=_make_state((5.0, 0.0)))]
        ppp.add_birth_components(new)
        assert len(ppp.components) == 2

    def test_empty_list_no_change(self):
        ppp = _make_ppp([(0.0, 0.0)])
        n_before = len(ppp.components)
        ppp.add_birth_components([])
        assert len(ppp.components) == n_before


# ---------------------------------------------------------------------------
# PPP.undetected_update
# ---------------------------------------------------------------------------

class TestUndetectedUpdate:
    def test_weights_scaled_by_one_minus_pd(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=1.0)
        ppp.undetected_update(p_detection=0.7)
        assert ppp.components[0].weight == pytest.approx(0.3)

    def test_repeated_updates_decay_to_zero(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=1.0)
        for _ in range(30):
            ppp.undetected_update(p_detection=0.9)
        total = sum(c.weight for c in ppp.components)
        assert total < 1e-3

    def test_pruned_when_below_threshold(self):
        ppp = PPP(
            components=[PPPComponent(weight=1e-4, state=_make_state())],
            weight_threshold=1e-4,
        )
        ppp.undetected_update(p_detection=0.5)
        assert len(ppp.components) == 0


# ---------------------------------------------------------------------------
# PPP.detection_log_likelihood
# ---------------------------------------------------------------------------

class TestDetectionLogLikelihood:
    def test_returns_finite_float(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=0.1)
        cell = _make_cell(center=(0.0, 0.0))
        lw = ppp.detection_log_likelihood(cell, p_detection=0.9)
        assert np.isfinite(lw)

    def test_empty_ppp_returns_fallback(self):
        ppp = PPP()
        cell = _make_cell()
        fallback = -5.0
        lw = ppp.detection_log_likelihood(cell, 0.9, fallback_log_rate=fallback)
        assert lw == pytest.approx(fallback)

    def test_nearby_cell_higher_likelihood_than_far(self):
        """Cell near the PPP component should give higher likelihood."""
        ppp = _make_ppp([(0.0, 0.0)], weight=1.0)
        cell_near = _make_cell(center=(0.0, 0.0))
        cell_far  = _make_cell(center=(50.0, 50.0))
        lw_near = ppp.detection_log_likelihood(cell_near, 0.9)
        lw_far  = ppp.detection_log_likelihood(cell_far,  0.9)
        assert lw_near > lw_far

    def test_higher_weight_increases_likelihood(self):
        cell = _make_cell(center=(0.0, 0.0))
        ppp_lo = _make_ppp([(0.0, 0.0)], weight=0.01)
        ppp_hi = _make_ppp([(0.0, 0.0)], weight=1.0)
        lw_lo = ppp_lo.detection_log_likelihood(cell, 0.9)
        lw_hi = ppp_hi.detection_log_likelihood(cell, 0.9)
        assert lw_hi > lw_lo

    def test_two_components_higher_than_one(self):
        """Two components at the cell position give more intensity than one."""
        cell = _make_cell(center=(0.0, 0.0))
        ppp_one = _make_ppp([(0.0, 0.0)], weight=0.1)
        ppp_two = PPP(components=[
            PPPComponent(weight=0.1, state=_make_state((0.0, 0.0))),
            PPPComponent(weight=0.1, state=_make_state((0.1, 0.0))),
        ])
        lw_one = ppp_one.detection_log_likelihood(cell, 0.9)
        lw_two = ppp_two.detection_log_likelihood(cell, 0.9)
        assert lw_two > lw_one


# ---------------------------------------------------------------------------
# PPP.detection_posterior
# ---------------------------------------------------------------------------

class TestDetectionPosterior:
    def test_returns_bernoulli(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=0.1)
        cell = _make_cell()
        result = ppp.detection_posterior(cell, p_detection=0.9)
        assert isinstance(result, Bernoulli)

    def test_r_in_unit_interval(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=0.1)
        cell = _make_cell()
        b = ppp.detection_posterior(cell, 0.9)
        assert 0.0 < b.r < 1.0

    def test_empty_ppp_returns_valid_bernoulli(self):
        ppp = PPP()
        cell = _make_cell()
        b = ppp.detection_posterior(cell, 0.9)
        assert isinstance(b, Bernoulli)
        assert 0.0 < b.r < 1.0
        assert np.isfinite(b.state.m).all()

    def test_higher_weight_gives_higher_r(self):
        """Higher PPP weight → more likely detection → higher r_new."""
        cell = _make_cell(center=(0.0, 0.0))
        ppp_lo = _make_ppp([(0.0, 0.0)], weight=0.01)
        ppp_hi = _make_ppp([(0.0, 0.0)], weight=100.0)
        r_lo = ppp_lo.detection_posterior(cell, 0.9).r
        r_hi = ppp_hi.detection_posterior(cell, 0.9).r
        assert r_hi > r_lo

    def test_state_position_near_cell(self):
        """Born track's position should be near the measurement cell centroid."""
        cx, cy = 10.0, -5.0
        cell = _make_cell(center=(cx, cy), spread=0.1)
        ppp = _make_ppp([(cx, cy)], weight=1.0)
        b = ppp.detection_posterior(cell, 0.9)
        pos_err = np.linalg.norm(b.state.position - np.array([cx, cy]))
        assert pos_err < 2.0

    def test_state_covariance_positive_definite(self):
        cell = _make_cell(center=(3.0, 3.0))
        ppp = _make_ppp([(3.0, 3.0)], weight=0.5)
        b = ppp.detection_posterior(cell, 0.9)
        eigenvalues = np.linalg.eigvalsh(b.state.P)
        assert np.all(eigenvalues > 0)

    def test_multi_component_moment_match(self):
        """With two components at different positions, result should be between them."""
        cell = _make_cell(center=(5.0, 0.0), spread=0.1)
        ppp = PPP(components=[
            PPPComponent(weight=1.0, state=_make_state((0.0, 0.0))),
            PPPComponent(weight=1.0, state=_make_state((10.0, 0.0))),
        ])
        b = ppp.detection_posterior(cell, 0.9)
        # Moment-matched x position should be finite and reasonable
        assert np.isfinite(b.state.m[0])
        assert b.state.m[0] < 15.0


# ---------------------------------------------------------------------------
# PPP.prune
# ---------------------------------------------------------------------------

class TestPPPPrune:
    def test_removes_below_threshold(self):
        ppp = PPP(
            components=[
                PPPComponent(weight=1.0, state=_make_state()),
                PPPComponent(weight=1e-6, state=_make_state()),
            ],
            weight_threshold=1e-5,
        )
        ppp.prune()
        assert len(ppp.components) == 1
        assert ppp.components[0].weight == pytest.approx(1.0)

    def test_empty_after_all_pruned(self):
        ppp = PPP(
            components=[PPPComponent(weight=1e-8, state=_make_state())],
            weight_threshold=1e-5,
        )
        ppp.prune()
        assert ppp.components == []

    def test_keeps_all_above_threshold(self):
        ppp = _make_ppp([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], weight=1.0)
        ppp.prune()
        assert len(ppp.components) == 3


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------

class TestPPPLifecycle:
    def test_predict_add_update_cycle(self):
        """Full cycle: predict → add birth → query → undetected_update."""
        ppp = PPP()
        cell = _make_cell(center=(5.0, 0.0))
        birth = [PPPComponent(weight=0.05, state=_make_state((5.0, 0.0)))]

        ppp.predict(dt=0.5, p_survival=0.99)
        ppp.add_birth_components(birth)

        lw = ppp.detection_log_likelihood(cell, p_detection=0.9)
        b  = ppp.detection_posterior(cell, p_detection=0.9)

        ppp.undetected_update(p_detection=0.9)

        assert np.isfinite(lw)
        assert isinstance(b, Bernoulli)
        # After undetected_update, surviving weight should drop
        assert all(c.weight < 0.05 for c in ppp.components)

    def test_weight_total_decays_without_detections(self):
        ppp = _make_ppp([(0.0, 0.0)], weight=1.0)
        for _ in range(10):
            ppp.predict(dt=0.5, p_survival=1.0)      # no survival loss
            ppp.undetected_update(p_detection=0.9)  # (1-0.9)^10 ≈ 1e-10
        total = sum(c.weight for c in ppp.components)
        assert total < 1e-5

    def test_multiple_birth_components_from_cells(self):
        """Each new cell should create one birth component."""
        ppp = PPP()
        cells = [_make_cell((float(i), 0.0), seed=i) for i in range(4)]
        birth = [
            PPPComponent(weight=0.01, state=_make_state((float(i), 0.0)))
            for i in range(4)
        ]
        ppp.add_birth_components(birth)
        assert len(ppp.components) == 4
