"""Tests for GGIW state and update equations."""

import numpy as np
import pytest

from src.tracking.ggiw.ggiw_state import GGIWState, _DOF_MIN
from src.tracking.ggiw.ggiw_update import H, predict, update, likelihood


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_state() -> GGIWState:
    """A well-conditioned GGIW state near the origin."""
    return GGIWState(
        m=np.array([5.0, 3.0, 1.0, 0.0]),
        P=np.diag([1.0, 1.0, 4.0, 4.0]),
        v=20.0,
        V=np.eye(2) * (20.0 - _DOF_MIN),  # extent mean = I
        alpha=5.0,
        beta=1.0,
    )


@pytest.fixture
def measurements_on_target(simple_state) -> np.ndarray:
    """4 measurements scattered around the target position."""
    rng = np.random.default_rng(42)
    pos = simple_state.m[:2]
    return pos + rng.standard_normal((4, 2)) * 0.5


# ---------------------------------------------------------------------------
# GGIWState
# ---------------------------------------------------------------------------

class TestGGIWState:
    def test_extent_mean_equals_identity(self, simple_state):
        # V = (v-6)*I  →  E[X] = V/(v-6) = I
        assert np.allclose(simple_state.extent_mean, np.eye(2))

    def test_rate_mean(self, simple_state):
        assert simple_state.rate_mean == pytest.approx(5.0)

    def test_position_velocity(self, simple_state):
        assert np.allclose(simple_state.position, [5.0, 3.0])
        assert np.allclose(simple_state.velocity, [1.0, 0.0])

    def test_invalid_dof_raises(self):
        with pytest.raises(ValueError, match="degrees of freedom"):
            GGIWState(m=np.zeros(4), P=np.eye(4), v=5.9,
                      V=np.eye(2), alpha=1.0, beta=1.0)

    def test_invalid_gamma_params_raise(self):
        with pytest.raises(ValueError, match="Gamma"):
            GGIWState(m=np.zeros(4), P=np.eye(4), v=10.0,
                      V=np.eye(2), alpha=-1.0, beta=1.0)

    def test_from_position_constructor(self):
        s = GGIWState.from_position([10.0, -5.0])
        assert np.allclose(s.position, [10.0, -5.0])
        assert np.allclose(s.velocity, [0.0, 0.0])
        assert s.v > _DOF_MIN


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_type(self, simple_state):
        pred = predict(simple_state, dt=0.1)
        assert isinstance(pred, GGIWState)

    def test_kinematic_position_advances(self, simple_state):
        # With vx=1, vy=0, after dt=1.0 the position should move by ~1 in x.
        pred = predict(simple_state, dt=1.0)
        assert pred.m[0] == pytest.approx(simple_state.m[0] + simple_state.m[2])

    def test_covariance_grows(self, simple_state):
        pred = predict(simple_state, dt=0.1)
        # Trace of kinematic covariance should increase
        assert np.trace(pred.P) > np.trace(simple_state.P)

    def test_extent_mean_preserved(self, simple_state):
        # Prediction should not change E[X] = V/(v-6)
        pred = predict(simple_state, dt=0.1, eta_x=0.8)
        assert np.allclose(pred.extent_mean, simple_state.extent_mean, atol=1e-10)

    def test_dof_decreases(self, simple_state):
        pred = predict(simple_state, dt=0.1, eta_x=0.8)
        assert pred.v < simple_state.v

    def test_rate_mean_preserved_uncertainty_grows(self, simple_state):
        # Scaling both α and β by η preserves the mean E[γ] = α/β.
        # The prediction inflates uncertainty (reduces effective sample size).
        pred = predict(simple_state, dt=0.1, eta_gamma=0.9)
        assert pred.rate_mean == pytest.approx(simple_state.rate_mean)
        # Gamma variance = α/β² → variance grows when η < 1
        prior_var = simple_state.alpha / simple_state.beta**2
        pred_var = pred.alpha / pred.beta**2
        assert pred_var > prior_var

    def test_v_stays_above_minimum(self):
        # Start near minimum; ensure prediction doesn't violate constraint
        s = GGIWState(m=np.zeros(4), P=np.eye(4), v=7.0,
                      V=np.eye(2), alpha=1.0, beta=1.0)
        pred = predict(s, dt=0.1, eta_x=0.01)
        assert pred.v > _DOF_MIN


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_output_type(self, simple_state, measurements_on_target):
        post = update(simple_state, measurements_on_target)
        assert isinstance(post, GGIWState)

    def test_dof_increases_by_n(self, simple_state, measurements_on_target):
        n = len(measurements_on_target)
        post = update(simple_state, measurements_on_target)
        assert post.v == pytest.approx(simple_state.v + n)

    def test_alpha_increases_by_n(self, simple_state, measurements_on_target):
        n = len(measurements_on_target)
        post = update(simple_state, measurements_on_target)
        assert post.alpha == pytest.approx(simple_state.alpha + n)

    def test_beta_increases_by_one(self, simple_state, measurements_on_target):
        post = update(simple_state, measurements_on_target)
        assert post.beta == pytest.approx(simple_state.beta + 1.0)

    def test_kinematic_covariance_shrinks(self, simple_state, measurements_on_target):
        post = update(simple_state, measurements_on_target)
        assert np.trace(post.P) < np.trace(simple_state.P)

    def test_covariance_symmetric_pd(self, simple_state, measurements_on_target):
        post = update(simple_state, measurements_on_target)
        assert np.allclose(post.P, post.P.T)
        assert np.allclose(post.V, post.V.T)
        assert np.all(np.linalg.eigvalsh(post.P) > 0)
        assert np.all(np.linalg.eigvalsh(post.V) > 0)

    def test_position_pulled_toward_measurements(self, simple_state):
        # Measurements far from prior position — posterior should move toward them
        far_measurements = np.array([[20.0, 20.0], [20.5, 19.5], [19.5, 20.5]])
        post = update(simple_state, far_measurements)
        prior_dist = np.linalg.norm(simple_state.position - [20.0, 20.0])
        post_dist = np.linalg.norm(post.position - [20.0, 20.0])
        assert post_dist < prior_dist

    def test_single_measurement(self, simple_state):
        single = np.array([[5.0, 3.0]])
        post = update(simple_state, single)
        assert post.v == pytest.approx(simple_state.v + 1)

    def test_empty_measurements_raises(self, simple_state):
        with pytest.raises(ValueError, match="at least one"):
            update(simple_state, np.empty((0, 2)))

    def test_multiple_updates_reduce_uncertainty(self, simple_state):
        # Feeding consistent measurements should reduce kinematic uncertainty
        rng = np.random.default_rng(0)
        state = simple_state
        for _ in range(10):
            Z = state.m[:2] + rng.standard_normal((5, 2)) * 0.3
            state = update(predict(state, dt=0.1), Z)
        assert np.trace(state.P) < np.trace(simple_state.P)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

class TestLikelihood:
    def test_positive_scalar(self, simple_state, measurements_on_target):
        l = likelihood(simple_state, measurements_on_target)
        assert isinstance(l, float)
        assert l > 0.0

    def test_closer_measurements_give_higher_likelihood(self, simple_state):
        # Centroid near the target position vs. centroid far away.
        near = simple_state.m[:2] + np.array([[0.1, 0.0], [0.0, 0.1]])
        far = simple_state.m[:2] + np.array([[8.0, 8.0], [8.1, 8.1]])
        assert likelihood(simple_state, near) > likelihood(simple_state, far)

    def test_empty_returns_zero(self, simple_state):
        assert likelihood(simple_state, np.empty((0, 2))) == 0.0

    def test_single_measurement(self, simple_state):
        l = likelihood(simple_state, np.array([[5.0, 3.0]]))
        assert l > 0.0
