"""Unit tests for src/tracking/shape/."""

import numpy as np
import pytest

from src.tracking.shape.rectangle_fitting import OBBResult, fit_rectangle
from src.tracking.shape.smoothing import ExponentialSmoother, OBBKalmanSmoother


# ---------------------------------------------------------------------------
# Helpers

def _rect_points(theta: float, length: float, width: float,
                 center=(0.0, 0.0), n: int = 40) -> np.ndarray:
    """Generate points uniformly on the perimeter of a rotated rectangle."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = []
    # Long edges (top and bottom)
    for t in np.linspace(-length / 2, length / 2, n // 2):
        pts.append([t, width / 2])
        pts.append([t, -width / 2])
    # Short edges
    for t in np.linspace(-width / 2, width / 2, n // 2):
        pts.append([length / 2, t])
        pts.append([-length / 2, t])
    pts_arr = np.array(pts)
    return (R @ pts_arr.T).T + np.array(center)


# ---------------------------------------------------------------------------
# OBBResult


class TestOBBResult:
    def test_corners_returns_4_rows(self):
        obb = OBBResult(theta=0.0, length=4.0, width=2.0,
                        center=np.array([0.0, 0.0]))
        assert obb.corners().shape == (4, 2)

    def test_corners_axis_aligned(self):
        obb = OBBResult(theta=0.0, length=4.0, width=2.0,
                        center=np.array([0.0, 0.0]))
        corners = obb.corners()
        xs = sorted(corners[:, 0])
        ys = sorted(corners[:, 1])
        assert np.isclose(xs[0], -2.0) and np.isclose(xs[-1], 2.0)
        assert np.isclose(ys[0], -1.0) and np.isclose(ys[-1], 1.0)

    def test_corners_rotated(self):
        obb = OBBResult(theta=np.pi / 4, length=4.0, width=2.0,
                        center=np.array([0.0, 0.0]))
        corners = obb.corners()
        # All corners should be equidistant from centre (rhombus-like)
        dists = np.linalg.norm(corners, axis=1)
        assert np.allclose(dists, dists[0], atol=1e-10)

    def test_corners_translated(self):
        cx, cy = 5.0, 3.0
        obb = OBBResult(theta=0.0, length=2.0, width=1.0,
                        center=np.array([cx, cy]))
        corners = obb.corners()
        assert np.allclose(corners.mean(axis=0), [cx, cy], atol=1e-10)

    def test_equality(self):
        o1 = OBBResult(theta=0.1, length=4.0, width=2.0,
                       center=np.array([1.0, 2.0]))
        o2 = OBBResult(theta=0.1, length=4.0, width=2.0,
                       center=np.array([1.0, 2.0]))
        assert o1 == o2


# ---------------------------------------------------------------------------
# fit_rectangle


class TestFitRectangle:
    def test_axis_aligned_rectangle(self):
        pts = _rect_points(theta=0.0, length=4.0, width=2.0)
        obb = fit_rectangle(pts)
        assert np.isclose(obb.length, 4.0, atol=0.3)
        assert np.isclose(obb.width, 2.0, atol=0.3)
        assert np.isclose(obb.theta, 0.0, atol=0.1)

    def test_rotated_45_degrees(self):
        pts = _rect_points(theta=np.pi / 4, length=4.0, width=2.0)
        obb = fit_rectangle(pts)
        assert np.isclose(obb.length, 4.0, atol=0.3)
        assert np.isclose(obb.width, 2.0, atol=0.3)
        assert np.isclose(abs(obb.theta), np.pi / 4, atol=0.1)

    def test_rotated_30_degrees(self):
        pts = _rect_points(theta=np.deg2rad(30), length=5.0, width=2.0)
        obb = fit_rectangle(pts)
        assert np.isclose(obb.length, 5.0, atol=0.4)
        assert np.isclose(obb.width, 2.0, atol=0.4)

    def test_center_near_true_center(self):
        cx, cy = 10.0, -3.0
        pts = _rect_points(theta=0.3, length=4.0, width=2.0, center=(cx, cy))
        obb = fit_rectangle(pts)
        assert np.linalg.norm(obb.center - np.array([cx, cy])) < 0.5

    def test_length_geq_width(self):
        """OBB convention: length >= width."""
        pts = _rect_points(theta=0.0, length=2.0, width=4.0)  # width > length
        obb = fit_rectangle(pts)
        assert obb.length >= obb.width

    def test_theta_in_range(self):
        for angle in np.linspace(-1.5, 1.5, 10):
            pts = _rect_points(theta=angle, length=4.0, width=2.0)
            obb = fit_rectangle(pts)
            assert -np.pi / 2 <= obb.theta < np.pi / 2

    def test_partial_observation_robust(self):
        """Fitting one-sided points (partial observation) should not explode."""
        pts = _rect_points(theta=0.0, length=4.0, width=2.0)
        pts = pts[pts[:, 0] > 0]   # keep only right half
        obb = fit_rectangle(pts)
        assert np.isfinite(obb.length) and np.isfinite(obb.width)
        assert obb.length > 0 and obb.width > 0

    def test_min_size_enforced(self):
        """Degenerate collinear input must still return positive size."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        obb = fit_rectangle(pts, min_size=0.1)
        assert obb.length >= 0.1
        assert obb.width >= 0.1

    def test_two_points_minimum(self):
        pts = np.array([[0.0, 0.0], [4.0, 0.0]])
        obb = fit_rectangle(pts)
        assert np.isfinite(obb.length)

    def test_fewer_than_two_points_raises(self):
        with pytest.raises(ValueError):
            fit_rectangle(np.array([[1.0, 2.0]]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            fit_rectangle(np.ones((5, 3)))

    def test_percentile_vs_minmax_with_outlier(self):
        """Percentile sizing should be smaller than min-max when an outlier is present."""
        pts = _rect_points(theta=0.0, length=4.0, width=2.0)
        outlier = np.array([[50.0, 0.0]])  # far outlier
        pts_with_outlier = np.vstack([pts, outlier])

        obb_pct = fit_rectangle(pts_with_outlier, low_pct=5, high_pct=95)
        obb_minmax = fit_rectangle(pts_with_outlier, low_pct=0, high_pct=100)

        assert obb_pct.length < obb_minmax.length


# ---------------------------------------------------------------------------
# ExponentialSmoother


class TestExponentialSmoother:
    def _obb(self, theta=0.0, length=4.0, width=2.0,
             center=(0.0, 0.0)) -> OBBResult:
        return OBBResult(theta=theta, length=length, width=width,
                         center=np.array(center, dtype=float))

    def test_first_update_returns_input(self):
        sm = ExponentialSmoother(alpha=0.5)
        obb = self._obb(theta=0.2, length=5.0, width=1.5)
        out = sm.update(obb)
        assert np.isclose(out.theta, 0.2)
        assert np.isclose(out.length, 5.0)
        assert np.isclose(out.width, 1.5)

    def test_smoothing_converges_to_constant_input(self):
        sm = ExponentialSmoother(alpha=0.3)
        obb = self._obb(theta=0.5, length=4.0, width=2.0)
        for _ in range(50):
            out = sm.update(obb)
        assert np.isclose(out.length, 4.0, atol=0.01)
        assert np.isclose(out.width, 2.0, atol=0.01)

    def test_length_tracks_increasing_sequence(self):
        sm = ExponentialSmoother(alpha=0.5)
        obbs = [self._obb(length=float(i)) for i in range(1, 10)]
        outputs = [sm.update(o) for o in obbs]
        lengths = [o.length for o in outputs]
        # Output should generally increase
        assert lengths[-1] > lengths[0]

    def test_angle_wrapping_near_pi_over_2(self):
        """Smoother must not jump when theta crosses ±π/2 boundary."""
        sm = ExponentialSmoother(alpha=0.5)
        sm.update(self._obb(theta=np.pi / 2 - 0.05))
        out = sm.update(self._obb(theta=-(np.pi / 2 - 0.05)))
        # Result should be near ±π/2, not in the middle of range
        assert abs(out.theta) > 1.0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ExponentialSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            ExponentialSmoother(alpha=1.5)

    def test_reset_clears_state(self):
        sm = ExponentialSmoother(alpha=0.3)
        for _ in range(10):
            sm.update(self._obb(theta=1.0, length=10.0))
        sm.reset()
        out = sm.update(self._obb(theta=0.0, length=1.0))
        assert np.isclose(out.length, 1.0)

    def test_returns_obb_result(self):
        sm = ExponentialSmoother()
        out = sm.update(self._obb())
        assert isinstance(out, OBBResult)

    def test_alpha_one_is_passthrough(self):
        sm = ExponentialSmoother(alpha=1.0)
        obb = self._obb(theta=0.3, length=3.5, width=1.2)
        sm.update(self._obb())  # initialise
        out = sm.update(obb)
        assert np.isclose(out.length, 3.5)
        assert np.isclose(out.width, 1.2)


# ---------------------------------------------------------------------------
# OBBKalmanSmoother


class TestOBBKalmanSmoother:
    def _obb(self, theta=0.0, length=4.0, width=2.0,
             center=(0.0, 0.0)) -> OBBResult:
        return OBBResult(theta=theta, length=length, width=width,
                         center=np.array(center, dtype=float))

    def test_returns_obb_result(self):
        sm = OBBKalmanSmoother()
        out = sm.update(self._obb())
        assert isinstance(out, OBBResult)

    def test_first_update_close_to_input(self):
        sm = OBBKalmanSmoother()
        obb = self._obb(theta=0.3, length=5.0, width=1.5)
        out = sm.update(obb)
        assert np.isclose(out.length, 5.0, atol=0.2)
        assert np.isclose(out.width, 1.5, atol=0.2)

    def test_noisy_measurements_smoothed(self):
        """Kalman should reduce variance compared to raw noisy measurements."""
        rng = np.random.default_rng(0)
        sm = OBBKalmanSmoother(meas_noise_size=0.5)
        true_length = 4.0
        noisy = true_length + rng.normal(0, 0.5, 30)
        outputs = [sm.update(self._obb(length=float(l))) for l in noisy]
        smoothed_lengths = np.array([o.length for o in outputs[10:]])
        assert smoothed_lengths.std() < np.std(noisy[10:])

    def test_predict_without_measurement(self):
        """predict() should not crash and should return an OBBResult."""
        sm = OBBKalmanSmoother()
        sm.update(self._obb())
        out = sm.predict()
        assert isinstance(out, OBBResult)

    def test_predict_increases_uncertainty(self):
        """After predict(), covariance trace should grow."""
        sm = OBBKalmanSmoother()
        sm.update(self._obb())
        P_before = sm._P.trace()
        sm.predict()
        P_after = sm._P.trace()
        assert P_after > P_before

    def test_converges_to_constant(self):
        sm = OBBKalmanSmoother()
        obb = self._obb(theta=0.4, length=4.0, width=2.0)
        for _ in range(30):
            out = sm.update(obb)
        assert np.isclose(out.length, 4.0, atol=0.1)
        assert np.isclose(out.width, 2.0, atol=0.1)

    def test_length_geq_width(self):
        sm = OBBKalmanSmoother()
        for _ in range(20):
            out = sm.update(self._obb(length=2.0, width=4.0))
        assert out.length >= out.width

    def test_theta_in_range(self):
        sm = OBBKalmanSmoother()
        for angle in np.linspace(-1.4, 1.4, 20):
            out = sm.update(self._obb(theta=float(angle)))
            assert -np.pi / 2 <= out.theta < np.pi / 2

    def test_reset_clears_state(self):
        sm = OBBKalmanSmoother()
        for _ in range(10):
            sm.update(self._obb(theta=1.0, length=10.0))
        sm.reset()
        assert not sm._initialized
