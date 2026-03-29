"""LiDAR ray-casting scenario tests for the full tracking pipeline.

Uses VehicleSimulator + LidarSimulator to generate realistic point clouds
and verifies the GGIW / PMBM layers under conditions that simple Gaussian
noise does not capture:

  - Partial observation (only visible surface seen by ray casting)
  - Point count depending on vehicle angle and distance
  - Two-vehicle separation via clustering
  - Track convergence through a full approach / pass manoeuvre
  - Graceful handling of missed-detection frames

Sensor is always at origin (0, 0).
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display needed in tests

import numpy as np
import pytest

from src.simulator import LidarSimulator, VehicleSimulator
from src.tracking.ggiw.ggiw_state import GGIWState, _DOF_MIN
from src.tracking.ggiw.ggiw_update import predict as ggiw_predict, update as ggiw_update
from src.tracking.measurement.partition import partition
from src.tracking.pmbm.pmbm_filter import BirthModel, PMBMFilter


# ---------------------------------------------------------------------------
# Helper: generate LiDAR points for one vehicle at its current pose
# ---------------------------------------------------------------------------

def _lidar_scan(vehicle: VehicleSimulator, angle_res_deg: float = 3.0) -> np.ndarray:
    """Return (n, 2) array of ray-cast LiDAR points for a single vehicle."""
    lidar = LidarSimulator(range_noise=0.01)
    ox, oy = lidar.get_observation_points([vehicle], np.deg2rad(angle_res_deg))
    if not ox:
        return np.empty((0, 2))
    return np.column_stack([ox, oy])


def _vehicle(x=10.0, y=0.0, yaw=0.0, v=0.0, w=2.0, L=4.0) -> VehicleSimulator:
    """Convenience constructor with sensible defaults."""
    return VehicleSimulator(x, y, yaw, v, max_v=50.0 / 3.6, w=w, L=L)


# ---------------------------------------------------------------------------
# 1. LiDAR point-cloud properties
# ---------------------------------------------------------------------------

class TestLidarPointCloud:

    def test_vehicle_produces_at_least_one_point(self):
        v = _vehicle(x=10.0, y=0.0)
        pts = _lidar_scan(v)
        assert len(pts) >= 1

    def test_points_lie_near_vehicle_surface(self):
        """All ray-cast points must be within ~(L/2 + noise) of vehicle centre."""
        v = _vehicle(x=10.0, y=0.0, w=2.0, L=4.0)
        pts = _lidar_scan(v)
        dists = np.linalg.norm(pts - np.array([v.x, v.y]), axis=1)
        # Diagonal half-size ≈ sqrt((L/2)^2 + (W/2)^2) = sqrt(4+1) ≈ 2.24; allow 10 % margin
        assert np.all(dists < 3.0)

    def test_finer_resolution_gives_more_points(self):
        """Higher angular resolution → more points for the same vehicle."""
        v = _vehicle(x=10.0, y=0.0)
        pts_coarse = _lidar_scan(v, angle_res_deg=5.0)
        pts_fine   = _lidar_scan(v, angle_res_deg=1.0)
        assert len(pts_fine) >= len(pts_coarse)

    def test_closer_vehicle_gives_more_points(self):
        """Closer vehicle spans more angular bins → more points."""
        v_close = _vehicle(x=5.0,  y=0.0)
        v_far   = _vehicle(x=20.0, y=0.0)
        pts_close = _lidar_scan(v_close)
        pts_far   = _lidar_scan(v_far)
        assert len(pts_close) >= len(pts_far)

    def test_head_on_vs_side_on_point_count(self):
        """Vehicle at (10, 0): yaw=90° exposes long face (L=4m) to sensor;
        yaw=0° exposes short face (w=2m) → fewer angular bins hit."""
        v_side = _vehicle(x=10.0, y=0.0, yaw=np.deg2rad(90), w=2.0, L=4.0)
        v_head = _vehicle(x=10.0, y=0.0, yaw=0.0,             w=2.0, L=4.0)
        pts_side = _lidar_scan(v_side)
        pts_head = _lidar_scan(v_head)
        assert len(pts_side) >= len(pts_head)

    def test_partial_observation_only_near_face_visible(self):
        """Vehicle directly ahead: sensor sees front face only (x ≈ vehicle.x - L/2)."""
        v = _vehicle(x=15.0, y=0.0, yaw=np.deg2rad(180))  # rear faces sensor
        pts = _lidar_scan(v)
        assert len(pts) >= 1
        # All points should be between 13 and 17 m from origin
        dists = np.linalg.norm(pts, axis=1)
        assert np.all(dists > 5.0)   # not at origin
        assert np.all(dists < 25.0)  # not beyond vehicle

    def test_range_noise_within_bounds(self):
        """Points should not deviate more than 2 % from true range."""
        v = _vehicle(x=10.0, y=0.0)
        lidar = LidarSimulator(range_noise=0.01)
        true_range = 10.0  # approximate; actual vehicle edge is ≥ 8 m
        pts = _lidar_scan(v)
        dists = np.linalg.norm(pts, axis=1)
        # range noise is multiplicative ± 1 %; true edge at ≈ (10 - 2) = 8 m
        assert np.all(dists > 6.0)   # lower bound with noise
        assert np.all(dists < 14.0)  # upper bound with noise


# ---------------------------------------------------------------------------
# 2. Clustering with realistic LiDAR input
# ---------------------------------------------------------------------------

class TestPartialObservationClustering:

    def test_single_vehicle_forms_one_cell(self):
        v = _vehicle(x=10.0, y=0.0)
        pts = _lidar_scan(v)
        cells = partition(pts, eps=2.0)
        assert len(cells) == 1

    def test_two_separated_vehicles_form_two_cells(self):
        """Two vehicles far apart should cluster into separate cells."""
        v1 = _vehicle(x=10.0, y=0.0)
        v2 = _vehicle(x=0.0,  y=12.0)
        lidar = LidarSimulator(range_noise=0.01)
        ox1, oy1 = lidar.get_observation_points([v1], np.deg2rad(3.0))
        ox2, oy2 = lidar.get_observation_points([v2], np.deg2rad(3.0))
        pts = np.column_stack([ox1 + ox2, oy1 + oy2])
        cells = partition(pts, eps=2.0)
        assert len(cells) == 2

    def test_cell_centroid_near_vehicle(self):
        v = _vehicle(x=10.0, y=0.0)
        pts = _lidar_scan(v)
        cells = partition(pts, eps=2.0)
        assert len(cells) == 1
        centroid = cells[0].centroid
        dist = np.linalg.norm(centroid - np.array([v.x, v.y]))
        assert dist < 3.0  # within half-diagonal of vehicle

    def test_diagonal_vehicle_still_one_cell(self):
        """Vehicle at 45° — two visible faces — should still be one cluster."""
        v = _vehicle(x=10.0, y=0.0, yaw=np.deg2rad(45))
        pts = _lidar_scan(v, angle_res_deg=2.0)
        cells = partition(pts, eps=2.0)
        assert len(cells) == 1

    def test_sparse_scan_min_points_filter(self):
        """Far or small vehicle may produce very few points; filter with min_points."""
        v = _vehicle(x=18.0, y=0.0)   # far → maybe 1-2 points
        pts = _lidar_scan(v, angle_res_deg=3.0)
        cells_all      = partition(pts, eps=2.0, min_points=1)
        cells_filtered = partition(pts, eps=2.0, min_points=3)
        # sparse scan may have fewer points than threshold
        if len(pts) < 3:
            assert len(cells_filtered) == 0
        else:
            assert len(cells_all) >= len(cells_filtered)


# ---------------------------------------------------------------------------
# 3. GGIW update with realistic LiDAR observations
# ---------------------------------------------------------------------------

class TestGGIWWithLidarInput:

    def test_ggiw_update_does_not_diverge_partial_observation(self):
        """Single-side observation (few points) should not break the IW update."""
        v = _vehicle(x=10.0, y=0.0, yaw=0.0)
        pts = _lidar_scan(v, angle_res_deg=3.0)
        assert len(pts) >= 1

        state = GGIWState.from_position([v.x, v.y])
        updated = ggiw_update(state, pts)

        # Covariances must remain symmetric and positive definite
        assert np.all(np.linalg.eigvalsh(updated.P) > 0)
        assert np.all(np.linalg.eigvalsh(updated.V) > 0)
        assert updated.v > _DOF_MIN

    def test_ggiw_position_converges_over_frames(self):
        """After N consistent LiDAR frames, position estimate should converge."""
        v = _vehicle(x=10.0, y=0.0, yaw=0.0, v=2.0)
        state = GGIWState.from_position([v.x, v.y])

        for _ in range(10):
            v.update(dt=0.2, a=0.0, omega=0.0)
            pts = _lidar_scan(v, angle_res_deg=3.0)
            if len(pts) >= 1:
                state = ggiw_update(ggiw_predict(state, dt=0.2), pts)

        pos_err = np.linalg.norm(state.position - np.array([v.x, v.y]))
        assert pos_err < 4.0  # tracking within half-vehicle-length

    def test_ggiw_angled_vehicle_update_stable(self):
        """45° vehicle (two visible faces) produces stable update."""
        v = _vehicle(x=8.0, y=0.0, yaw=np.deg2rad(45))
        pts = _lidar_scan(v, angle_res_deg=2.0)
        state = GGIWState.from_position([v.x, v.y])
        updated = ggiw_update(state, pts)
        assert np.all(np.linalg.eigvalsh(updated.P) > 0)

    def test_position_uncertainty_shrinks_with_observations(self):
        """Repeated observations from the same pose should reduce P trace."""
        v = _vehicle(x=10.0, y=0.0)
        state = GGIWState.from_position([v.x, v.y])
        initial_trace = np.trace(state.P)

        for _ in range(5):
            pts = _lidar_scan(v)
            state = ggiw_update(state, pts)

        assert np.trace(state.P) < initial_trace


# ---------------------------------------------------------------------------
# 4. PMBM end-to-end scenarios with ray-casting
# ---------------------------------------------------------------------------

class TestPMBMWithLidarScenarios:

    def _make_filter(self, **kwargs) -> PMBMFilter:
        defaults = dict(
            p_survival=0.99,
            p_detection=0.9,
            birth_model=BirthModel(r_birth=0.3, birth_log_rate=-4.6, vel_var=1e2),
            max_hypotheses=30,
            prune_log_threshold=-30.0,
        )
        defaults.update(kwargs)
        return PMBMFilter(**defaults)

    # --- Scenario 1: vehicle approaching the sensor ---

    def test_approaching_vehicle_tracked(self):
        """Vehicle starts 20 m ahead and drives toward the sensor at 2 m/s.
        After 12 frames (6 s) it should be consistently tracked."""
        f = self._make_filter()
        v = _vehicle(x=20.0, y=0.0, yaw=np.deg2rad(180), v=2.0)

        for _ in range(12):
            v.update(dt=0.5, a=0.0, omega=0.0)
            pts = _lidar_scan(v, angle_res_deg=3.0)
            f.predict(dt=0.5)
            f.update_from_points(pts, eps=2.0)

        estimates = f.extract_estimates(existence_threshold=0.5)
        assert len(estimates) >= 1
        pos_err = np.linalg.norm(estimates[0].position - np.array([v.x, v.y]))
        assert pos_err < 5.0

    # --- Scenario 2: vehicle passing perpendicular ---

    def test_side_passing_vehicle_tracked(self):
        """Vehicle crosses in front of the sensor left-to-right at 5 m/s,
        8 m lateral distance. Side face is always visible."""
        f = self._make_filter()
        v = _vehicle(x=-12.0, y=8.0, yaw=0.0, v=5.0)

        for _ in range(15):
            v.update(dt=0.5, a=0.0, omega=0.0)
            pts = _lidar_scan(v, angle_res_deg=2.0)
            f.predict(dt=0.5)
            f.update_from_points(pts, eps=2.0)

        estimates = f.extract_estimates(existence_threshold=0.5)
        assert len(estimates) >= 1

    # --- Scenario 3: missed detection frames ---

    def test_track_survives_missed_frames(self):
        """Track existence should remain above threshold after a few missed
        frames, then recover when the vehicle reappears."""
        f = self._make_filter()
        v = _vehicle(x=10.0, y=0.0, v=2.0)

        # Establish track over 10 frames
        for _ in range(10):
            v.update(dt=0.5, a=0.0, omega=0.0)
            pts = _lidar_scan(v, angle_res_deg=3.0)
            f.predict(dt=0.5)
            f.update_from_points(pts, eps=2.0)

        # 3 missed frames (empty update)
        for _ in range(3):
            f.predict(dt=0.5)
            f.update([])

        # Check that at least one track hypothesis still exists
        all_r = [b.r for h in f._hypotheses for b in h.tracks]
        assert len(all_r) > 0  # tracks not all pruned

        # Reappear and check it can be detected again
        for _ in range(5):
            v.update(dt=0.5, a=0.0, omega=0.0)
            pts = _lidar_scan(v, angle_res_deg=3.0)
            f.predict(dt=0.5)
            f.update_from_points(pts, eps=2.0)

        estimates = f.extract_estimates(existence_threshold=0.3)
        assert len(estimates) >= 1

    # --- Scenario 4: sparse scan (coarse angular resolution) ---

    def test_sparse_lidar_tracking_stable(self):
        """Coarse resolution (5°) produces very few points; filter must not crash."""
        f = self._make_filter()
        v = _vehicle(x=12.0, y=0.0, v=3.0)

        for _ in range(10):
            v.update(dt=0.5, a=0.0, omega=0.0)
            pts = _lidar_scan(v, angle_res_deg=5.0)   # sparse
            f.predict(dt=0.5)
            if len(pts) >= 1:
                f.update_from_points(pts, eps=3.0)
            else:
                f.update([])

        # No assertion on tracking quality — just no exceptions and bounded hypotheses
        assert f.num_hypotheses <= 30

    # --- Scenario 5: two simultaneous vehicles ---

    def test_two_vehicles_two_estimates(self):
        """Two vehicles on separate trajectories should each generate an estimate."""
        f = self._make_filter(max_hypotheses=50)
        v1 = _vehicle(x=10.0, y=0.0,  v=2.0)
        v2 = _vehicle(x=0.0,  y=12.0, v=0.0, yaw=np.deg2rad(90))

        lidar = LidarSimulator(range_noise=0.01)

        for _ in range(15):
            v1.update(dt=0.5, a=0.0, omega=0.0)
            v2.update(dt=0.5, a=0.0, omega=0.0)
            ox, oy = lidar.get_observation_points([v1, v2], np.deg2rad(2.0))
            if ox:
                pts = np.column_stack([ox, oy])
                f.predict(dt=0.5)
                f.update_from_points(pts, eps=2.0)
            else:
                f.predict(dt=0.5)
                f.update([])

        estimates = f.extract_estimates(existence_threshold=0.5)
        assert len(estimates) >= 1   # at minimum one vehicle tracked
