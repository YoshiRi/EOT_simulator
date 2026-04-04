"""Tests for measurement clustering and partition."""

import numpy as np
import pytest

from src.tracking.measurement.clustering import MeasurementCell, cluster_points
from src.tracking.measurement.partition import partition


# ---------------------------------------------------------------------------
# MeasurementCell
# ---------------------------------------------------------------------------

class TestMeasurementCell:
    def test_n(self):
        cell = MeasurementCell(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert cell.n == 3

    def test_centroid(self):
        cell = MeasurementCell(np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]))
        assert np.allclose(cell.centroid, [1.0, 2.0 / 3.0])

    def test_scatter_single_point_is_zero(self):
        cell = MeasurementCell(np.array([[3.0, 5.0]]))
        assert np.allclose(cell.scatter, np.zeros((2, 2)))

    def test_scatter_symmetric(self):
        rng = np.random.default_rng(0)
        cell = MeasurementCell(rng.standard_normal((10, 2)))
        assert np.allclose(cell.scatter, cell.scatter.T)

    def test_scatter_positive_semidefinite(self):
        rng = np.random.default_rng(1)
        cell = MeasurementCell(rng.standard_normal((8, 2)))
        eigvals = np.linalg.eigvalsh(cell.scatter)
        assert np.all(eigvals >= -1e-12)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            MeasurementCell(np.ones((5, 3)))  # 3 columns, not 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MeasurementCell(np.empty((0, 2)))

    def test_single_point(self):
        cell = MeasurementCell(np.array([[7.0, -3.0]]))
        assert cell.n == 1
        assert np.allclose(cell.centroid, [7.0, -3.0])

    def test_points_coerced_to_float(self):
        cell = MeasurementCell(np.array([[1, 2], [3, 4]]))
        assert cell.points.dtype == float


# ---------------------------------------------------------------------------
# cluster_points
# ---------------------------------------------------------------------------

class TestClusterPoints:
    def test_two_well_separated_groups(self):
        # Group A near (0, 0), Group B near (100, 100)
        group_a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        group_b = np.array([[100.0, 100.0], [100.1, 100.0]])
        pts = np.vstack([group_a, group_b])
        clusters = cluster_points(pts, eps=1.0)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [2, 3]

    def test_all_points_in_one_cluster(self):
        pts = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])  # chain within eps=1
        clusters = cluster_points(pts, eps=1.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_single_point(self):
        clusters = cluster_points(np.array([[1.0, 2.0]]), eps=1.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_empty_input(self):
        clusters = cluster_points(np.empty((0, 2)), eps=1.0)
        assert clusters == []

    def test_min_points_filters_small_clusters(self):
        # Singleton at origin, large cluster elsewhere
        pts = np.vstack([
            [[0.0, 0.0]],
            np.ones((5, 2)) * 50.0 + np.random.default_rng(7).standard_normal((5, 2)) * 0.1,
        ])
        clusters = cluster_points(pts, eps=1.0, min_points=2)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_three_clusters(self):
        rng = np.random.default_rng(42)
        centers = [(0, 0), (20, 0), (0, 20)]
        pts = np.vstack([c + rng.standard_normal((6, 2)) * 0.3 for c in centers])
        clusters = cluster_points(pts, eps=1.0)
        assert len(clusters) == 3

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            cluster_points(np.ones((5, 3)), eps=1.0)

    def test_output_points_match_input(self):
        pts = np.array([[1.0, 2.0], [1.1, 2.0], [50.0, 50.0]])
        clusters = cluster_points(pts, eps=0.5)
        total = sum(len(c) for c in clusters)
        assert total == len(pts)


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------

class TestPartition:
    def test_returns_measurement_cells(self):
        pts = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
        cells = partition(pts, eps=1.0)
        assert all(isinstance(c, MeasurementCell) for c in cells)

    def test_two_objects_two_cells(self):
        rng = np.random.default_rng(5)
        obj_a = rng.standard_normal((4, 2)) * 0.2
        obj_b = rng.standard_normal((4, 2)) * 0.2 + 20.0
        cells = partition(np.vstack([obj_a, obj_b]), eps=1.0)
        assert len(cells) == 2

    def test_empty_input_returns_empty(self):
        cells = partition(np.empty((0, 2)), eps=1.0)
        assert cells == []

    def test_min_points_forwarded(self):
        pts = np.vstack([[[0.0, 0.0]], np.ones((5, 2)) * 50.0])
        cells_default = partition(pts, eps=1.0, min_points=1)
        cells_filtered = partition(pts, eps=1.0, min_points=2)
        assert len(cells_default) > len(cells_filtered)

    def test_cell_points_cover_all_input(self):
        rng = np.random.default_rng(9)
        pts = rng.standard_normal((20, 2)) * 0.5  # tight cloud → one cell
        cells = partition(pts, eps=2.0)
        total = sum(c.n for c in cells)
        assert total == len(pts)

    def test_cells_usable_with_ggiw_update(self):
        """Smoke test: cells produced here can be passed to ggiw update."""
        from src.tracking.ggiw.ggiw_state import GGIWState, _DOF_MIN
        from src.tracking.ggiw.ggiw_update import update, predict

        state = GGIWState.from_position([5.0, 5.0])
        rng = np.random.default_rng(11)
        raw = np.array([5.0, 5.0]) + rng.standard_normal((6, 2)) * 0.3
        cells = partition(raw, eps=1.0)
        assert len(cells) == 1
        updated = update(predict(state, dt=0.1), cells[0].points)
        assert isinstance(updated, GGIWState)
