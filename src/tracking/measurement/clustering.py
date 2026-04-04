"""Point cloud clustering for extended object tracking.

Converts a raw sensor point cloud into a set of clusters, where each
cluster is treated as one measurement cell (one target candidate) by
the PMBM filter.

Algorithm
---------
Single-linkage distance-threshold clustering:

1. Build a KD-tree over all points.
2. Find all pairs (i, j) with Euclidean distance ≤ eps.
3. Group connected pairs via union-find.
4. Discard clusters with fewer than ``min_points`` members.

Single-linkage is appropriate for LiDAR / Radar scenarios where
objects are well-separated in space. It runs in O(n log n) thanks to
the KD-tree range query.

Note: if objects can appear very close together, consider raising eps
adaptively or switching to a multi-pass algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# MeasurementCell
# ---------------------------------------------------------------------------

@dataclass
class MeasurementCell:
    """A set of spatially clustered point measurements.

    This is the atomic unit fed into the PMBM update step.
    Pre-computes centroid and scatter once on construction.

    Attributes:
        points: Raw measurements in this cell, shape (n, 2).
    """

    points: np.ndarray  # (n, 2)

    def __post_init__(self) -> None:
        self.points = np.atleast_2d(np.asarray(self.points, dtype=float))
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must have shape (n, 2), got {self.points.shape}")
        if len(self.points) == 0:
            raise ValueError("MeasurementCell must contain at least one point")

    @property
    def n(self) -> int:
        """Number of points in this cell."""
        return len(self.points)

    @property
    def centroid(self) -> np.ndarray:
        """Sample mean of the points, shape (2,)."""
        return self.points.mean(axis=0)

    @property
    def scatter(self) -> np.ndarray:
        """Scatter matrix Σ(z_i - ẑ)(z_i - ẑ)^T, shape (2, 2)."""
        d = self.points - self.centroid
        return d.T @ d


# ---------------------------------------------------------------------------
# Union-Find (path compression + union by rank)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_points(
    points: np.ndarray,
    eps: float,
    min_points: int = 1,
) -> list[np.ndarray]:
    """Cluster a 2-D point cloud by distance threshold.

    Args:
        points:     Input points, shape (n, 2).
        eps:        Maximum Euclidean distance for two points to be
                    considered neighbours (same cluster).
        min_points: Minimum cluster size; smaller clusters are dropped.

    Returns:
        List of arrays, each of shape (k, 2), one per accepted cluster.
        Empty list if no points or no cluster meets ``min_points``.

    Raises:
        ValueError: If points is not a 2-D array with 2 columns.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")

    n = len(pts)
    if n == 0:
        return []

    # --- Build clusters via union-find on eps-neighbourhood pairs ---
    uf = _UnionFind(n)
    tree = cKDTree(pts)
    for i, j in tree.query_pairs(eps):
        uf.union(i, j)

    # --- Collect indices by root ---
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    # --- Filter by min_points and return as arrays ---
    return [pts[idx] for idx in groups.values() if len(idx) >= min_points]
