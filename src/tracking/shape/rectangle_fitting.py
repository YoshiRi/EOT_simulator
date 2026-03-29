"""PCA-based Oriented Bounding Box (OBB) fitting.

Converts a point cluster assigned to a confirmed track into an OBB described
by (theta, length, width).  This is a **post-processing** step and is kept
entirely separate from the GGIW extent model.

Algorithm (per CLAUDE.md):
    1. Centre  : points_centered = z_i - centroid
    2. PCA     : eigendecompose sample covariance → principal axes
    3. Rotate  : project into principal-axis frame
    4. Size    : percentile-based (robust to outliers / partial observation)
    5. Heading : theta = angle of principal eigenvector (longer axis)

Typical usage::

    from src.tracking.shape.rectangle_fitting import fit_rectangle, OBBResult
    result = fit_rectangle(points)           # (N, 2) array
    print(result.theta, result.length, result.width)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OBBResult:
    """Oriented bounding box described in the sensor / world frame.

    Attributes
    ----------
    theta : float
        Heading angle of the *long* axis in radians, in [-π/2, π/2).
    length : float
        Extent along the principal (long) axis, > 0.
    width : float
        Extent along the secondary (short) axis, > 0.
    center : np.ndarray
        2-D centre of mass of the input points, shape (2,).
    """

    theta: float
    length: float
    width: float
    center: np.ndarray

    # numpy arrays are not hashable; override __eq__ / __hash__ manually
    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, OBBResult):
            return NotImplemented
        return (
            np.isclose(self.theta, other.theta)
            and np.isclose(self.length, other.length)
            and np.isclose(self.width, other.width)
            and np.allclose(self.center, other.center)
        )

    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    def corners(self) -> np.ndarray:
        """Return the 4 corners of the OBB as an (4, 2) array (counter-clockwise).

        The first corner is bottom-left when the box is axis-aligned.
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        half = np.array(
            [
                [-self.length / 2, -self.width / 2],
                [self.length / 2, -self.width / 2],
                [self.length / 2, self.width / 2],
                [-self.length / 2, self.width / 2],
            ]
        )
        return (R @ half.T).T + self.center


# ---------------------------------------------------------------------------
# Minimum point count below which fitting is refused
_MIN_POINTS = 2


def fit_rectangle(
    points: np.ndarray,
    *,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
    min_size: float = 0.1,
) -> OBBResult:
    """Fit an OBB to *points* using PCA.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        2-D point cluster belonging to a single track.
    low_pct, high_pct : float
        Percentiles used for size estimation along each axis.
        Defaults (5, 95) give a robust inlier range while still capturing
        most of the object extent despite partial observation.
    min_size : float
        Minimum allowed length / width in metres.  Prevents degenerate
        boxes when only one or two points are available.

    Returns
    -------
    OBBResult

    Raises
    ------
    ValueError
        If *points* has fewer than ``_MIN_POINTS`` rows or wrong shape.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {pts.shape}")
    if len(pts) < _MIN_POINTS:
        raise ValueError(f"Need at least {_MIN_POINTS} points, got {len(pts)}")

    center = pts.mean(axis=0)
    centered = pts - center

    if len(pts) == 1:
        return OBBResult(theta=0.0, length=min_size, width=min_size, center=center)

    # --- PCA ---
    cov = centered.T @ centered / len(centered)  # (2, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; principal axis is the *last* eigenvector
    principal = eigenvectors[:, -1]  # (2,)

    # Heading angle, normalised to [-π/2, π/2)
    theta = float(np.arctan2(principal[1], principal[0]))
    if theta >= np.pi / 2:
        theta -= np.pi
    elif theta < -np.pi / 2:
        theta += np.pi

    # --- Rotate into principal-axis frame ---
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])          # world → body
    rotated = centered @ R.T                  # (N, 2): col0 = long axis, col1 = short

    # --- Percentile-based sizing ---
    def _size(axis: int) -> float:
        proj = rotated[:, axis]
        return float(
            max(
                np.percentile(proj, high_pct) - np.percentile(proj, low_pct),
                min_size,
            )
        )

    length = _size(0)
    width = _size(1)

    # Ensure length ≥ width (swap axes if needed to keep convention)
    if width > length:
        length, width = width, length
        theta = theta + np.pi / 2
        if theta >= np.pi / 2:
            theta -= np.pi

    return OBBResult(theta=float(theta), length=length, width=width, center=center)
