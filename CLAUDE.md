# CLAUDE.md — EOT Simulator Development Guide

## Project Overview

Extended Object Tracking (EOT) simulation framework for Python.
Estimates the position, velocity, existence probability, and shape (OBB) of multiple objects
from LiDAR / Radar point cloud measurements.

## Development Commands

```bash
uv sync --extra dev       # Set up environment (first time)
uv run pytest             # Run tests
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run mypy src/          # Type check
```

---

## Architecture

The system is split into two independent layers. **Do not conflate them.**

### Layer A — Multi-object Tracking (GGIW-PMBM)

Handles object count estimation, data association, and track lifecycle.

- **Method**: GGIW-PMBM (Gamma Gaussian Inverse Wishart — Poisson Multi-Bernoulli Mixture)
- **Why GGIW**: Closed-form update for extended objects under Gaussian / Inverse-Wishart assumptions
- **Why PMBM**: Strong existence management for multi-target scenarios
- **Extent model**: Ellipsoid (SPD matrix X ∈ R^{2×2}), NOT rectangle — this is intentional

### Layer B — Shape Estimation (Rectangle Fitting)

Post-processing step that converts the ellipse-based track output into an OBB.

- **Why separate**: Rectangular models are incompatible with Gaussian assumptions in the Kalman framework
- **Input**: Point cluster assigned to each confirmed track
- **Output**: (θ, l, w) — orientation, length, width

**Core principle: tracking uses the ellipsoid model; rectangularization is a post-processing step.**

---

## State Definitions

### Single Target State (GGIW)

```
kinematic:  x = [px, py, vx, vy]
extent:     X ∈ R^{2×2}  (SPD matrix)
rate:       γ  (Gamma distribution — expected measurements per frame)
```

### Multi-object State (PMBM)

- **PPP** (Poisson Point Process): undetected / birth targets
- **MBM** (Multi-Bernoulli Mixture): detected track hypotheses

Each Bernoulli component carries:
- existence probability `r`
- a GGIW state

---

## Processing Pipeline

```
sensor point cloud
    ↓
clustering / partition  (measurement cells, 1 cell = 1 target candidate)
    ↓
PMBM update
  - assignment hypotheses to existing tracks
  - birth hypotheses for new tracks
  - missed detection hypotheses
    ↓
single-target update (per hypothesis)
  - kinematic update (Kalman)
  - extent update (Inverse Wishart)
  - rate update (Gamma)
    ↓
hypothesis management (pruning → merging → truncation)
    ↓
extract tracks where existence > threshold  (MAP hypothesis selection)
    ↓
rectangle fitting (per track, using assigned point history)
    ↓
output OBB
```

---

## Rectangle Fitting Algorithm

Operates on the point cluster assigned to a confirmed track.
Using a sliding window of historical points is recommended over single-frame fitting.

```
1. Center:    points_centered = z_i - μ
2. PCA:       eigendecompose covariance → principal axes
3. Rotate:    project into principal axis frame
4. Size:      l = percentile_high(x) - percentile_low(x)
              w = percentile_high(y) - percentile_low(y)
              (percentile preferred over min-max for outlier robustness)
5. Heading:   θ = angle of principal eigenvector
6. Smooth:    apply low-pass or Kalman smoothing on (θ, l, w) across frames
```

---

## Module Layout (Target Structure)

```
src/
├── tracking/
│   ├── pmbm/
│   │   ├── pmbm_filter.py
│   │   ├── bernoulli.py
│   │   └── hypothesis.py
│   ├── ggiw/
│   │   ├── ggiw_state.py
│   │   └── ggiw_update.py
│   ├── measurement/
│   │   ├── clustering.py
│   │   └── partition.py
│   └── shape/
│       ├── rectangle_fitting.py
│       └── smoothing.py
├── simulator.py
├── EKF.py
└── utils.py
```

Existing code in `src/tracker/` contains earlier implementations
(Random Matrix Tracker, EKF-based Rectangle Tracker, GM-PHD).
New development should follow the GGIW-PMBM structure above.

---

## Design Constraints

### Do Not Mix Layers
- The GGIW extent matrix X is an ellipsoid parameter. Do not treat it as a rectangle.
- Rectangle output (θ, l, w) is computed separately in the shape layer.

### Partial Observation Handling
- When points appear on only one side of an object, naïve min-max estimation shrinks l/w.
- Temporal smoothing on (θ, l, w) is mandatory to suppress this artifact.

### Radar-specific Concerns
- Sparse point clouds and specular reflections are expected.
- Tune the Gamma rate parameter γ accordingly.
- Apply a minimum point count threshold before fitting.

### Hypothesis Explosion Prevention
- PMBM hypothesis count grows combinatorially.
- Always apply pruning threshold and max-hypothesis cap after each update step.

---

## Out of Scope (This Iteration)

- Probabilistic rectangular shape model (e.g., RHM / superellipse)
- Gaussian mixture shape representation
- Particle-based multi-object filter
- Camera OBB fusion
