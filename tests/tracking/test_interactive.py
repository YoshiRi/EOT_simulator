"""Smoke tests for the Pyodide interactive simulation module.

These tests ensure that the simulation code used in sim.html:
  - Imports without error
  - Accepts correct constructor arguments (guards against e.g. seed= on LidarSimulator)
  - Runs end-to-end for every scenario without crashing
  - Returns the expected data structure

Runs are kept to n_mc=1 and n_frames=3 for speed.
"""

import json
import math

import numpy as np
import pytest

from src.tracking.evaluation.interactive import (
    SCENARIOS,
    run_once,
    run_simulation_json,
)
from src.tracking.pmbm.pmbm_filter import BirthModel, PMBMFilter
from src.simulator import LidarSimulator, VehicleSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filter() -> PMBMFilter:
    return PMBMFilter(
        birth_model=BirthModel(birth_weight=0.2, vel_var=1e2),
        max_hypotheses=20,
        prune_log_threshold=-15.0,
    )


def _short_cfg(scenario_key: str) -> dict:
    """Return a scenario config with n_frames reduced for fast tests."""
    cfg = dict(SCENARIOS[scenario_key])
    cfg["n_frames"] = 3
    return cfg


# ---------------------------------------------------------------------------
# Simulator API smoke tests (guard against wrong constructor kwargs)
# ---------------------------------------------------------------------------

class TestSimulatorAPI:
    def test_lidar_simulator_no_seed_arg(self):
        """LidarSimulator takes only range_noise; passing seed= must fail."""
        with pytest.raises(TypeError):
            LidarSimulator(range_noise=0.05, seed=0)

    def test_lidar_simulator_creates_ok(self):
        lidar = LidarSimulator(range_noise=0.05)
        assert lidar.range_noise == pytest.approx(0.05)

    def test_vehicle_simulator_positional_args(self):
        """VehicleSimulator uses positional args (x, y, yaw, v, max_v, w, L)."""
        v = VehicleSimulator(10.0, 0.0, math.pi, 3.0, 50 / 3.6, 2.0, 4.0)
        assert v.x == pytest.approx(10.0)
        assert v.W == pytest.approx(2.0)
        assert v.L == pytest.approx(4.0)

    def test_vehicle_simulator_no_keyword_args(self):
        """Keyword form must raise TypeError — guard against accidental yaw=... usage."""
        with pytest.raises(TypeError):
            VehicleSimulator(x=10.0, y=0.0, yaw=math.pi, v=3.0, max_v=50/3.6, w=2.0, L=4.0)


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    @pytest.mark.parametrize("key", list(SCENARIOS.keys()))
    def test_runs_without_error(self, key):
        result = run_once(_short_cfg(key), _make_filter(), seed=0)
        assert "gospa" in result
        assert "pos_rmse" in result
        assert "id_switches" in result

    @pytest.mark.parametrize("key", list(SCENARIOS.keys()))
    def test_output_length_matches_n_frames(self, key):
        cfg = _short_cfg(key)
        result = run_once(cfg, _make_filter(), seed=0)
        assert len(result["gospa"])       == cfg["n_frames"]
        assert len(result["pos_rmse"])    == cfg["n_frames"]
        assert len(result["id_switches"]) == cfg["n_frames"]

    def test_gospa_values_non_negative(self):
        cfg = _short_cfg("single_approach")
        result = run_once(cfg, _make_filter(), seed=42)
        finite = [v for v in result["gospa"] if np.isfinite(v)]
        assert all(v >= 0 for v in finite)

    def test_seed_affects_lidar_noise(self):
        """random.seed() should change LidarSimulator output (range_noise > 0)."""
        cfg = _short_cfg("single_approach")
        vehicles0 = cfg["vehicles"]()
        vehicles1 = cfg["vehicles"]()
        lidar = LidarSimulator(range_noise=0.1)  # large noise to ensure observable difference

        import random
        random.seed(0)
        ox0, oy0 = lidar.get_observation_points(vehicles0, np.deg2rad(2.0))

        random.seed(99)
        ox1, oy1 = lidar.get_observation_points(vehicles1, np.deg2rad(2.0))

        # Different seeds should produce different range-noised observations
        assert ox0 != ox1 or oy0 != oy1

    def test_missed_recovery_correct_frame_count(self):
        """run_once on missed_recovery returns n_frames entries even through miss window."""
        cfg = dict(SCENARIOS["missed_recovery"])
        cfg["n_frames"] = 13  # covers miss window 8-11
        result = run_once(cfg, _make_filter(), seed=0)
        assert len(result["gospa"])       == 13
        assert len(result["pos_rmse"])    == 13
        assert len(result["id_switches"]) == 13

    def test_missed_recovery_miss_frames_do_not_crash(self):
        """All miss_frames run without error; filter continues via prediction only."""
        cfg = dict(SCENARIOS["missed_recovery"])  # miss_frames = [8, 9, 10, 11]
        result = run_once(cfg, _make_filter(), seed=0)
        # GOSPA must be a finite float for every frame (filter always produces a value)
        assert all(np.isfinite(v) for v in result["gospa"])


# ---------------------------------------------------------------------------
# run_simulation_json
# ---------------------------------------------------------------------------

class TestRunSimulationJson:
    def _run(self, scenario_key="single_approach", n_mc=1) -> dict:
        raw = run_simulation_json(
            scenario_key=scenario_key,
            n_mc=n_mc,
            p_survival=0.99,
            p_detection=0.90,
            birth_weight=0.20,
            prune_log_threshold=-15.0,
        )
        return json.loads(raw)

    def test_returns_valid_json(self):
        raw = run_simulation_json("single_approach", 1, 0.99, 0.9, 0.2, -15.0)
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_summary_keys_present(self):
        data = self._run()
        assert {"gospa", "pos_rmse", "id_switches"} <= data["summary"].keys()

    def test_summary_values_finite(self):
        data = self._run()
        s = data["summary"]
        assert np.isfinite(s["gospa"])
        assert np.isfinite(s["id_switches"])

    def test_svg_present_and_non_empty(self):
        data = self._run()
        assert "svg" in data
        assert data["svg"].strip().startswith("<svg") or "svg" in data["svg"]

    @pytest.mark.parametrize("key", list(SCENARIOS.keys()))
    def test_all_scenarios_run(self, key):
        data = self._run(scenario_key=key, n_mc=1)
        assert "summary" in data
        assert "svg" in data

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError):
            run_simulation_json("nonexistent", 1, 0.99, 0.9, 0.2, -15.0)
