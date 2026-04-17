"""GitHub Pages 用 JSON レポートを生成するスクリプト。

実行方法
--------
    uv run python scripts/generate_report.py            # 全シナリオ (n_mc=30)
    uv run python scripts/generate_report.py --n-mc 5  # 高速確認
    uv run python scripts/generate_report.py --skip-tests

出力先: docs/data/
    test_results.json   - pytest 実行結果
    eval_results.json   - シナリオ評価指標 (per-frame + summary)
    whl_info.json       - wheel ファイル名 (Pyodide sim 用)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# pytest runner
# ---------------------------------------------------------------------------

def run_pytest() -> dict:
    """pytest を実行して結果 dict を返す。"""
    import pytest

    class _Collector:
        def __init__(self):
            self.tests = []
            self.passed = self.failed = self.error = 0
            self._start: float = 0.0
            self.duration: float = 0.0

        def pytest_sessionstart(self, session):
            import time
            self._start = time.monotonic()

        def pytest_sessionfinish(self, session, exitstatus):
            import time
            self.duration = time.monotonic() - self._start

        def pytest_runtest_logreport(self, report):
            if report.when == "setup" and report.failed:
                self.tests.append({
                    "nodeid": report.nodeid,
                    "outcome": "error",
                    "duration": round(getattr(report, "duration", 0.0), 4),
                })
                self.error += 1
            elif report.when == "call":
                self.tests.append({
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "duration": round(report.duration, 4),
                })
                if report.outcome == "passed":
                    self.passed += 1
                else:
                    self.failed += 1

    collector = _Collector()
    exit_code = pytest.main(
        ["tests/", "-q", "--tb=no", "--no-header"],
        plugins=[collector],
    )
    total = collector.passed + collector.failed + collector.error
    return {
        "summary": {
            "total": total,
            "passed": collector.passed,
            "failed": collector.failed,
            "error": collector.error,
            "exit_code": int(exit_code),
            "duration_s": round(collector.duration, 2),
        },
        "tests": collector.tests,
    }


# ---------------------------------------------------------------------------
# 評価ランナー
# ---------------------------------------------------------------------------

def run_evaluation(n_mc: int, seed: int) -> dict:
    """全シナリオを実行して結果 dict を返す。"""
    from src.tracking.evaluation.runner import (
        MissedDetectionRecovery,
        SingleVehicleApproach,
        SingleVehiclePassBy,
        TwoVehicles,
        run_scenario,
    )

    scenarios = {
        "single_approach": SingleVehicleApproach,
        "single_passby":   SingleVehiclePassBy,
        "two_vehicles":    TwoVehicles,
        "missed_recovery": MissedDetectionRecovery,
    }

    output: dict = {}
    for key, cls in scenarios.items():
        print(f"  [{key}] {n_mc} MC runs ...", flush=True)
        result = run_scenario(cls(), n_mc=n_mc, seed=seed)

        def _clean(arr: np.ndarray) -> list:
            return [None if not np.isfinite(v) else round(float(v), 6) for v in arr]

        output[key] = {
            "n_mc": result.n_mc,
            "n_frames": result.n_frames,
            "summary": {
                "gospa":             round(float(result.mean_gospa), 4),
                "pos_rmse":          round(float(result.mean_pos_rmse), 4),
                "vel_rmse":          round(float(result.mean_vel_rmse), 4),
                "yaw_rmse_deg":      round(float(np.rad2deg(result.mean_yaw_rmse)), 2),
                "length_rmse":       round(float(result.mean_length_rmse), 4),
                "width_rmse":        round(float(result.mean_width_rmse), 4),
                "id_switches":       round(float(result.total_id_switches), 2),
                "motp":              round(float(result.mean_motp), 4),
                "mota":              round(float(result.mean_mota), 4),
                "false_alarm_rate":  round(float(result.mean_false_alarm_rate), 4),
                "miss_rate":         round(float(result.mean_miss_rate), 4),
            },
            "by_frame": {
                "gospa":       _clean(result.gospa_by_frame),
                "pos_rmse":    _clean(result.pos_rmse_by_frame),
                "vel_rmse":    _clean(result.vel_rmse_by_frame),
                "yaw_rmse_deg": _clean(np.rad2deg(result.yaw_rmse_by_frame)),
                "length_rmse": _clean(result.length_rmse_by_frame),
                "width_rmse":  _clean(result.width_rmse_by_frame),
                "id_switches": _clean(result.id_switches_by_frame),
            },
        }
        print(f"    GOSPA={result.mean_gospa:.3f}m  MOTA={result.mean_mota:.3f}", flush=True)

    return output


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="GitHub Pages 用 JSON レポートを生成")
    p.add_argument("--n-mc",       type=int, default=30)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="docs/data")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }

    # --- テスト結果 ---
    if not args.skip_tests:
        print("Running pytest ...", flush=True)
        test_data = {**meta, **run_pytest()}
        (out_dir / "test_results.json").write_text(json.dumps(test_data, indent=2))
        s = test_data["summary"]
        print(f"  → {s['passed']}/{s['total']} passed  ({s['duration_s']:.1f}s)", flush=True)

    # --- 評価結果 ---
    print("\nRunning evaluation scenarios ...", flush=True)
    eval_data = {**meta, "n_mc": args.n_mc, "scenarios": run_evaluation(args.n_mc, args.seed)}
    (out_dir / "eval_results.json").write_text(json.dumps(eval_data, indent=2))
    print(f"  → saved to {out_dir}/eval_results.json", flush=True)

    # --- whl 情報 (Pyodide 用) ---
    whl_files = sorted(Path("dist").glob("*.whl"))
    whl_name = whl_files[-1].name if whl_files else None
    (out_dir / "whl_info.json").write_text(json.dumps({"whl_name": whl_name, **meta}))
    print(f"  → whl: {whl_name}", flush=True)


if __name__ == "__main__":
    main()
