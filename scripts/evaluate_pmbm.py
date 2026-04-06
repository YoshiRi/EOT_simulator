"""GGIW-PMBM 性能評価スクリプト。

実行方法
--------
::

    uv run python scripts/evaluate_pmbm.py              # 全シナリオ
    uv run python scripts/evaluate_pmbm.py --scenario single_approach
    uv run python scripts/evaluate_pmbm.py --n-mc 100 --out results/
    uv run python scripts/evaluate_pmbm.py --no-plot    # テキスト出力のみ

出力
----
* コンソール: 各シナリオのサマリー表
* PNG: シナリオごとに per-frame 指標のグラフ (--out で保存先指定)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# パッケージルートをパスに追加 (uv run で自動解決される場合は不要)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.evaluation.runner import (
    EvalResult,
    MissedDetectionRecovery,
    SingleVehicleApproach,
    SingleVehiclePassBy,
    TwoVehicles,
    run_scenario,
)

ALL_SCENARIOS = {
    "single_approach":  SingleVehicleApproach,
    "single_passby":    SingleVehiclePassBy,
    "two_vehicles":     TwoVehicles,
    "missed_recovery":  MissedDetectionRecovery,
}


# ---------------------------------------------------------------------------
# グラフ描画
# ---------------------------------------------------------------------------

def plot_result(result: EvalResult, out_dir: Path | None) -> None:
    """per-frame 指標を 6 パネルで描画する。"""
    frames = np.arange(result.n_frames)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"{result.scenario_name}  (MC={result.n_mc})\n"
        f"GOSPA={result.mean_gospa:.2f}m  "
        f"MOTA={result.mean_mota:.3f}  "
        f"MOTP={result.mean_motp:.2f}m  "
        f"IDSW/run={result.total_id_switches:.1f}",
        fontsize=11,
    )

    def _plot(ax, y, ylabel, color="steelblue"):
        valid = np.isfinite(y)
        ax.plot(frames[valid], y[valid], color=color, linewidth=1.8)
        ax.fill_between(frames[valid], 0, y[valid], alpha=0.15, color=color)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, result.n_frames - 1)

    _plot(axes[0, 0], result.gospa_by_frame,          "GOSPA [m]",       "steelblue")
    _plot(axes[0, 1], result.pos_rmse_by_frame,        "Pos RMSE [m]",    "darkorange")
    _plot(axes[1, 0], result.vel_rmse_by_frame,        "Vel RMSE [m/s]",  "seagreen")
    _plot(axes[1, 1], np.rad2deg(result.yaw_rmse_by_frame), "Yaw RMSE [deg]", "crimson")
    _plot(axes[2, 0], result.length_rmse_by_frame,     "Length RMSE [m]", "mediumpurple")
    _plot(axes[2, 1], result.width_rmse_by_frame,      "Width RMSE [m]",  "saddlebrown")

    # ID switch を棒グラフで重ねる (pos パネル)
    ax_idsw = axes[0, 1].twinx()
    ax_idsw.bar(frames, result.id_switches_by_frame, alpha=0.3, color="red", label="ID sw")
    ax_idsw.set_ylabel("ID switches/run", color="red")
    ax_idsw.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{result.scenario_name}.png"
        fig.savefig(path, dpi=120)
        print(f"  → saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# テキスト表出力
# ---------------------------------------------------------------------------

def print_table(results: list[EvalResult]) -> None:
    """全シナリオの比較表を出力する。"""
    cols = [
        ("Scenario",       "scenario_name",        "<18", lambda r: r.scenario_name),
        ("GOSPA[m]",       "mean_gospa",            ">8",  lambda r: f"{r.mean_gospa:.3f}"),
        ("PosRMSE[m]",     "mean_pos_rmse",         ">10", lambda r: f"{r.mean_pos_rmse:.3f}"),
        ("VelRMSE[m/s]",   "mean_vel_rmse",         ">11", lambda r: f"{r.mean_vel_rmse:.3f}"),
        ("YawRMSE[deg]",   "mean_yaw_rmse",         ">12", lambda r: f"{np.rad2deg(r.mean_yaw_rmse):.2f}"),
        ("LenRMSE[m]",     "mean_length_rmse",      ">10", lambda r: f"{r.mean_length_rmse:.3f}"),
        ("WidRMSE[m]",     "mean_width_rmse",       ">10", lambda r: f"{r.mean_width_rmse:.3f}"),
        ("IDSW/run",       "total_id_switches",     ">8",  lambda r: f"{r.total_id_switches:.1f}"),
        ("MOTP[m]",        "mean_motp",             ">8",  lambda r: f"{r.mean_motp:.3f}"),
        ("MOTA",           "mean_mota",             ">6",  lambda r: f"{r.mean_mota:.3f}"),
        ("FA/frame",       "mean_false_alarm_rate", ">8",  lambda r: f"{r.mean_false_alarm_rate:.3f}"),
        ("MissRate",       "mean_miss_rate",        ">8",  lambda r: f"{r.mean_miss_rate:.3f}"),
    ]

    header = "  ".join(f"{name:{fmt}}" for name, _, fmt, _ in cols)
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        row = "  ".join(f"{fn(r):{fmt}}" for _, _, fmt, fn in cols)
        print(row)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GGIW-PMBM 性能評価")
    p.add_argument("--scenario", choices=list(ALL_SCENARIOS.keys()) + ["all"],
                   default="all", help="実行するシナリオ (default: all)")
    p.add_argument("--n-mc", type=int, default=30,
                   help="Monte Carlo 試行回数 (default: 30)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                   help="グラフ保存ディレクトリ (省略時は表示しない)")
    p.add_argument("--no-plot", action="store_true",
                   help="グラフ出力をスキップ")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out) if args.out else None

    # 実行するシナリオを選択
    if args.scenario == "all":
        scenario_keys = list(ALL_SCENARIOS.keys())
    else:
        scenario_keys = [args.scenario]

    results: list[EvalResult] = []
    for key in scenario_keys:
        scenario = ALL_SCENARIOS[key]()
        print(f"\n[{key}] {args.n_mc} MC runs ...")
        result = run_scenario(scenario, n_mc=args.n_mc, seed=args.seed, verbose=True)
        print(result.summary())
        results.append(result)

        if not args.no_plot:
            plot_result(result, out_dir)

    if len(results) > 1:
        print_table(results)


if __name__ == "__main__":
    main()
