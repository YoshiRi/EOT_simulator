"""Pyodide インタラクティブ Sim 用シミュレーション関数。

sim.html から micropip でインストールされたパッケージ経由で呼び出される。
通常の pytest でもそのままテスト可能。
"""

from __future__ import annotations

import io
import json
import math
import random

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.simulator import LidarSimulator, VehicleSimulator
from src.tracking.evaluation.association import associate
from src.tracking.evaluation.metrics import gospa as gospa_metric
from src.tracking.measurement.partition import partition
from src.tracking.pmbm.pmbm_filter import BirthModel, PMBMFilter
from src.tracking.shape.rectangle_fitting import OBBResult, fit_rectangle
from src.tracking.shape.smoothing import ExponentialSmoother


# ---------------------------------------------------------------------------
# シナリオ定義
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "single_approach": {
        "vehicles": lambda: [VehicleSimulator(25.0, 0.0, math.pi, 3.0, 50 / 3.6, 2.0, 4.0)],
        "n_frames": 20, "dt": 0.5, "angle_res_deg": 2.0, "miss_frames": [],
    },
    "single_passby": {
        "vehicles": lambda: [VehicleSimulator(-20.0, 8.0, 0.0, 5.0, 50 / 3.6, 2.0, 4.0)],
        "n_frames": 20, "dt": 0.5, "angle_res_deg": 2.0, "miss_frames": [],
    },
    "two_vehicles": {
        "vehicles": lambda: [
            VehicleSimulator(20.0, 0.0,  math.pi,       3.0, 50 / 3.6, 2.0, 4.0),
            VehicleSimulator(0.0,  15.0, math.pi * 1.5, 2.0, 50 / 3.6, 2.0, 4.0),
        ],
        "n_frames": 20, "dt": 0.5, "angle_res_deg": 2.0, "miss_frames": [],
    },
    "missed_recovery": {
        "vehicles": lambda: [VehicleSimulator(20.0, 0.0, math.pi, 2.0, 50 / 3.6, 2.0, 4.0)],
        "n_frames": 25, "dt": 0.5, "angle_res_deg": 2.0, "miss_frames": list(range(8, 12)),
    },
}


# ---------------------------------------------------------------------------
# 1 MC ラン
# ---------------------------------------------------------------------------

def _ggiw_to_ellipse(state) -> dict:
    """GGIW extent matrix → 描画用楕円パラメータ (半軸長・角度)。"""
    X = state.extent_mean  # 2×2 SPD
    vals, vecs = np.linalg.eigh(X)
    # eigh は昇順で返す → vals[1] が長軸
    return {
        "x":         float(state.m[0]),
        "y":         float(state.m[1]),
        "vx":        float(state.m[2]),
        "vy":        float(state.m[3]),
        "ext_a":     float(np.sqrt(max(vals[1], 1e-6))),
        "ext_b":     float(np.sqrt(max(vals[0], 1e-6))),
        "ext_theta": float(np.arctan2(vecs[1, 1], vecs[0, 1])),
    }


def _obb_to_dict(obb: OBBResult) -> dict:
    """OBBResult → JSON シリアライズ可能な dict。"""
    return {
        "cx":      float(obb.center[0]),
        "cy":      float(obb.center[1]),
        "theta":   float(obb.theta),
        "l":       float(obb.length),
        "w":       float(obb.width),
        "corners": obb.corners().tolist(),
    }


def _match_clusters_to_estimates(cells, ests) -> list:
    """クラスタと推定をセントロイド距離で最近傍マッチング (ゲート 6.0 m)。

    Returns:
        ests と同じ長さのリスト。対応クラスタなしの場合は None。
    """
    if not ests or not cells:
        return [None] * len(ests)

    gate = 6.0
    est_pos = np.array([s.position for s in ests])          # (n_est, 2)
    cell_ctr = np.array([c.centroid for c in cells])         # (n_cell, 2)

    # コスト行列 (n_est, n_cell)
    diff = est_pos[:, None, :] - cell_ctr[None, :, :]        # (n_est, n_cell, 2)
    cost = np.linalg.norm(diff, axis=2)

    gated = np.where(cost < gate, cost, 1e9)
    row_ind, col_ind = linear_sum_assignment(gated)

    matched: list = [None] * len(ests)
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] < gate:
            matched[ri] = cells[ci]
    return matched


class LocalTracker:
    """フレーム間でローカル ID と ExponentialSmoother を管理するトラッカー。

    PMBM はトラックに永続 ID を付与しないため、推定位置のハンガリアン法マッチで
    フレーム間同一性を近似し、スムーザ状態を継続する。
    """

    def __init__(self, gate_dist: float = 5.0, smoother_alpha: float = 0.4) -> None:
        self._gate_dist = gate_dist
        self._smoother_alpha = smoother_alpha
        self._smoothers: dict[int, ExponentialSmoother] = {}
        self._positions: dict[int, np.ndarray] = {}
        self._next_id = 0

    def update(self, ests: list, matched_cells: list) -> list:
        """推定リストとマッチ済みクラスタからスムーズ OBB を計算する。

        Returns:
            ests と同じ長さのリスト (OBBResult | None)。
        """
        if not ests:
            self._positions = {}
            return []

        curr_pos = np.array([s.position for s in ests])  # (n, 2)

        # 前フレーム推定との Hungarian マッチング
        est_to_id: dict[int, int] = {}
        if self._positions:
            prev_ids = list(self._positions.keys())
            prev_pos = np.array([self._positions[pid] for pid in prev_ids])

            n_prev, n_curr = len(prev_ids), len(ests)
            cost = np.full((n_prev, n_curr), 1e9)
            for i, pp in enumerate(prev_pos):
                dists = np.linalg.norm(curr_pos - pp, axis=1)
                mask = dists < self._gate_dist
                cost[i, mask] = dists[mask]

            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] < self._gate_dist:
                    est_to_id[ci] = prev_ids[ri]

        # 未マッチ推定に新規 ID を割り当て
        for j in range(len(ests)):
            if j not in est_to_id:
                est_to_id[j] = self._next_id
                self._next_id += 1

        # 消滅したトラックのスムーザを削除
        active_ids = set(est_to_id.values())
        for tid in list(self._smoothers.keys()):
            if tid not in active_ids:
                del self._smoothers[tid]

        # OBB フィッティング + スムージング
        result: list = []
        new_positions: dict[int, np.ndarray] = {}
        for j, (est, cell) in enumerate(zip(ests, matched_cells)):
            tid = est_to_id[j]
            new_positions[tid] = curr_pos[j]
            if tid not in self._smoothers:
                self._smoothers[tid] = ExponentialSmoother(alpha=self._smoother_alpha)

            if cell is not None and len(cell.points) >= 2:
                raw_obb = fit_rectangle(cell.points)
                smoothed = self._smoothers[tid].update(raw_obb)
                result.append(smoothed)
            else:
                result.append(None)

        self._positions = new_positions
        return result


def run_once(
    sc_cfg: dict,
    filt: PMBMFilter,
    seed: int = 0,
    record_frames: bool = False,
) -> dict:
    """1 MC ランを実行して per-frame 指標を返す。

    Args:
        sc_cfg:        SCENARIOS の値 dict。
        filt:          初期化済み PMBMFilter。
        seed:          Python の random モジュールへのシード。
        record_frames: True のとき GT/obs/est の 2D データを snapshots に記録。

    Returns:
        {"gospa": [...], "pos_rmse": [...], "id_switches": [...],
         "snapshots": [...] or None}
    """
    random.seed(seed)
    lidar = LidarSimulator(range_noise=0.05)
    vehicles = sc_cfg["vehicles"]()
    dt = sc_cfg["dt"]
    miss_frames = set(sc_cfg["miss_frames"])

    gospa_vals: list[float] = []
    pos_errs: list[float] = []
    id_sw_vals: list[float] = []
    prev_asgn: dict[int, int] = {}
    snapshots: list[dict] | None = [] if record_frames else None
    tracker = LocalTracker() if record_frames else None

    for fi in range(sc_cfg["n_frames"]):
        for v in vehicles:
            v.update(dt=dt, a=0.0, omega=0.0)

        gt_pos = np.array([[v.x, v.y] for v in vehicles])

        if fi not in miss_frames:
            ox, oy = lidar.get_observation_points(vehicles, np.deg2rad(sc_cfg["angle_res_deg"]))
        else:
            ox, oy = [], []

        filt.predict(dt=dt)
        if ox:
            filt.update_from_points(np.column_stack([ox, oy]), eps=2.0)
        else:
            filt.update([])

        ests = filt.extract_estimates(existence_threshold=0.5)
        est_pos = np.array([s.position for s in ests]) if ests else np.empty((0, 2))

        g = gospa_metric(gt_pos, est_pos, c=2.0, p=2, alpha=2.0)
        gospa_vals.append(g["gospa"])

        asgn = associate(gt_pos, est_pos, gate_distance=6.0)
        pos_errs.append(float(np.mean(asgn.distances)) if asgn.distances else float("nan"))

        curr_asgn = {ei: gi for gi, ei in zip(asgn.matched_gt, asgn.matched_est)}
        id_sw_vals.append(float(sum(
            1 for ei, gi in curr_asgn.items() if prev_asgn.get(ei, gi) != gi
        )))
        prev_asgn = curr_asgn

        if record_frames:
            assert tracker is not None
            # クラスタ取得 (OBB フィッティング用; フィルタ内部の partition と独立)
            if ox:
                pts = np.column_stack([ox, oy])
                cells = partition(pts, eps=2.0)
            else:
                cells = []

            matched_cells = _match_clusters_to_estimates(cells, ests)
            obbs = tracker.update(ests, matched_cells)

            clusters_data = [
                {
                    "centroid": c.centroid.tolist(),
                    "points":   c.points.tolist(),
                }
                for c in cells
            ]

            snapshots.append({
                "fi":       fi,
                "missed":   fi in miss_frames,
                "gt":       [{"x": float(v.x), "y": float(v.y),
                               "yaw": float(v.yaw), "l": float(v.L), "w": float(v.W)}
                              for v in vehicles],
                "obs":      [[float(x), float(y)] for x, y in zip(ox, oy)],
                "clusters": clusters_data,
                "est":      [
                    {**_ggiw_to_ellipse(s), "obb": _obb_to_dict(o) if o is not None else None}
                    for s, o in zip(ests, obbs)
                ],
            })

    return {
        "gospa":     gospa_vals,
        "pos_rmse":  pos_errs,
        "id_switches": id_sw_vals,
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# MC 集計 + プロット
# ---------------------------------------------------------------------------

def run_simulation_json(
    scenario_key: str,
    n_mc: int,
    p_survival: float,
    p_detection: float,
    birth_weight: float,
    prune_log_threshold: float,
) -> str:
    """MC シミュレーションを実行して JSON 文字列を返す。

    Returns:
        JSON string: {"summary": {...}, "svg": "<svg>..."}
    """
    sc_cfg = SCENARIOS[scenario_key]
    n_frames = sc_cfg["n_frames"]

    all_gospa = np.zeros((n_mc, n_frames))
    all_pos   = np.zeros((n_mc, n_frames))
    all_idsw  = np.zeros((n_mc, n_frames))

    frame_data: dict | None = None

    for mc_i in range(n_mc):
        filt = PMBMFilter(
            p_survival=p_survival,
            p_detection=p_detection,
            birth_model=BirthModel(birth_weight=birth_weight, vel_var=1e2),
            max_hypotheses=50,
            prune_log_threshold=prune_log_threshold,
        )
        # MC run 0 だけフレームデータを記録する
        r = run_once(sc_cfg, filt, seed=mc_i * 17, record_frames=(mc_i == 0))
        all_gospa[mc_i] = np.where(np.isfinite(r["gospa"]),    r["gospa"],    np.nan)
        all_pos[mc_i]   = np.where(np.isfinite(r["pos_rmse"]), r["pos_rmse"], np.nan)
        all_idsw[mc_i]  = r["id_switches"]
        if mc_i == 0:
            frame_data = {"sensor_origin": [0, 0], "frames": r["snapshots"]}

    gospa_mean = np.nanmean(all_gospa, axis=0)
    pos_mean   = np.nanmean(all_pos,   axis=0)
    idsw_mean  = np.nanmean(all_idsw,  axis=0)

    summary = {
        "gospa":       float(np.nanmean(gospa_mean)),
        "pos_rmse":    float(np.nanmean(pos_mean)),
        "id_switches": float(np.nansum(idsw_mean)),
    }

    svg = _make_plot(scenario_key, n_mc, p_survival, p_detection, birth_weight,
                     n_frames, gospa_mean, pos_mean, idsw_mean)
    return json.dumps({"summary": summary, "svg": svg, "frame_data": frame_data})


def _make_plot(
    scenario_key: str, n_mc: int,
    p_survival: float, p_detection: float, birth_weight: float,
    n_frames: int,
    gospa_mean: np.ndarray, pos_mean: np.ndarray, idsw_mean: np.ndarray,
) -> str:
    frames = np.arange(n_frames)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.patch.set_facecolor("#1a1d27")

    for ax, title, data, color in zip(
        axes,
        ["GOSPA [m]", "Pos RMSE [m]", "ID Switches/run"],
        [gospa_mean, pos_mean, idsw_mean],
        ["#60a5fa", "#fb923c", "#facc15"],
    ):
        ax.set_facecolor("#252836")
        valid = np.isfinite(data)
        ax.plot(frames[valid], data[valid], color=color, linewidth=2)
        ax.fill_between(frames[valid], 0, data[valid], alpha=0.2, color=color)
        ax.set_title(title, color="#e2e8f0", fontsize=11)
        ax.set_xlabel("Frame", color="#8892a4", fontsize=9)
        ax.tick_params(colors="#8892a4", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3347")
        ax.grid(True, alpha=0.2, color="#2e3347")

    fig.suptitle(
        f"{scenario_key}  (MC={n_mc}  pS={p_survival:.3f}  pD={p_detection:.3f}  bw={birth_weight:.2f})",
        color="#e2e8f0", fontsize=10, y=1.02,
    )
    plt.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
