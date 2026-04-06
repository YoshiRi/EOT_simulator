"""Monte Carlo 評価ランナー。

シナリオ定義 → フレームループ → 指標集計 の流れを担う。

使い方
------
::

    scenario = SingleVehicleApproach()
    result = run_scenario(scenario, n_mc=50)
    print(result.summary())
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.simulator import LidarSimulator, VehicleSimulator
from src.tracking.ggiw.ggiw_state import GGIWState
from src.tracking.measurement.partition import partition
from src.tracking.pmbm.pmbm_filter import BirthModel, PMBMFilter
from src.tracking.shape.rectangle_fitting import OBBResult, fit_rectangle
from src.tracking.shape.smoothing import ExponentialSmoother

from .association import FrameAssignment, associate
from .metrics import (
    count_id_switches,
    gospa,
    motp,
    mota,
    position_error,
    shape_error,
    summarise,
    velocity_error,
    yaw_error,
)


# ---------------------------------------------------------------------------
# GT フレーム情報
# ---------------------------------------------------------------------------

@dataclass
class GTTarget:
    """1 フレーム 1 目標の地上真値。"""
    x: float
    y: float
    vx: float
    vy: float
    yaw: float      # [rad]
    length: float   # [m]
    width: float    # [m]

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


# ---------------------------------------------------------------------------
# フレームごとの評価結果
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    """1 フレーム分の指標。"""
    frame_idx: int
    n_gt: int
    n_est: int
    n_tp: int
    n_fp: int
    n_fn: int
    id_switches: int

    gospa_total: float
    gospa_loc: float
    gospa_missed: float
    gospa_false: float

    pos_errors: list[float]       # マッチした対の位置誤差
    vel_errors: list[float]       # 速度誤差
    yaw_errors: list[float]       # 方位角誤差
    length_errors: list[float]    # 長さ誤差
    width_errors: list[float]     # 幅誤差


# ---------------------------------------------------------------------------
# シナリオプロトコル
# ---------------------------------------------------------------------------

class Scenario(Protocol):
    """評価シナリオのインターフェース。"""

    name: str
    n_frames: int
    dt: float
    angle_res_deg: float
    eps_cluster: float
    gate_distance: float

    def make_vehicles(self) -> list[VehicleSimulator]: ...
    def make_filter(self) -> PMBMFilter: ...


# ---------------------------------------------------------------------------
# 組み込みシナリオ
# ---------------------------------------------------------------------------

@dataclass
class SingleVehicleApproach:
    """1 台の車両が正面から近づくシナリオ。"""
    name: str = "single_approach"
    n_frames: int = 20
    dt: float = 0.5
    angle_res_deg: float = 2.0
    eps_cluster: float = 2.0
    gate_distance: float = 6.0
    start_x: float = 25.0
    speed: float = 3.0

    def make_vehicles(self) -> list[VehicleSimulator]:
        return [VehicleSimulator(
            self.start_x, 0.0,
            math.pi,          # i_yaw: 正面向き (-x 方向)
            self.speed,       # i_v
            50.0 / 3.6,       # max_v
            2.0, 4.0,         # w, L
        )]

    def make_filter(self) -> PMBMFilter:
        return PMBMFilter(
            birth_model=BirthModel(birth_weight=0.2, vel_var=1e2),
            max_hypotheses=50,
            prune_log_threshold=-15.0,
        )


@dataclass
class SingleVehiclePassBy:
    """1 台が横を通り過ぎるシナリオ。"""
    name: str = "single_passby"
    n_frames: int = 20
    dt: float = 0.5
    angle_res_deg: float = 2.0
    eps_cluster: float = 2.0
    gate_distance: float = 6.0

    def make_vehicles(self) -> list[VehicleSimulator]:
        return [VehicleSimulator(
            -20.0, 8.0,
            0.0,              # i_yaw
            5.0,              # i_v
            50.0 / 3.6,       # max_v
            2.0, 4.0,         # w, L
        )]

    def make_filter(self) -> PMBMFilter:
        return PMBMFilter(
            birth_model=BirthModel(birth_weight=0.2, vel_var=1e2),
            max_hypotheses=50,
            prune_log_threshold=-15.0,
        )


@dataclass
class TwoVehicles:
    """2 台の車両が同時に存在するシナリオ。"""
    name: str = "two_vehicles"
    n_frames: int = 20
    dt: float = 0.5
    angle_res_deg: float = 2.0
    eps_cluster: float = 2.0
    gate_distance: float = 6.0

    def make_vehicles(self) -> list[VehicleSimulator]:
        return [
            VehicleSimulator(20.0, 0.0,  math.pi,       3.0, 50/3.6, 2.0, 4.0),
            VehicleSimulator(0.0,  15.0, math.pi * 1.5, 2.0, 50/3.6, 2.0, 4.0),
        ]

    def make_filter(self) -> PMBMFilter:
        return PMBMFilter(
            birth_model=BirthModel(birth_weight=0.2, vel_var=1e2),
            max_hypotheses=100,
            prune_log_threshold=-15.0,
        )


@dataclass
class MissedDetectionRecovery:
    """数フレーム観測が途切れた後に再検出するシナリオ。"""
    name: str = "missed_recovery"
    n_frames: int = 25
    dt: float = 0.5
    angle_res_deg: float = 2.0
    eps_cluster: float = 2.0
    gate_distance: float = 6.0
    miss_start: int = 8    # 欠損開始フレーム
    miss_end: int = 12     # 欠損終了フレーム

    def make_vehicles(self) -> list[VehicleSimulator]:
        return [VehicleSimulator(20.0, 0.0, math.pi, 2.0, 50/3.6, 2.0, 4.0)]

    def make_filter(self) -> PMBMFilter:
        return PMBMFilter(
            birth_model=BirthModel(birth_weight=0.2, vel_var=1e2),
            max_hypotheses=50,
            prune_log_threshold=-15.0,
        )


# ---------------------------------------------------------------------------
# 1 回の MC ランを実行
# ---------------------------------------------------------------------------

def _run_once(
    scenario,
    lidar: LidarSimulator,
    rng: np.random.Generator,
) -> list[FrameMetrics]:
    """1 MC run を実行して per-frame 指標リストを返す。"""
    vehicles = scenario.make_vehicles()
    filt = scenario.make_filter()
    smoothers: dict[int, ExponentialSmoother] = {}

    prev_assignment: dict[int, int] = {}  # est_local_idx -> gt_idx
    frame_metrics: list[FrameMetrics] = []

    for frame_idx in range(scenario.n_frames):
        # --- シミュレーション: GT 位置を進める ---
        for v in vehicles:
            v.update(dt=scenario.dt, a=0.0, omega=0.0)

        # GT 情報収集
        gt_targets: list[GTTarget] = []
        for v in vehicles:
            spd = v.v
            gt_targets.append(GTTarget(
                x=v.x, y=v.y,
                vx=spd * math.cos(v.yaw),
                vy=spd * math.sin(v.yaw),
                yaw=v.yaw,
                length=v.L,
                width=v.W,
            ))

        # 欠損フレーム (MissedDetectionRecovery シナリオ用)
        is_missed = (
            hasattr(scenario, "miss_start")
            and scenario.miss_start <= frame_idx < scenario.miss_end
        )

        # --- LiDAR スキャン ---
        if not is_missed:
            ox, oy = lidar.get_observation_points(vehicles, np.deg2rad(scenario.angle_res_deg))
        else:
            ox, oy = [], []

        # --- フィルタ更新 ---
        filt.predict(dt=scenario.dt)
        if ox:
            pts = np.column_stack([ox, oy])
            filt.update_from_points(pts, eps=scenario.eps_cluster)
        else:
            filt.update([])

        # --- 推定値抽出 ---
        ggiw_states: list[GGIWState] = filt.extract_estimates(existence_threshold=0.5)

        # 形状推定 (rectangle fitting)
        est_obbs: list[OBBResult | None] = []
        if ox and ggiw_states:
            pts_all = np.column_stack([ox, oy])
            cells = partition(pts_all, eps=scenario.eps_cluster)
            for idx, state in enumerate(ggiw_states):
                # 最近傍セルを選択
                if cells:
                    dists = [np.linalg.norm(c.centroid - state.position) for c in cells]
                    best = cells[int(np.argmin(dists))]
                    if len(best.points) >= 2:
                        obb = fit_rectangle(best.points)
                        sm = smoothers.setdefault(idx, ExponentialSmoother(alpha=0.3))
                        est_obbs.append(sm.update(obb))
                    else:
                        est_obbs.append(None)
                else:
                    est_obbs.append(None)
        else:
            est_obbs = [None] * len(ggiw_states)

        # --- GT / 推定 マッチング ---
        gt_pos = np.array([g.position for g in gt_targets]) if gt_targets else np.empty((0, 2))
        est_pos = np.array([s.position for s in ggiw_states]) if ggiw_states else np.empty((0, 2))

        asgn: FrameAssignment = associate(gt_pos, est_pos, gate_distance=scenario.gate_distance)

        # ID Switch カウント
        curr_assignment = {
            est_local: gt_local
            for gt_local, est_local in zip(asgn.matched_gt, asgn.matched_est)
        }
        id_sw = count_id_switches(prev_assignment, curr_assignment)
        prev_assignment = curr_assignment

        # GOSPA
        gsp = gospa(gt_pos, est_pos)

        # 各ペアの誤差
        pos_errs, vel_errs, yaw_errs, len_errs, wid_errs = [], [], [], [], []
        for gt_i, est_i, dist in zip(asgn.matched_gt, asgn.matched_est, asgn.distances):
            g = gt_targets[gt_i]
            s = ggiw_states[est_i]
            pos_errs.append(dist)
            vel_errs.append(velocity_error(s.velocity, g.velocity))

            obb = est_obbs[est_i] if est_i < len(est_obbs) else None
            if obb is not None:
                yaw_errs.append(yaw_error(obb.theta, g.yaw))
                le, we = shape_error(obb.length, obb.width, g.length, g.width)
                len_errs.append(le)
                wid_errs.append(we)

        n_tp = len(asgn.matched_gt)
        n_fp = len(asgn.false_alarms)
        n_fn = len(asgn.missed_gt)

        frame_metrics.append(FrameMetrics(
            frame_idx=frame_idx,
            n_gt=len(gt_targets),
            n_est=len(ggiw_states),
            n_tp=n_tp,
            n_fp=n_fp,
            n_fn=n_fn,
            id_switches=id_sw,
            gospa_total=gsp["gospa"],
            gospa_loc=gsp["loc"],
            gospa_missed=gsp["missed"],
            gospa_false=gsp["false"],
            pos_errors=pos_errs,
            vel_errors=vel_errs,
            yaw_errors=yaw_errs,
            length_errors=len_errs,
            width_errors=wid_errs,
        ))

    return frame_metrics


# ---------------------------------------------------------------------------
# Monte Carlo 集計
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """MC 評価の集計結果。"""
    scenario_name: str
    n_mc: int
    n_frames: int

    # per-frame 平均 (フレーム軸でまず平均、次に MC で平均)
    gospa_by_frame: np.ndarray       # (n_frames,) GOSPA 平均
    pos_rmse_by_frame: np.ndarray    # (n_frames,) 位置 RMSE
    vel_rmse_by_frame: np.ndarray
    yaw_rmse_by_frame: np.ndarray
    length_rmse_by_frame: np.ndarray
    width_rmse_by_frame: np.ndarray
    id_switches_by_frame: np.ndarray  # (n_frames,) ID sw 合計

    # スカラー集計
    mean_gospa: float
    mean_pos_rmse: float
    mean_vel_rmse: float
    mean_yaw_rmse: float
    mean_length_rmse: float
    mean_width_rmse: float
    total_id_switches: float
    mean_motp: float
    mean_mota: float
    mean_false_alarm_rate: float  # FP/frame
    mean_miss_rate: float         # FN/GT per frame

    def summary(self) -> str:
        lines = [
            f"=== {self.scenario_name} ({self.n_mc} MC runs) ===",
            f"  GOSPA            : {self.mean_gospa:.3f} m",
            f"  Pos RMSE         : {self.mean_pos_rmse:.3f} m",
            f"  Vel RMSE         : {self.mean_vel_rmse:.3f} m/s",
            f"  Yaw RMSE         : {np.rad2deg(self.mean_yaw_rmse):.2f} deg",
            f"  Length RMSE      : {self.mean_length_rmse:.3f} m",
            f"  Width RMSE       : {self.mean_width_rmse:.3f} m",
            f"  ID Switches/run  : {self.total_id_switches:.1f}",
            f"  MOTP             : {self.mean_motp:.3f} m",
            f"  MOTA             : {self.mean_mota:.3f}",
            f"  False alarm/frame: {self.mean_false_alarm_rate:.3f}",
            f"  Miss rate        : {self.mean_miss_rate:.3f}",
        ]
        return "\n".join(lines)


def run_scenario(
    scenario,
    n_mc: int = 30,
    seed: int = 0,
    range_noise: float = 0.05,
    verbose: bool = True,
) -> EvalResult:
    """Monte Carlo 評価を実行して EvalResult を返す。"""
    rng = np.random.default_rng(seed)
    lidar = LidarSimulator(range_noise=range_noise)

    # (n_mc, n_frames) 分の FrameMetrics を収集
    all_runs: list[list[FrameMetrics]] = []
    for mc_i in range(n_mc):
        run = _run_once(scenario, lidar, rng)
        all_runs.append(run)
        if verbose and (mc_i + 1) % 10 == 0:
            print(f"  {mc_i + 1}/{n_mc} runs done")

    n_frames = scenario.n_frames

    # per-frame 集計 (MC 平均)
    def _frame_mean(extract_fn, default=float("nan")):
        """各フレーム × 各 run から値を取り出して MC 平均する。"""
        result = np.full(n_frames, float("nan"))
        for f in range(n_frames):
            vals = []
            for run in all_runs:
                v = extract_fn(run[f])
                if isinstance(v, list):
                    vals.extend(v)
                elif np.isfinite(v):
                    vals.append(v)
            result[f] = float(np.nanmean(vals)) if vals else default
        return result

    def _frame_rmse(extract_list_fn):
        result = np.full(n_frames, float("nan"))
        for f in range(n_frames):
            vals = []
            for run in all_runs:
                vals.extend(extract_list_fn(run[f]))
            arr = np.array([v for v in vals if np.isfinite(v)])
            result[f] = float(np.sqrt((arr ** 2).mean())) if len(arr) > 0 else float("nan")
        return result

    gospa_by_frame    = _frame_mean(lambda m: m.gospa_total)
    pos_rmse_by_frame  = _frame_rmse(lambda m: m.pos_errors)
    vel_rmse_by_frame  = _frame_rmse(lambda m: m.vel_errors)
    yaw_rmse_by_frame  = _frame_rmse(lambda m: m.yaw_errors)
    len_rmse_by_frame  = _frame_rmse(lambda m: m.length_errors)
    wid_rmse_by_frame  = _frame_rmse(lambda m: m.width_errors)
    idsw_by_frame      = _frame_mean(lambda m: float(m.id_switches), default=0.0)

    # スカラー
    def _flat(extract_list_fn):
        vals = []
        for run in all_runs:
            for fm in run:
                vals.extend(extract_list_fn(fm))
        return [v for v in vals if np.isfinite(v)]

    def _rmse(vals):
        if not vals:
            return float("nan")
        return float(np.sqrt(np.mean(np.array(vals) ** 2)))

    pos_rmse   = _rmse(_flat(lambda m: m.pos_errors))
    vel_rmse   = _rmse(_flat(lambda m: m.vel_errors))
    yaw_rmse   = _rmse(_flat(lambda m: m.yaw_errors))
    len_rmse   = _rmse(_flat(lambda m: m.length_errors))
    wid_rmse   = _rmse(_flat(lambda m: m.width_errors))

    mean_gospa = float(np.nanmean(gospa_by_frame))

    # MOTP / MOTA
    total_tp_dist = _flat(lambda m: m.pos_errors)
    mean_motp = float(np.mean(total_tp_dist)) if total_tp_dist else float("nan")

    all_n_gt  = sum(fm.n_gt  for run in all_runs for fm in run)
    all_n_tp  = sum(fm.n_tp  for run in all_runs for fm in run)
    all_n_fp  = sum(fm.n_fp  for run in all_runs for fm in run)
    all_idsw  = sum(fm.id_switches for run in all_runs for fm in run)
    n_cells   = n_mc * n_frames

    mean_mota = mota(all_n_gt, all_n_tp, all_n_fp, all_idsw)
    mean_fa_rate = all_n_fp / n_cells
    mean_miss_rate = (all_n_gt - all_n_tp) / all_n_gt if all_n_gt > 0 else float("nan")
    total_idsw_per_run = all_idsw / n_mc

    return EvalResult(
        scenario_name=scenario.name,
        n_mc=n_mc,
        n_frames=n_frames,
        gospa_by_frame=gospa_by_frame,
        pos_rmse_by_frame=pos_rmse_by_frame,
        vel_rmse_by_frame=vel_rmse_by_frame,
        yaw_rmse_by_frame=yaw_rmse_by_frame,
        length_rmse_by_frame=len_rmse_by_frame,
        width_rmse_by_frame=wid_rmse_by_frame,
        id_switches_by_frame=idsw_by_frame,
        mean_gospa=mean_gospa,
        mean_pos_rmse=pos_rmse,
        mean_vel_rmse=vel_rmse,
        mean_yaw_rmse=yaw_rmse,
        mean_length_rmse=len_rmse,
        mean_width_rmse=wid_rmse,
        total_id_switches=total_idsw_per_run,
        mean_motp=mean_motp,
        mean_mota=mean_mota,
        mean_false_alarm_rate=mean_fa_rate,
        mean_miss_rate=mean_miss_rate,
    )
