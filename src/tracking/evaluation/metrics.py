"""EOT 性能指標の計算。

実装する指標
-----------
位置・姿勢・形状
  position_error      : 推定位置 vs GT 位置 [m]
  velocity_error      : 推定速度 vs GT 速度 [m/s]
  yaw_error           : 推定方位角 vs GT yaw [rad]  (π 周期アンラップ)
  shape_error         : (length_err, width_err) [m]

多目標指標
  GOSPA               : Generalized OSPA (Rahmathullah 2017)
  MOTP                : Multiple Object Tracking Precision
  MOTA                : Multiple Object Tracking Accuracy

トラック品質
  id_switches         : フレーム間の GT-Est 対応の入れ替わり回数
  track_latency       : 新目標の検出に要したフレーム数
  false_alarm_rate    : フレームあたりのゴーストトラック数

参考文献
--------
Rahmathullah, A. S., García-Fernández, A. F., & Svensson, L. (2017).
  Generalized optimal sub-pattern assignment metric. FUSION 2017.
Bernardin, K., & Stiefelhagen, R. (2008).
  Evaluating multiple object tracking performance: the CLEAR MOT metrics.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 単一ペア誤差
# ---------------------------------------------------------------------------

def position_error(est_pos: np.ndarray, gt_pos: np.ndarray) -> float:
    """ユークリッド位置誤差 [m]。"""
    return float(np.linalg.norm(np.asarray(est_pos) - np.asarray(gt_pos)))


def velocity_error(est_vel: np.ndarray, gt_vel: np.ndarray) -> float:
    """速度ベクトル誤差のノルム [m/s]。"""
    return float(np.linalg.norm(np.asarray(est_vel) - np.asarray(gt_vel)))


def yaw_error(est_yaw: float, gt_yaw: float) -> float:
    """方位角誤差 [rad]、π 周期でアンラップ。

    OBB の theta は [-π/2, π/2) で方向が曖昧なため、
    gt_yaw との差を π でフォールドして最小値を返す。
    """
    d = float(est_yaw) - float(gt_yaw)
    # [-π/2, π/2) に折り畳む (π 周期)
    d = (d + np.pi / 2) % np.pi - np.pi / 2
    return float(abs(d))


def shape_error(
    est_length: float,
    est_width: float,
    gt_length: float,
    gt_width: float,
) -> tuple[float, float]:
    """(length 誤差, width 誤差) [m]。

    OBB は length ≥ width の慣例なので、GT も同じ順序に揃える。
    """
    gl, gw = sorted([gt_length, gt_width], reverse=True)
    return abs(est_length - gl), abs(est_width - gw)


# ---------------------------------------------------------------------------
# GOSPA (Generalized OSPA)
# ---------------------------------------------------------------------------

def gospa(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    *,
    c: float = 5.0,
    p: int = 2,
    alpha: float = 2.0,
) -> dict[str, float]:
    """GOSPA 指標を計算する。

    Parameters
    ----------
    gt_positions  : (N, 2)  地上真値位置
    est_positions : (M, 2)  推定位置
    c             : カットオフ距離 [m]
    p             : べき乗次数
    alpha         : カーディナリティペナルティ倍率 (通常 2)

    Returns
    -------
    dict with keys:
        gospa       : 総合 GOSPA スコア (小さいほど良い)
        loc         : 位置誤差成分
        missed      : 未検出ペナルティ成分
        false       : 誤検出ペナルティ成分
        n_matched   : マッチ数
    """
    gt = np.atleast_2d(gt_positions).reshape(-1, 2) if len(gt_positions) else np.empty((0, 2))
    est = np.atleast_2d(est_positions).reshape(-1, 2) if len(est_positions) else np.empty((0, 2))
    N, M = len(gt), len(est)

    if N == 0 and M == 0:
        return dict(gospa=0.0, loc=0.0, missed=0.0, false=0.0, n_matched=0)

    # ゼロ目標または推定なし
    if N == 0:
        val = (c ** p * M / max(N, M)) ** (1 / p)
        return dict(gospa=val, loc=0.0, missed=0.0, false=val, n_matched=0)
    if M == 0:
        val = (c ** p * N / max(N, M)) ** (1 / p)
        return dict(gospa=val, loc=0.0, missed=val, false=0.0, n_matched=0)

    # 距離行列
    diff = gt[:, None, :] - est[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)          # (N, M)
    d_cut = np.minimum(dist, c)                   # 閾値切り捨て

    # 最適割り当て (コスト = d_cut^p)
    cost = d_cut ** p
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    loc_sum = 0.0
    n_matched = 0
    for r, c_idx in zip(row_ind, col_ind):
        loc_sum += cost[r, c_idx]
        if dist[r, c_idx] < c:
            n_matched += 1

    n_max = max(N, M)
    missed_penalty = (c ** p) * (N - n_matched) / alpha
    false_penalty = (c ** p) * (M - n_matched) / alpha
    total = (loc_sum + missed_penalty + false_penalty) / n_max

    return dict(
        gospa=float(total ** (1 / p)),
        loc=float((loc_sum / n_max) ** (1 / p)),
        missed=float((missed_penalty / n_max) ** (1 / p)),
        false=float((false_penalty / n_max) ** (1 / p)),
        n_matched=n_matched,
    )


# ---------------------------------------------------------------------------
# MOTP / MOTA
# ---------------------------------------------------------------------------

def motp(matched_distances: list[float]) -> float:
    """MOTP: マッチしたペアの平均位置誤差 [m]。

    マッチが 0 件の場合は nan を返す。
    """
    if not matched_distances:
        return float("nan")
    return float(np.mean(matched_distances))


def mota(
    n_gt: int,
    n_tp: int,
    n_fp: int,
    n_id_switches: int,
) -> float:
    """MOTA: Multiple Object Tracking Accuracy。

    MOTA = 1 - (FN + FP + IDSW) / Σ GT

    完全にミスなし・誤検出なし・ID 変化なし → 1.0
    負になりうる (障害が多い場合)。
    """
    if n_gt == 0:
        return float("nan")
    n_fn = n_gt - n_tp
    return float(1.0 - (n_fn + n_fp + n_id_switches) / n_gt)


# ---------------------------------------------------------------------------
# ID Switch カウント
# ---------------------------------------------------------------------------

def count_id_switches(
    prev_assignment: dict[int, int],
    curr_assignment: dict[int, int],
) -> int:
    """フレーム間での GT-Est 対応の入れ替わり回数。

    Parameters
    ----------
    prev_assignment : {est_idx -> gt_idx}  前フレームの対応
    curr_assignment : {est_idx -> gt_idx}  今フレームの対応

    Returns
    -------
    int  入れ替わり回数 (GT 側で見た「以前と別の Est に割り当て直された」件数)
    """
    # GT 側の逆引き辞書
    prev_gt2est = {v: k for k, v in prev_assignment.items()}
    curr_gt2est = {v: k for k, v in curr_assignment.items()}

    switches = 0
    for gt_idx, est_idx in curr_gt2est.items():
        if gt_idx in prev_gt2est and prev_gt2est[gt_idx] != est_idx:
            switches += 1
    return switches


# ---------------------------------------------------------------------------
# 集計ヘルパー
# ---------------------------------------------------------------------------

def summarise(values: list[float], name: str = "") -> dict[str, float]:
    """リストの統計量を辞書で返す。"""
    arr = np.array([v for v in values if np.isfinite(v)])
    if len(arr) == 0:
        return {f"{name}_mean": float("nan"), f"{name}_std": float("nan"),
                f"{name}_rmse": float("nan"), f"{name}_p95": float("nan")}
    return {
        f"{name}_mean": float(arr.mean()),
        f"{name}_std":  float(arr.std()),
        f"{name}_rmse": float(np.sqrt((arr ** 2).mean())),
        f"{name}_p95":  float(np.percentile(arr, 95)),
    }
