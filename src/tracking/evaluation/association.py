"""GT ↔ 推定値のハンガリアン法マッチング。

各フレームで地上真値 (GT) と推定値 (estimates) を最小コスト対応で結び、
TP / FP / FN と各対のインデックスリストを返す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class FrameAssignment:
    """1 フレーム分のマッチング結果。

    Attributes
    ----------
    matched_gt  : list[int]  対応した GT インデックス
    matched_est : list[int]  対応した 推定値インデックス
    missed_gt   : list[int]  対応なし GT (Miss)
    false_alarms: list[int]  対応なし 推定値 (False Alarm)
    distances   : list[float] マッチしたペアの位置距離 [m]
    """

    matched_gt: list[int]
    matched_est: list[int]
    missed_gt: list[int]
    false_alarms: list[int]
    distances: list[float]


def associate(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    gate_distance: float = 5.0,
) -> FrameAssignment:
    """GT と推定値を Hungarian 法で対応づける。

    Parameters
    ----------
    gt_positions  : (N, 2) ndarray  地上真値の 2D 位置
    est_positions : (M, 2) ndarray  推定値の 2D 位置
    gate_distance : float           これを超えるペアは対応不可 [m]

    Returns
    -------
    FrameAssignment
    """
    N = len(gt_positions)
    M = len(est_positions)

    if N == 0 and M == 0:
        return FrameAssignment([], [], [], [], [])
    if N == 0:
        return FrameAssignment([], [], [], list(range(M)), [])
    if M == 0:
        return FrameAssignment([], [], list(range(N)), [], [])

    # コスト行列: (N, M) ユークリッド距離
    gt_pos = np.asarray(gt_positions)    # (N, 2)
    est_pos = np.asarray(est_positions)  # (M, 2)
    diff = gt_pos[:, None, :] - est_pos[None, :, :]   # (N, M, 2)
    cost = np.linalg.norm(diff, axis=-1)               # (N, M)

    # gate を超えるペアは大コストに
    cost_gated = cost.copy()
    cost_gated[cost > gate_distance] = gate_distance * 1e3

    row_ind, col_ind = linear_sum_assignment(cost_gated)

    matched_gt: list[int] = []
    matched_est: list[int] = []
    distances: list[float] = []

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= gate_distance:
            matched_gt.append(int(r))
            matched_est.append(int(c))
            distances.append(float(cost[r, c]))

    matched_gt_set = set(matched_gt)
    matched_est_set = set(matched_est)
    missed_gt = [i for i in range(N) if i not in matched_gt_set]
    false_alarms = [j for j in range(M) if j not in matched_est_set]

    return FrameAssignment(
        matched_gt=matched_gt,
        matched_est=matched_est,
        missed_gt=missed_gt,
        false_alarms=false_alarms,
        distances=distances,
    )
