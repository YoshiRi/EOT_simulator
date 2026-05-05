# 矩形推定結果の可視化設計

**ステータス**: 承認済み  
**対象ブランチ**: `claude/review-repository-status-1GXDH`

---

## 目的

`demo.html` の 2D Canvas 可視化に、GGIW 楕円に加えて **OBB（推定矩形）** と
**クラスタ点群の色分け** を追加する。

現状は `run_once()` がフィルタ更新後にクラスタ情報を捨てており、
`fit_rectangle()` を呼ぶ手がかりがない。
本設計はこの欠落を解消し、フレームスナップショットに矩形推定を含める。

---

## アーキテクチャ概要

```
 LiDAR 点群
     │
     ▼
 partition(pts, eps)           ← MeasurementCell のリストを保持する
     │
     ├──▶ filt.update(cells)   ← 既存処理（内部で仮説更新）
     │
     └──▶ ests = filt.extract_estimates()
               │
               ▼
         クラスタ↔推定の対応 (最近傍マッチング)
               │
               ▼
         fit_rectangle(cell.points)   ← OBBResult
               │
               ▼
         LocalTracker.update(obb)     ← 軽量 ID 管理 + ExponentialSmoother
               │
               ▼
         フレームスナップショットに obb フィールドを追加
```

---

## コンポーネント詳細

### 1. クラスタ↔推定の対応 (`_match_clusters_to_estimates`)

- 入力: `cells: list[MeasurementCell]`, `ests: list[GGIWState]`
- アルゴリズム: セントロイド間のユークリッド距離で線形割り当て（`scipy.optimize.linear_sum_assignment`）
  - コスト行列サイズ: `(len(ests), len(cells))`
  - ゲート距離: `6.0 m`（超えた場合は None を割り当て）
- 出力: `list[MeasurementCell | None]`（ests と同じ長さ）

**最近傍マッチングを使う理由**:  
PMBM 内部の仮説割り当て（MAP 仮説）を外部に公開すると実装が複雑になる。
可視化用途では近似的な対応で十分。

### 2. 軽量トラック ID 管理 (`LocalTracker`)

PMBM はトラックに永続 ID を付与しない。
フレーム間でスムーザ状態を継続するために、ローカルで ID を管理する。

```python
class LocalTracker:
    """フレーム間の推定位置をハンガリアン法でマッチし、
    ローカル ID を維持しながら ExponentialSmoother を管理する。"""

    gate_dist: float = 5.0          # 同一トラックとみなす最大距離 [m]
    smoother_alpha: float = 0.4     # ExponentialSmoother の忘却係数

    _smoothers: dict[int, ExponentialSmoother]
    _positions: dict[int, np.ndarray]  # 前フレームの推定位置
    _next_id: int
```

**ID 割り当てルール**:
- 前フレームの推定位置と今フレームの推定位置をハンガリアン法でマッチ
- コスト = ユークリッド距離、ゲート距離超過はコスト∞（新規扱い）
- マッチした推定 → 既存 smoother を継続
- マッチしなかった推定 → 新規 smoother を生成

### 3. OBB スナップショット (`_obb_to_dict`)

`OBBResult` を JSON シリアライズ可能な dict に変換する。

```python
{
    "cx":      float,          # OBB 中心 x [m]
    "cy":      float,          # OBB 中心 y [m]
    "theta":   float,          # 長軸の角度 [rad]  ([-π/2, π/2))
    "l":       float,          # 長辺 [m]
    "w":       float,          # 短辺 [m]
    "corners": [[x,y]*4]       # 反時計回り 4 頂点
}
```

---

## JSON スキーマ変更

### フレームスナップショット（追加フィールド）

```jsonc
{
  "fi": 5,
  "missed": false,
  "gt":   [{"x":…, "y":…, "yaw":…, "l":…, "w":…}],
  "obs":  [[x, y], …],
  "clusters": [                          // NEW: クラスタ点群
    {
      "centroid": [x, y],
      "points":   [[x, y], …]
    }
  ],
  "est": [
    {
      "x":…, "y":…, "vx":…, "vy":…,
      "ext_a":…, "ext_b":…, "ext_theta":…,
      "obb": {                           // NEW: 推定 OBB（null の場合あり）
        "cx":…, "cy":…,
        "theta":…, "l":…, "w":…,
        "corners": [[x,y],[x,y],[x,y],[x,y]]
      }
    }
  ]
}
```

**`obb` が `null` になるケース**:
- 対応クラスタの点数が 2 未満（`fit_rectangle` に必要な最低点数）
- 推定に対応するクラスタが見つからない（ゲート距離超過）

---

## Canvas 描画の変更 (`demo.html`)

### 描画レイヤー順序（下から上）

| 順序 | 要素 | 色 |
|------|------|----|
| 1 | グリッド・軸 | `#2e3347` |
| 2 | LiDAR 観測点（obs） | 黄 `#facc15` |
| 3 | クラスタ識別色（clusters） | 推定ごとに色を変える |
| 4 | GT 矩形 | 緑 `#4ade80` |
| 5 | GGIW 楕円（extent） | 赤 `#f87171`（半透明） |
| 6 | 推定 OBB 矩形 | 橙 `#fb923c` |
| 7 | 速度ベクトル | 橙 `#fb923c` |
| 8 | センサ原点マーカー | 青 `#60a5fa` |

### 凡例追加

```
● センサ原点   ■ GT（緑）   ○ GGIW楕円（赤）   □ 推定OBB（橙）   · 観測点（黄）
```

---

## 実装対象ファイル

| ファイル | 変更種別 | 概要 |
|---------|---------|------|
| `src/tracking/evaluation/interactive.py` | 改修 | `LocalTracker` 追加、`_match_clusters_to_estimates()` 追加、`_obb_to_dict()` 追加、`run_once()` の `record_frames=True` パスで clusters・obb を記録 |
| `docs/demo.html` | 改修 | クラスタ点群の色分け描画、OBB 矩形描画、凡例更新 |
| `tests/tracking/test_interactive.py` | 追加 | フレームスナップショットの `clusters`・`obb` フィールド検証 |

**変更しないファイル**:
- `src/tracking/shape/rectangle_fitting.py` — API そのまま使用
- `src/tracking/shape/smoothing.py` — `ExponentialSmoother` そのまま使用
- `src/tracking/pmbm/pmbm_filter.py` — フィルタ内部には触れない

---

## テスト仕様

| テスト名 | 検証内容 |
|---------|---------|
| `test_frame_clusters_present` | `frame["clusters"]` が存在し、各要素が `centroid`・`points` を持つ |
| `test_frame_est_obb_structure` | 推定に `obb` キーがある（null または dict） |
| `test_obb_corners_are_4_points` | `obb.corners` が 4 点の 2D 座標リスト |
| `test_obb_dimensions_positive` | `obb.l > 0` かつ `obb.w > 0` |
| `test_obb_center_near_cluster_centroid` | OBB 中心とクラスタ重心の距離 < 3 m |
| `test_missed_frame_obb_is_null` | miss フレームのすべての est で `obb is null` |

---

## 制約・既知の限界

- **部分観測**: 車両の片側しか見えない場合、OBB の長辺は実際より短くなる。
  スムーザが緩和するが、完全には解消しない。
- **クラスタ誤対応**: 2 台シナリオで車両が接近すると、
  セントロイド最近傍マッチが誤る可能性がある。可視化用途なので許容する。
- **初期フレーム**: 誕生直後のトラックはスムーザの学習が浅く、OBB が不安定。
