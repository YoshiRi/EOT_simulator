"""静的デモ用フレームデータを生成するスクリプト。

実行方法
--------
    uv run python scripts/generate_demo.py                     # single_approach
    uv run python scripts/generate_demo.py --scenario two_vehicles
    uv run python scripts/generate_demo.py --all              # 全シナリオ

出力: docs/data/demo_<scenario>.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.evaluation.interactive import SCENARIOS, run_simulation_json


def main() -> None:
    p = argparse.ArgumentParser(description="静的デモ用フレームデータを生成")
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="single_approach")
    p.add_argument("--all", action="store_true", help="全シナリオを生成")
    p.add_argument("--n-mc", type=int, default=5)
    p.add_argument("--out", default="docs/data")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(SCENARIOS.keys()) if args.all else [args.scenario]

    for key in keys:
        print(f"[{key}] n_mc={args.n_mc} ...", flush=True)
        raw = run_simulation_json(
            scenario_key=key,
            n_mc=args.n_mc,
            p_survival=0.99,
            p_detection=0.90,
            birth_weight=0.20,
            prune_log_threshold=-15.0,
        )
        data = json.loads(raw)
        out_path = out_dir / f"demo_{key}.json"
        out_path.write_text(json.dumps(data, indent=2))
        n_frames = len(data["frame_data"]["frames"])
        print(f"  → {out_path}  ({n_frames} frames)", flush=True)


if __name__ == "__main__":
    main()
