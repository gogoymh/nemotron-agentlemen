"""Aggregate per-task pass rates from a rollouts JSONL.

Usage:
    python -m demo.env.reward_profile \\
        --input  demo/env/data/example.jsonl \\
        --rollouts results/rollouts.jsonl \\
        --output  results/profiled.jsonl \\
        --pass-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def profile(rollouts: List[Dict[str, Any]], pass_threshold: float) -> List[Dict[str, Any]]:
    groups: Dict[int, List[float]] = defaultdict(list)
    breakdowns: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        idx = int(r.get("task_index", -1))
        groups[idx].append(float(r.get("reward", 0.0)))
        if "reward_breakdown" in r:
            breakdowns[idx].append(r["reward_breakdown"])

    out = []
    for idx in sorted(groups):
        rewards = groups[idx]
        n = len(rewards)
        avg = sum(rewards) / n
        mx = max(rewards)
        passed = sum(1 for x in rewards if x >= pass_threshold)
        row = {
            "task_index": idx,
            "n_rollouts": n,
            "avg_reward": round(avg, 4),           # pass@1 proxy
            "max_reward": round(mx, 4),             # pass@k upper bound
            "pass_rate": round(passed / n, 4),
            "reward_stdev": round(statistics.stdev(rewards), 4) if n > 1 else 0.0,
        }
        # Aggregate breakdown (mean of numeric fields).
        if breakdowns[idx]:
            keys = set()
            for b in breakdowns[idx]:
                keys |= {k for k, v in b.items() if isinstance(v, (int, float))}
            agg: Dict[str, float] = {}
            for k in keys:
                vals = [b[k] for b in breakdowns[idx] if isinstance(b.get(k), (int, float))]
                if vals:
                    agg[k] = round(sum(vals) / len(vals), 4)
            row["mean_breakdown"] = agg
        out.append(row)
    return out


def print_aggregate(profiled: List[Dict[str, Any]]) -> None:
    if not profiled:
        print("(no tasks)")
        return
    n = len(profiled)
    avg = sum(p["avg_reward"] for p in profiled) / n
    pass1 = sum(1 for p in profiled if p["avg_reward"] >= 0.5) / n
    passk = sum(1 for p in profiled if p["max_reward"] >= 0.5) / n
    print(f"tasks: {n}")
    print(f"mean avg_reward: {avg:.4f}")
    print(f"pass@1 (avg_reward ≥ 0.5): {pass1:.2%}")
    print(f"pass@k (max_reward ≥ 0.5): {passk:.2%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute per-task pass rates from rollouts")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pass-threshold", type=float, default=0.5)
    ap.add_argument("--input", help="(optional) original tasks JSONL — unused, accepted for ng_reward_profile parity")
    args = ap.parse_args()

    rollouts = _load_jsonl(Path(args.rollouts))
    profiled = profile(rollouts, args.pass_threshold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in profiled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_aggregate(profiled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
