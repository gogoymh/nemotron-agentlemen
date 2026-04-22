"""Compute precision / recall / F1 from BYOB's aggregated pass@1 output.

BYOB only mean-aggregates, so `benchmark.py` emits per-sample tp/fp/fn/tn
indicators whose means are the per-row rates. This script reads the
published evaluation JSON, reconstructs counts (mean × n), and prints
precision / recall / F1 alongside the raw means.

Usage:
    python -m src.evals.product_matching.report <run-output-dir>
    python -m src.evals.product_matching.report <path/to/results.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_results(root: Path) -> Path:
    if root.is_file():
        return root
    for candidate in sorted(root.rglob("*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "tasks" in data:
            return candidate
    raise FileNotFoundError(f"no BYOB results JSON found under {root}")


def _unwrap_score(entry: dict) -> tuple[float, int]:
    value = entry.get("value", entry.get("mean", 0.0))
    count = (entry.get("stats") or {}).get("count") or entry.get("count") or 0
    return float(value), int(count)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--out-summary", type=Path, default=None,
                   help="If set, also write a concise JSON summary here.")
    args = p.parse_args()

    results_path = _find_results(args.path)
    blob = json.loads(results_path.read_text(encoding="utf-8"))

    tasks = blob.get("tasks") or {}
    if "product_matching" not in tasks:
        sys.exit(
            f"expected 'product_matching' task in {results_path}; "
            f"found: {list(tasks)}"
        )
    scores = tasks["product_matching"]["metrics"]["pass@1"]["scores"]

    means: dict[str, float] = {}
    n = 0
    for key, entry in scores.items():
        mean, count = _unwrap_score(entry)
        means[key] = mean
        n = max(n, count)

    tp = means.get("tp", 0.0) * n
    fp = means.get("fp", 0.0) * n
    fn = means.get("fn", 0.0) * n
    tn = means.get("tn", 0.0) * n

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"=== Product-matching eval report ({results_path.name}) ===")
    print(f"  samples:          {n}")
    print(f"  accuracy:         {means.get('accuracy', 0.0):.4f}")
    print(f"  precision:        {precision:.4f}")
    print(f"  recall:           {recall:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  parse_rate:        {means.get('parse_rate', 0.0):.4f}")
    print(f"  reason_emit_rate:  {means.get('reason_emit_rate', 0.0):.4f}")
    print(f"  mean_reason_chars: {means.get('reason_chars', 0.0):.1f}")
    print(f"  confusion:         TP={tp:.0f} FN={fn:.0f} FP={fp:.0f} TN={tn:.0f}")

    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(
            json.dumps(
                {
                    "samples": n,
                    "accuracy": means.get("accuracy", 0.0),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "parse_rate": means.get("parse_rate", 0.0),
                    "reason_emit_rate": means.get("reason_emit_rate", 0.0),
                    "mean_reason_chars": means.get("reason_chars", 0.0),
                    "confusion": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
                    "source": str(results_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  summary written → {args.out_summary}")


if __name__ == "__main__":
    main()
