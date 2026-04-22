"""Flatten data/multi-platform/test.csv → BYOB-friendly `{question, gold_label}`.

`question` holds the fully-rendered `prompt/reason_prompt.txt` (with
{p1_name}/{p2_name} filled in). BYOB's static `system_prompt` can't be
templated per row, so we keep system_prompt.txt minimal and feed the
whole rendered prompt through BYOB's per-row `prompt` template.

Re-run this whenever `data/multi-platform/test.csv` or `reason_prompt.txt`
changes. Artifacts under `data/` here are .gitignored.

Usage:
    python -m src.evals.product_matching.prepare_data
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import (  # noqa: E402
    DATA_DIR,
    DECISION_TO_LABEL,
    load_reason_prompt,
    render_prompt,
)

# Minimal static system_prompt — full instructions travel per-row in `question`.
_SYSTEM_STUB = "You are a product-title matching analyst."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DATA_DIR / "test.csv")
    p.add_argument("--out", type=Path, default=_HERE / "data" / "test.jsonl")
    p.add_argument(
        "--system-prompt-out", type=Path, default=_HERE / "system_prompt.txt"
    )
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.system_prompt_out.write_text(_SYSTEM_STUB, encoding="utf-8")

    template = load_reason_prompt()

    n_in = n_out = n_drop = 0
    with open(args.source, encoding="utf-8", newline="") as f_in, open(
        args.out, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            n_in += 1
            if not row.get("p1_name") or not row.get("p2_name") or not row.get("decision"):
                n_drop += 1
                continue
            decision = row["decision"].strip().lower()
            gold_label = DECISION_TO_LABEL.get(decision)
            if gold_label is None:
                n_drop += 1
                continue
            question = render_prompt(row["p1_name"], row["p2_name"], template=template)
            f_out.write(
                json.dumps(
                    {"question": question, "gold_label": gold_label},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_out += 1

    print(f"source:        {args.source}")
    print(f"system prompt: {args.system_prompt_out}")
    print(f"output jsonl:  {args.out}")
    print(f"rows read:     {n_in}")
    print(f"rows written:  {n_out}")
    print(f"rows dropped:  {n_drop}")


if __name__ == "__main__":
    main()
