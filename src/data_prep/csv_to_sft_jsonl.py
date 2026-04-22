"""data/multi-platform/{sft,val,test}.csv → JSONL chat format for Megatron-Bridge.

CSV columns: p1_name,p2_name,decision,evidence

Produces two-message `{messages:[system, assistant]}` records:

  system    = prompt/reason_prompt.txt rendered with {p1_name}, {p2_name}
  assistant = "<reason>{evidence}</reason><label>{0|1}</label>"

Decision mapping: matched → "1", not_matched → "0".

Megatron-Bridge's `llm-finetune-preloaded` dataset expects filenames
`training.jsonl` / `validation.jsonl` directly under `dataset.dataset_root`.

Usage:
    python -m src.data_prep.csv_to_sft_jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import (  # noqa: E402
    DATA_DIR,
    DECISION_TO_LABEL,
    SFT_DATA_DIR,
    load_reason_prompt,
    render_prompt,
)


def to_record(row: dict, template: str) -> dict:
    system_content = render_prompt(row["p1_name"], row["p2_name"], template=template)
    evidence = (row.get("evidence") or "").strip()
    decision = row["decision"].strip().lower()
    label = DECISION_TO_LABEL.get(decision)
    if label is None:
        raise ValueError(f"unexpected decision={decision!r}; expected matched/not_matched")
    assistant_content = f"<reason>{evidence}</reason><label>{label}</label>"
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def convert(src_csv: Path, dst_jsonl: Path, template: str) -> int:
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src_csv, encoding="utf-8", newline="") as f_in, open(
        dst_jsonl, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if not row.get("p1_name") or not row.get("p2_name") or not row.get("decision"):
                continue
            rec = to_record(row, template)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=SFT_DATA_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    template = load_reason_prompt()

    jobs = [
        (args.data_dir / "sft.csv", args.out_dir / "training.jsonl"),
        (args.data_dir / "val.csv", args.out_dir / "validation.jsonl"),
        (args.data_dir / "test.csv", args.out_dir / "test.jsonl"),
    ]

    for src, dst in jobs:
        if not src.exists():
            sys.exit(f"ERROR: missing source CSV: {src}")
        n = convert(src, dst, template)
        print(f"{src.name} → {dst} ({n} rows)")


if __name__ == "__main__":
    main()
