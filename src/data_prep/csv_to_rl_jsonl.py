"""data/multi-platform/{rl,val}.csv → JSONL for NeMo RL GRPO (copy_lr.md).

CSV columns (both files): p1_name, p2_name, decision, evidence

NeMo RL's data processor reads two fields by default (`input_key`/`output_key`):
  - input  : the full rendered system prompt (same template as SFT — see
             prompt/reason_prompt.txt). The policy's chat template is applied
             at data-processor time with `add_generation_prompt=True`, so the
             JSONL only carries the raw prompt text plus metadata.
  - output : the gold label, either "0" or "1" (matched → "1",
             not_matched → "0"). The custom reward env parses
             `<label>0|1</label>` from the generation and compares to this.

Outputs:
  data/rl/training.jsonl    ← data/multi-platform/rl.csv
  data/rl/validation.jsonl  ← data/multi-platform/val.csv

The SFT path uses the same val.csv for its own `validation.jsonl`, so the
RL validation set matches what the SFT model was evaluated against.

Usage:
    python -m src.data_prep.csv_to_rl_jsonl
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
    REPO_ROOT,
    load_reason_prompt,
    render_prompt,
)

RL_DATA_DIR = REPO_ROOT / "data" / "rl"


def iter_rows(src_csv: Path):
    with open(src_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("p1_name") or not row.get("p2_name") or not row.get("decision"):
                continue
            yield row


def to_record(row: dict, template: str) -> dict | None:
    decision = row["decision"].strip().lower()
    label = DECISION_TO_LABEL.get(decision)
    if label is None:
        return None
    system_content = render_prompt(row["p1_name"], row["p2_name"], template=template)
    return {
        # Field names align with NeMo RL defaults (data.default.input_key /
        # output_key). Keep them plain strings — the data processor renders
        # the chat template itself.
        "input": system_content,
        "output": label,
        # Extra metadata carried through `extra_env_info` for logging.
        "decision": decision,
        "p1_name": row["p1_name"],
        "p2_name": row["p2_name"],
        "reference_evidence": (row.get("evidence") or "").strip(),
    }


def write_jsonl(records, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def convert(src_csv: Path, dst_jsonl: Path, template: str) -> tuple[int, int]:
    """Returns (n_rows_written, n_skipped)."""
    records, skipped = [], 0
    for row in iter_rows(src_csv):
        rec = to_record(row, template)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    return write_jsonl(records, dst_jsonl), skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, default=DATA_DIR / "rl.csv",
                   help="Source for training.jsonl (default: data/multi-platform/rl.csv).")
    p.add_argument("--val-csv", type=Path, default=DATA_DIR / "val.csv",
                   help="Source for validation.jsonl (default: data/multi-platform/val.csv).")
    p.add_argument("--out-dir", type=Path, default=RL_DATA_DIR)
    args = p.parse_args()

    for label, path in (("train", args.train_csv), ("val", args.val_csv)):
        if not path.exists():
            sys.exit(f"ERROR: missing {label} CSV: {path}")

    template = load_reason_prompt()

    jobs = [
        (args.train_csv, args.out_dir / "training.jsonl"),
        (args.val_csv, args.out_dir / "validation.jsonl"),
    ]
    for src, dst in jobs:
        n, skipped = convert(src, dst, template)
        suffix = f" (skipped {skipped} bad rows)" if skipped else ""
        print(f"{src} → {dst} ({n} rows{suffix})")


if __name__ == "__main__":
    main()
