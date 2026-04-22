"""C.4 — Post-hoc judge + structural validator over all C.1/C.2/C.3 outputs.

Two checks, both must pass:

1. Structure — regex validator: `<reason>...</reason><label>[01]</label>`,
   reason length within [min_reason_chars, max_reason_chars].
2. Consistency — sample N independent judgements from Nemotron-3 Super
   ("given these two titles, do you agree the label is X?") and require
   `judge_min_agreement` fraction to say Yes. Mirrors Nano §SFT R1 filter.

Rows that fail either check are dropped. The output is what Stage D blends.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import iter_jsonl, load_yaml, repo_root, resolve, write_jsonl
from common.nemotron_client import NemotronClient, from_env
from common.schema import LABEL, LABEL_RE, P1_NAME, P2_NAME, REASON, REASON_RE

log = logging.getLogger("stage_c.c4")


JUDGE_SYSTEM = (
    "You are verifying a product-match label. Reply with exactly one token: "
    "YES if the proposed label is correct for the two titles; NO otherwise."
)

JUDGE_USER = (
    "Title A: {p1}\n"
    "Title B: {p2}\n"
    "Proposed label: {label}\n"
    "Is the label correct?"
)


def _structural_ok(row: dict, min_len: int, max_len: int, require_label: bool, require_reason: bool) -> bool:
    reason = row.get(REASON) or ""
    label = row.get(LABEL) or ""
    if require_reason and not (min_len <= len(reason) <= max_len):
        return False
    if require_label and label not in ("matched", "not_matched"):
        return False
    blob = row.get("_raw_llm_output")
    if isinstance(blob, str):
        if require_reason and not re.search(REASON_RE, blob, flags=re.DOTALL):
            return False
        if require_label and not re.search(LABEL_RE, blob):
            return False
    return True


def _judge_batch(client: NemotronClient, row: dict, n: int) -> float:
    user = JUDGE_USER.format(p1=row[P1_NAME], p2=row[P2_NAME], label=row[LABEL])
    answers = client.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        n=n,
        enable_thinking=False,
        reasoning_budget=0,
        temperature=0.3,
    )
    yes = sum(1 for a in answers if a.strip().upper().startswith("YES"))
    return yes / max(len(answers), 1)


def _iter_inputs(cfg: dict) -> Iterable[dict]:
    for path in cfg["c4_judge_validator"]["inputs"]:
        p = resolve(path)
        if not p.exists():
            log.warning("C.4 input missing (skipping): %s", p)
            continue
        yield from iter_jsonl(p)


def run(cfg: dict) -> str:
    v = cfg["c4_judge_validator"]
    client = from_env()
    out_path = resolve(v["out_jsonl"])

    n_in = n_struct_ok = n_judge_ok = 0

    def emit():
        nonlocal n_in, n_struct_ok, n_judge_ok
        for row in _iter_inputs(cfg):
            n_in += 1
            if not _structural_ok(
                row,
                v["min_reason_chars"],
                v["max_reason_chars"],
                v["require_label_tag"],
                v["require_reason_tag"],
            ):
                continue
            n_struct_ok += 1
            agree = _judge_batch(client, row, v["judge_num_samples"])
            if agree < v["judge_min_agreement"]:
                continue
            n_judge_ok += 1
            row["_judge_agreement"] = round(agree, 3)
            yield row

    n_written = write_jsonl(out_path, emit())
    log.info(
        "C.4 in=%d struct_ok=%d judge_ok=%d written=%d → %s",
        n_in, n_struct_ok, n_judge_ok, n_written, out_path,
    )
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C.4 — judge + validator")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/datadesigner.yaml"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
