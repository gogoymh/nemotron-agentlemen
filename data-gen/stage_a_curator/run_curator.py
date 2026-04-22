"""Stage A orchestrator — runs A.1 → A.2 → A.3 → A.4 on 10M raw listings.

Run from repo root:

    python data-gen/stage_a_curator/run_curator.py \
        --config data-gen/configs/curator.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import count_jsonl, load_yaml, repo_root, resolve
from stage_a_curator import a1_fuzzy_dedup, a2_quality_filter, a3_text_modifier, a4_decontaminate

log = logging.getLogger("stage_a")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A — Curator pipeline")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/curator.yaml"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["a1", "a2", "a3", "a4"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    cfg = load_yaml(args.config)

    raw = resolve(cfg["io"]["raw_jsonl"])
    n_raw = count_jsonl(raw) if raw.exists() else -1
    log.info("Stage A start — raw listings: %s (%s)", n_raw, raw)

    t0 = time.time()
    cur_dir: Path | str = raw

    if "a1" not in args.skip:
        cur_dir = a1_fuzzy_dedup.run(cfg)
    if "a2" not in args.skip:
        cur_dir = a2_quality_filter.run(cfg, cur_dir)
    if "a3" not in args.skip:
        cur_dir = a3_text_modifier.run(cfg, cur_dir)
    if "a4" not in args.skip:
        final = a4_decontaminate.run(cfg, cur_dir)
    else:
        final = cur_dir

    log.info("Stage A done in %.1fs → %s (kept %d rows)",
             time.time() - t0, final, count_jsonl(final))


if __name__ == "__main__":
    main()
