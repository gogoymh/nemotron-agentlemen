"""A.1 — Fuzzy dedup via MinHash LSH.

Wraps `nemo_curator.stages.deduplication.fuzzy.FuzzyDeduplicationWorkflow` with
our config. Target: collapse near-duplicate product listings whose only
difference is a promotional prefix / bracketed tag.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

from common.io import load_yaml, repo_root, resolve
from common.schema import RAW_ID, RAW_TEXT

log = logging.getLogger("stage_a.a1")


def run(cfg: dict) -> Path:
    cache = resolve(cfg["io"]["cache_dir"]) / "a1_fuzzy"
    cache.mkdir(parents=True, exist_ok=True)
    out_dir = resolve(cfg["io"]["v2_root"]) / "_stage_a" / "a1_dedup"
    out_dir.mkdir(parents=True, exist_ok=True)

    fd = cfg["fuzzy_dedup"]
    workflow = FuzzyDeduplicationWorkflow(
        cache_path=str(cache),
        output_path=str(out_dir),
        input_path=str(resolve(cfg["io"]["raw_jsonl"])),
        input_filetype="jsonl",
        text_field=RAW_TEXT,
        id_field=RAW_ID,
        char_ngrams=fd["char_ngrams"],
        num_bands=fd["num_bands"],
        minhashes_per_band=fd["minhashes_per_band"],
        perform_removal=True,
    )
    result = workflow.run()
    log.info("A.1 fuzzy-dedup done: %s rows kept at %s", result, out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A.1 fuzzy dedup")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/curator.yaml"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
