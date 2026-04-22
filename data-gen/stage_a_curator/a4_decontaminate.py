"""A.4 — Remove rows whose product_name matches any p1/p2 in v2 val/test.

We ignore case/whitespace so "[쿠폰] Apple AirPods Pro 2" and "Apple AirPods Pro 2"
both get caught. We also run the same strip logic on `product_name_clean`
before comparing so the noisy raw title can't sneak evaluation pairs back in.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from nemo_curator.stages.text.filters.score_filter import ScoreFilter
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

from common.io import iter_jsonl, load_yaml, repo_root, resolve
from common.schema import CLEAN_TEXT, RAW_TEXT

log = logging.getLogger("stage_a.a4")

_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().lower()


class DecontamFilter(DocumentFilter):
    """Score = 1.0 if the row's title is NOT in the eval union, else 0.0."""

    def __init__(self, eval_union: frozenset[str]) -> None:
        super().__init__()
        self._eval = eval_union

    def score_document(self, text: str) -> float:
        return 0.0 if _norm(text) in self._eval else 1.0

    def keep_document(self, score: float) -> bool:
        return score > 0.5


def _build_eval_union(cfg: dict) -> frozenset[str]:
    dc = cfg["decontam"]
    union: set[str] = set()
    for key in ("val_jsonl", "test_jsonl"):
        path = resolve(dc[key])
        if not path.exists():
            log.warning("decontam source missing: %s (skipping)", path)
            continue
        for row in iter_jsonl(path):
            for field in dc["union_fields"]:
                v = row.get(field)
                if isinstance(v, str) and v:
                    union.add(_norm(v))
    log.info("decontam union size: %d unique titles", len(union))
    return frozenset(union)


def run(cfg: dict, input_dir: Path | str) -> Path:
    eval_union = _build_eval_union(cfg)
    out_path = resolve(cfg["io"]["curated_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = JsonlReaderStage(file_paths=str(input_dir), fields=None)
    drop_raw = ScoreFilter(
        filter_obj=DecontamFilter(eval_union), text_field=RAW_TEXT, score_field="_decontam_raw"
    )
    drop_clean = ScoreFilter(
        filter_obj=DecontamFilter(eval_union), text_field=CLEAN_TEXT, score_field="_decontam_clean"
    )
    writer = JsonlWriter(path=str(out_path.parent), filename=out_path.name)

    Pipeline(name="stage_a_decontam", stages=[reader, drop_raw, drop_clean, writer]).run()
    log.info("A.4 decontam done → %s", out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A.4 decontamination")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/curator.yaml"))
    ap.add_argument("--input-dir", required=True, help="output of A.3")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config), args.input_dir)


if __name__ == "__main__":
    main()
