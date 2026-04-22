"""A.2 — fastText quality filter + ScoreFilter stage.

fastText model is auto-trained from labeled.jsonl if missing: rows with
confidence ≥ hq_conf_min become __label__hq, rows ≤ lq_conf_max become
__label__lq. Applied over Stage A.1's deduped output.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters.fasttext.fasttext_filters import FastTextQualityFilter
from nemo_curator.stages.text.filters.score_filter import ScoreFilter
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

from common.io import iter_jsonl, load_yaml, repo_root, resolve
from common.schema import QUALITY_SCORE, RAW_TEXT

log = logging.getLogger("stage_a.a2")


def _train_fasttext_if_missing(cfg: dict) -> Path:
    qf = cfg["quality_filter"]
    model_path = resolve(qf["fasttext_model"])
    if model_path.exists():
        return model_path
    if not qf.get("train_from_labeled_if_missing", False):
        raise FileNotFoundError(f"fastText model not found at {model_path}")

    import fasttext

    labeled = resolve(cfg["io"]["labeled_jsonl"])
    hq_min, lq_max = qf["hq_conf_min"], qf["lq_conf_max"]
    log.info("training fastText quality model from %s", labeled)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for row in iter_jsonl(labeled):
            conf = float(row.get("confidence_score", row.get("confidence", 0.0)))
            text = row.get(RAW_TEXT) or row.get("p1_name") or ""
            if not text:
                continue
            if conf >= hq_min:
                f.write(f"__label__hq {text}\n")
            elif conf <= lq_max:
                f.write(f"__label__lq {text}\n")
        train_file = f.name

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = fasttext.train_supervised(
        input=train_file, lr=0.5, epoch=25, wordNgrams=2, minn=2, maxn=5, dim=64
    )
    model.save_model(str(model_path))
    log.info("fastText model → %s", model_path)
    return model_path


def run(cfg: dict, input_dir: Path | str) -> Path:
    model_path = _train_fasttext_if_missing(cfg)
    qf = cfg["quality_filter"]
    out_dir = resolve(cfg["io"]["v2_root"]) / "_stage_a" / "a2_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = JsonlReaderStage(file_paths=str(input_dir), fields=None)
    ft_filter = FastTextQualityFilter(
        model_path=str(model_path),
        label="__label__hq",
        alpha=qf["pareto_alpha"],
        seed=42,
    )
    scored = ScoreFilter(
        filter_obj=ft_filter,
        text_field=RAW_TEXT,
        score_field=QUALITY_SCORE,
        score_type=float,
    )
    writer = JsonlWriter(path=str(out_dir))

    Pipeline(name="stage_a_quality", stages=[reader, scored, writer]).run()
    log.info("A.2 quality filter done → %s", out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A.2 quality filter")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/curator.yaml"))
    ap.add_argument("--input-dir", required=True, help="output of A.1")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    random.seed(42)
    run(load_yaml(args.config), args.input_dir)


if __name__ == "__main__":
    main()
