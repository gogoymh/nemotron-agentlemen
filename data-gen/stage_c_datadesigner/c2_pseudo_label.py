"""C.2 — Pseudo-labelling on the curated 11M pool.

For each curated listing we mine a sibling candidate (brand-bucketed
nearest-neighbour over char TF-IDF on `product_name_clean`), then ask
Nemotron-3 Super to (a) decide matched/not_matched, (b) produce reasoning,
(c) self-assess confidence. Only `High` rows become SFT fodder with
`source="pseudo"`; `Low` rows are dropped to keep label noise out of SFT.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_designer.config.column_configs import (
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    Score,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.interface.data_designer import DataDesigner

from common.io import iter_jsonl, load_yaml, repo_root, resolve, write_jsonl
from common.schema import (
    CLEAN_TEXT,
    CONFIDENCE,
    LABEL,
    LABEL_RE,
    P1_NAME,
    P2_NAME,
    RAW_BRAND,
    RAW_TEXT,
    REASON,
    REASON_RE,
    SOURCE,
)
from stage_c_datadesigner._model import build_model_configs, v2_out

log = logging.getLogger("stage_c.c2")


PAIR_SYSTEM = (
    "You are a product-matching analyst for e-commerce. Decide whether two "
    "product listings refer to the same SKU. Output exactly:\n"
    "<reason>brief justification, mention brand/model/options</reason>"
    "<label>0|1</label>\n"
    "1 = matched, 0 = not_matched."
)

PAIR_USER = "Title A: {{ p1_name }}\nTitle B: {{ p2_name }}\n\nDecide and justify."


def _build_pairs(cfg: dict, k: int, sample_cap: int) -> Iterator[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in iter_jsonl(cfg["io"]["curated_jsonl"]):
        brand = row.get(RAW_BRAND) or "__unknown__"
        buckets[brand].append(row)

    rng = random.Random(42)
    emitted = 0
    for brand, rows in buckets.items():
        if len(rows) < 2:
            continue
        texts = [r.get(CLEAN_TEXT) or r[RAW_TEXT] for r in rows]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        mat = vec.fit_transform(texts)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(rows)), metric="cosine").fit(mat)
        _, idx = nn.kneighbors(mat)
        for i, neigh in enumerate(idx):
            cands = [j for j in neigh if j != i]
            if not cands:
                continue
            j = rng.choice(cands)
            yield {
                P1_NAME: texts[i],
                P2_NAME: texts[j],
                "_p1_raw": rows[i][RAW_TEXT],
                "_p2_raw": rows[j][RAW_TEXT],
                "_brand": brand,
            }
            emitted += 1
            if emitted >= sample_cap:
                return


def _extract_label(text: str) -> str:
    m = re.search(LABEL_RE, text or "")
    if not m:
        return ""
    return "matched" if m.group(1) == "1" else "not_matched"


def _extract_reason(text: str) -> str:
    m = re.search(REASON_RE, text or "", flags=re.DOTALL)
    return (m.group(1) if m else "").strip()


def run(cfg: dict) -> str:
    c2 = cfg["c2_pseudo_label"]
    target_hc = c2["target_high_conf"]
    sample_cap = int(target_hc * c2["oversample_factor"])

    providers, models = build_model_configs(cfg)
    seed_path = resolve(cfg["io"]["v2_out_dir"]) / "_c2_seed.jsonl"
    n_seed = write_jsonl(seed_path, _build_pairs(cfg, c2["sibling_sampler"]["k"], sample_cap))
    log.info("C.2 seed pairs: %d (target high-conf=%d, oversample=%.1fx)",
             n_seed, target_hc, c2["oversample_factor"])

    answer_col = LLMTextColumnConfig(
        name="model_answer",
        prompt=PAIR_USER,
        system_prompt=PAIR_SYSTEM,
        model_alias=cfg["model"]["alias"],
    )
    conf_col = LLMJudgeColumnConfig(
        name="confidence_judge",
        prompt=(
            "You previously answered:\n{{ model_answer }}\n\n"
            "Re-read the two titles and rate how confident you are in that answer."
        ),
        model_alias=cfg["model"]["alias"],
        scores=[
            Score(
                name="confidence",
                description=("High = titles clearly share brand/model/options; "
                             "Medium = at least one ambiguous attribute; "
                             "Low = insufficient signal or conflicting attributes."),
                categories=["High", "Medium", "Low"],
            ),
        ],
    )

    builder = DataDesignerConfigBuilder(model_providers=providers, model_configs=models)
    builder.with_seed_dataset(str(seed_path))
    builder.add_column(answer_col)
    builder.add_column(conf_col)

    designer = DataDesigner(
        artifact_path=str(resolve(cfg["io"]["v2_out_dir"]) / "_c2_artifacts")
    )
    result = designer.create(builder, num_records=n_seed, dataset_name="c2_pseudo")

    hc_tag = cfg["c2_pseudo_label"]["source_tag"]

    def emit():
        kept = 0
        for seed, out in zip(iter_jsonl(seed_path), result.dataset):
            conf = (out.get("confidence_judge") or {}).get("confidence", "Low")
            if conf != "High":
                continue
            yield {
                P1_NAME: seed[P1_NAME],
                P2_NAME: seed[P2_NAME],
                LABEL: _extract_label(out["model_answer"]),
                REASON: _extract_reason(out["model_answer"]),
                CONFIDENCE: conf,
                SOURCE: hc_tag,
                "_brand": seed["_brand"],
            }
            kept += 1
            if kept >= target_hc:
                return

    out_path = v2_out(cfg, "c2_pseudo_label")
    n_written = write_jsonl(out_path, emit())
    log.info("C.2 wrote %d High-confidence pseudo-labels → %s", n_written, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C.2 — pseudo-labelling")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/datadesigner.yaml"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
