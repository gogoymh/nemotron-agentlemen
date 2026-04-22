"""C.3 — Hard-pair synthesis (Nano §SFT 5-sample generate).

Given a seed title from the curated pool, we ask Nemotron-3 Super for five
variants:

- 2× hard-positive: same SKU, promotional rephrasings (bracketed tags,
  emoji prefixes, bundle-size callouts that don't change the unit).
- 3× hard-negative: same brand/line, different SKU along a single attribute
  (size, colour, generation, option pack).

Each variant is emitted as a labeled pair against the seed. C.4 validates.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.interface.data_designer import DataDesigner

from common.io import iter_jsonl, load_yaml, repo_root, resolve, write_jsonl
from common.schema import (
    CLEAN_TEXT,
    LABEL,
    P1_NAME,
    P2_NAME,
    RAW_BRAND,
    RAW_TEXT,
    REASON,
    SOURCE,
)
from stage_c_datadesigner._model import build_model_configs, v2_out

log = logging.getLogger("stage_c.c3")


VARIANT_SYSTEM = (
    "You generate adversarial product-title variants for a matching model.\n"
    "Given a seed product title, produce EXACTLY 5 variants as a JSON list.\n"
    'Schema: [{"kind": "hard_pos|hard_neg", "title": str, "reason": str}, ...]\n'
    "Rules:\n"
    "- 2 hard_pos variants: same SKU, different promotional wrapping (brackets, emoji, bundle size that\n"
    "  does not change the unit). Keep brand, model, generation, colour, size identical.\n"
    "- 3 hard_neg variants: same brand/line but differ along exactly ONE attribute\n"
    "  (size, colour, generation, option pack). Mention the differing attribute in `reason`.\n"
    "Return only the JSON list, no surrounding prose."
)

VARIANT_USER = "Seed title: {{ seed_title }}\nBrand: {{ seed_brand }}\n\nProduce the 5 variants now."


def _seed_rows(cfg: dict) -> Iterator[dict]:
    c3 = cfg["c3_hard_pairs"]
    rng = random.Random(1337)
    reservoir: list[dict] = []
    for i, row in enumerate(iter_jsonl(cfg["io"]["curated_jsonl"])):
        if len(reservoir) < c3["num_seeds"]:
            reservoir.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < c3["num_seeds"]:
                reservoir[j] = row
    n = 0
    for row in reservoir:
        n += 1
        yield {
            "seed_title": row.get(CLEAN_TEXT) or row[RAW_TEXT],
            "seed_brand": row.get(RAW_BRAND) or "unknown",
            "_seed_raw": row[RAW_TEXT],
        }
    log.info("C.3 seeded %d anchor titles", n)


def _parse_variants(raw: str) -> list[dict]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "[" in txt:
            txt = txt[txt.find("["):]
    try:
        parsed = json.loads(txt)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if (
            isinstance(item, dict)
            and item.get("kind") in ("hard_pos", "hard_neg")
            and isinstance(item.get("title"), str)
            and item["title"].strip()
        ):
            out.append(item)
    return out


def run(cfg: dict) -> str:
    providers, models = build_model_configs(cfg)
    c3 = cfg["c3_hard_pairs"]

    seed_path = resolve(cfg["io"]["v2_out_dir"]) / "_c3_seed.jsonl"
    n_seed = write_jsonl(seed_path, _seed_rows(cfg))

    variants_col = LLMTextColumnConfig(
        name="variants_json",
        prompt=VARIANT_USER,
        system_prompt=VARIANT_SYSTEM,
        model_alias=cfg["model"]["alias"],
    )

    builder = DataDesignerConfigBuilder(model_providers=providers, model_configs=models)
    builder.with_seed_dataset(str(seed_path))
    builder.add_column(variants_col)

    designer = DataDesigner(
        artifact_path=str(resolve(cfg["io"]["v2_out_dir"]) / "_c3_artifacts")
    )
    result = designer.create(builder, num_records=n_seed, dataset_name="c3_hard_pairs")

    def emit():
        for seed, out in zip(iter_jsonl(seed_path), result.dataset):
            for item in _parse_variants(out["variants_json"]):
                kind = item["kind"]
                yield {
                    P1_NAME: seed["seed_title"],
                    P2_NAME: item["title"].strip(),
                    LABEL: "matched" if kind == "hard_pos" else "not_matched",
                    REASON: (item.get("reason") or "").strip(),
                    SOURCE: f"{c3['source_tag']}_{'pos' if kind == 'hard_pos' else 'neg'}",
                    "_seed_brand": seed["seed_brand"],
                }

    out_path = v2_out(cfg, "c3_hard_pairs")
    n_written = write_jsonl(out_path, emit())
    log.info("C.3 wrote %d hard pairs → %s", n_written, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C.3 — hard-pair synthesis")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/datadesigner.yaml"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
