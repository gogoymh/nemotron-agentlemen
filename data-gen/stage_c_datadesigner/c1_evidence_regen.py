"""C.1 — Evidence regeneration to rescue v1 SFT drops.

Rows where `label != decision` were dropped in v1 SFT because the gpt-5-mini
reasoning supported the *wrong* side. We re-prompt Nemotron-3 Super with the
human `label` fixed as ground truth and ask it to produce a consistent
`<reason>...</reason><label>0|1</label>` chain. Output rows carry
`source="regenerated"` and feed Stage D at blend weight 1.0.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.interface.data_designer import DataDesigner

from common.io import iter_jsonl, load_yaml, repo_root, resolve, write_jsonl
from common.schema import (
    DECISION,
    DECISION_TO_LABEL,
    LABEL,
    P1_NAME,
    P2_NAME,
    REASON,
    SOURCE,
)
from stage_c_datadesigner._model import build_model_configs, v2_out

log = logging.getLogger("stage_c.c1")

REGEN_SYSTEM = (
    "You are a product-matching analyst. You will be given two product titles "
    "and the ground-truth label (matched/not_matched). Produce an evidence "
    "chain that justifies the label. Output exactly one block:\n"
    "<reason>...</reason><label>0|1</label>\n"
    "Use 1 for matched, 0 for not_matched. Be concise (<200 words)."
)

REGEN_USER = (
    "Title A: {{ p1_name }}\n"
    "Title B: {{ p2_name }}\n"
    "Ground truth: {{ label }}\n\n"
    "Write the justification now."
)


def _seed_rows(cfg: dict):
    src = resolve(cfg["io"]["labeled_jsonl"])
    n_total = n_kept = 0
    for row in iter_jsonl(src):
        n_total += 1
        lbl = row.get(LABEL)
        dec = row.get(DECISION)
        if cfg["c1_evidence_regen"]["filter"]["require_label_ne_decision"] and lbl == dec:
            continue
        yield {
            P1_NAME: row[P1_NAME],
            P2_NAME: row[P2_NAME],
            LABEL: lbl,
            "_gold_label_digit": DECISION_TO_LABEL[lbl],
        }
        n_kept += 1
    log.info("C.1 seed: %d/%d rows kept (label != decision)", n_kept, n_total)


def run(cfg: dict) -> str:
    providers, models = build_model_configs(cfg)
    out_path = v2_out(cfg, "c1_evidence_regen")

    seed_path = resolve(cfg["io"]["v2_out_dir"]) / "_c1_seed.jsonl"
    n_seed = write_jsonl(seed_path, _seed_rows(cfg))
    if n_seed == 0:
        log.warning("C.1 no rows to regenerate; exiting")
        return out_path

    regen_col = LLMTextColumnConfig(
        name="regenerated_answer",
        prompt=REGEN_USER,
        system_prompt=REGEN_SYSTEM,
        model_alias=cfg["model"]["alias"],
    )

    builder = DataDesignerConfigBuilder(model_providers=providers, model_configs=models)
    builder.with_seed_dataset(str(seed_path))
    builder.add_column(regen_col)

    designer = DataDesigner(
        artifact_path=str(resolve(cfg["io"]["v2_out_dir"]) / "_c1_artifacts")
    )
    result = designer.create(builder, num_records=n_seed, dataset_name="c1_evidence_regen")

    def emit():
        for seed, out in zip(iter_jsonl(seed_path), result.dataset):
            yield {
                P1_NAME: seed[P1_NAME],
                P2_NAME: seed[P2_NAME],
                LABEL: seed[LABEL],
                REASON: out["regenerated_answer"],
                SOURCE: cfg["c1_evidence_regen"]["source_tag"],
            }

    n_written = write_jsonl(out_path, emit())
    log.info("C.1 wrote %d regenerated rows → %s", n_written, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C.1 — evidence regeneration")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/datadesigner.yaml"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
