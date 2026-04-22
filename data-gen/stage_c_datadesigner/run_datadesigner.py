"""Stage C orchestrator — C.1 → C.2 → C.3 → C.4.

All four stages point the DataDesigner model registry at a local vLLM server
serving nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16. Start it first:

    python src/serve.py --model nemotron-120b --port 8000

Then from repo root:

    export NEMOTRON_API_KEY=EMPTY
    python data-gen/stage_c_datadesigner/run_datadesigner.py \
        --config data-gen/configs/datadesigner.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_yaml, repo_root
from stage_c_datadesigner import (
    c1_evidence_regen,
    c2_pseudo_label,
    c3_hard_pairs,
    c4_judge_validator,
)

log = logging.getLogger("stage_c")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C — DataDesigner pipeline")
    ap.add_argument("--config", default=str(repo_root() / "data-gen/configs/datadesigner.yaml"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["c1", "c2", "c3", "c4"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    cfg = load_yaml(args.config)

    t0 = time.time()
    if "c1" not in args.skip:
        c1_evidence_regen.run(cfg)
    if "c2" not in args.skip:
        c2_pseudo_label.run(cfg)
    if "c3" not in args.skip:
        c3_hard_pairs.run(cfg)
    if "c4" not in args.skip:
        c4_judge_validator.run(cfg)
    log.info("Stage C done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
