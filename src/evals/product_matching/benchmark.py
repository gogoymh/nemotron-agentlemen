"""Product-matching BYOB benchmark — evaluates against data/multi-platform/test.csv.

The full rendered `prompt/reason_prompt.txt` travels in the per-row
`question` field. Labels are "0" / "1" per the prompt's expected output:
`<reason>...</reason><label>0|1</label>`. Per-sample confusion indicators
(tp/fp/fn/tn) are emitted so `report.py` can compute P/R/F1 post-hoc.

Serving note: vLLM / OpenAI-compat endpoints should be launched with
`enable_thinking=False` (or equivalent) so the chat template doesn't open
a `<think>` block the model wasn't trained to continue.

Before compiling, run::

    python -m src.evals.product_matching.prepare_data

to regenerate `data/test.jsonl` and `system_prompt.txt`.

Compile::

    nemo-evaluator-byob src/evals/product_matching/benchmark.py

Run against a vLLM-served checkpoint::

    nemo-evaluator run_eval \\
        --eval_type byob_product_matching.product_matching \\
        --model_url http://localhost:8000 \\
        --model_id nemotron-30b-sft \\
        --model_type chat \\
        --output_dir results/SFT \\
        --api_key_name OPENAI_API_KEY
    python -m src.evals.product_matching.report results/SFT
"""
from __future__ import annotations

import re
from pathlib import Path

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data" / "test.jsonl"

if not _DATA.exists():
    raise FileNotFoundError(
        f"{_DATA} is missing. Run "
        "`python -m src.evals.product_matching.prepare_data` first."
    )

_LABEL = re.compile(r"<label>\s*([01])\s*</label>")
_REASON = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)


@benchmark(
    name="product_matching",
    dataset=str(_DATA),
    prompt="{question}",
    system_prompt="system_prompt.txt",
    target_field="gold_label",
    endpoint_type="chat",
)
@scorer
def product_matching_scorer(sample: ScorerInput) -> dict:
    response = sample.response or ""
    gold = str(sample.target).strip()

    label_m = _LABEL.search(response)
    pred = label_m.group(1) if label_m else None

    reason_m = _REASON.search(response)
    reason_body = reason_m.group(1).strip() if reason_m else ""
    reason_chars = len(reason_body)

    is_pos_gold = gold == "1"
    is_pos_pred = pred == "1"
    return {
        "accuracy": float(pred == gold),
        "parse_rate": float(pred is not None),
        "tp": float(is_pos_pred and is_pos_gold),
        "fp": float(is_pos_pred and not is_pos_gold),
        "fn": float((not is_pos_pred) and is_pos_gold),
        "tn": float((not is_pos_pred) and (not is_pos_gold)),
        "gold_matched": float(is_pos_gold),
        "pred_matched": float(is_pos_pred),
        "reason_emit_rate": float(reason_chars > 0),
        "reason_chars": float(reason_chars),
    }
