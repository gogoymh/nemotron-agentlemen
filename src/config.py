"""Shared SFT/eval config — single source of truth for paths, prompt, model spec.

The system prompt is loaded from `prompt/reason_prompt.txt` (with
`{p1_name}` / `{p2_name}` placeholders rendered per-row). Labels are 0/1
matching the prompt's expected output: `<reason>...</reason><label>0|1</label>`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data" / "multi-platform"

# Bridge's preloaded-finetune dataset maker hard-codes filenames
# `training.jsonl` / `validation.jsonl` under `dataset.dataset_root`.
# We stage that dir per-run and symlink into it.
SFT_DATA_DIR = REPO_ROOT / "data" / "sft"

RESULTS_DIR = REPO_ROOT / "results"
SFT_RESULTS_DIR = RESULTS_DIR / "SFT"

# Training artifacts (checkpoints, logs) live on /ephemeral because the 30B
# checkpoint alone is ~60GB bf16. See .claude/CLAUDE.md for the disk layout.
EPHEMERAL_ROOT = Path("/ephemeral")
SFT_OUTPUT_DIR = EPHEMERAL_ROOT / "nemotron-agentlemen_artifacts" / "sft"

# Reasoning prompt is the sole system message. `{p1_name}` / `{p2_name}` are
# the only placeholders — rendered per row via `render_prompt` below.
REASON_PROMPT_PATH = REPO_ROOT / "prompt" / "reason_prompt.txt"


def load_reason_prompt(path: Path | str | None = None) -> str:
    p = Path(path) if path else REASON_PROMPT_PATH
    return p.read_text(encoding="utf-8")


def render_prompt(p1_name: str, p2_name: str, template: str | None = None) -> str:
    """Fill {p1_name}/{p2_name} without touching any other braces."""
    tpl = template if template is not None else load_reason_prompt()
    return tpl.replace("{p1_name}", p1_name).replace("{p2_name}", p2_name)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str                 # HF repo id OR local path
    local_path: str            # pre-downloaded snapshot on /ephemeral
    sft_recipe: str            # Megatron-Bridge recipe name (full SFT)
    peft_recipe: str           # Megatron-Bridge recipe name (LoRA SFT)
    default_gpus: int


MODELS: dict[str, ModelSpec] = {
    "nemotron-30b": ModelSpec(
        key="nemotron-30b",
        hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        local_path="/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        sft_recipe="nemotron_3_nano_finetune_config",
        peft_recipe="nemotron_3_nano_finetune_config",
        default_gpus=8,
    ),
    "nemotron-120b": ModelSpec(
        key="nemotron-120b",
        hf_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16",
        local_path="/ephemeral/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        sft_recipe="nemotron_3_super_finetune_config",
        peft_recipe="nemotron_3_super_finetune_config",
        default_gpus=16,
    ),
}


def resolve_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'; choose from {list(MODELS)}")
    return MODELS[name]


# Label values used in JSONL targets and BYOB gold column. Mapping from the
# CSV `decision` column: matched → "1", not_matched → "0".
LABEL_MATCHED = "1"
LABEL_NOT_MATCHED = "0"
DECISION_TO_LABEL = {"matched": LABEL_MATCHED, "not_matched": LABEL_NOT_MATCHED}

LABEL_RE = r"<label>\s*([01])\s*</label>"
REASON_RE = r"<reason>(.*?)</reason>"


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """Minimal .env loader — sets os.environ and returns the parsed map.

    No dependency on python-dotenv so this works before the venv is set up.
    Lines beginning with `#` or missing `=` are ignored; values are NOT
    unquoted (wandb keys don't need it and shell-quoted values would
    confuse us otherwise).
    """
    if path is None:
        path = REPO_ROOT / ".env"
    path = Path(path)
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if val and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        env[key] = val
        os.environ.setdefault(key, val)
    return env
