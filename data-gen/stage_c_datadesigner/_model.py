"""Shared DataDesigner model-registry helper.

Every C.x script wants the same `ModelConfig` pointing at the local Nemotron
Super vLLM endpoint — define it once here so `--config` changes (URL, alias,
sampling) propagate everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_designer.config.models import ModelConfig, ModelProvider

from common.io import resolve


def build_model_configs(cfg: dict) -> tuple[list[ModelProvider], list[ModelConfig]]:
    m = cfg["model"]
    provider = ModelProvider(
        name=m["provider"],
        base_url=m["base_url"],
        api_key=f"${{{m['api_key_env']}}}",
    )
    sampling = m["sampling"]
    model = ModelConfig(
        alias=m["alias"],
        provider_name=m["provider"],
        model=m["hf_id"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_tokens=sampling["max_tokens"],
        extra_body={"chat_template_kwargs": m["default_chat_template_kwargs"]},
    )
    return [provider], [model]


def v2_out(cfg: dict, key: str) -> str:
    return str(resolve(cfg[key]["out_jsonl"]))
