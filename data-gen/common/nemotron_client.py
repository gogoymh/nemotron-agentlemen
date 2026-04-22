"""Small OpenAI-SDK wrapper for a locally-served Nemotron-3 Super 120B endpoint.

Assumes vLLM is running via `src/serve.py --model nemotron-120b` on
`http://localhost:8000/v1`. DataDesigner's model registry can hit the same URL
— this module is only for the direct-call paths (C.4 judge, C.3 seeding)
where we don't want to route through DataDesigner's record pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

NEMOTRON_SUPER_HF_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


@dataclass
class NemotronClient:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "nemotron-super"   # vLLM served-model-name; resolves to the HF id above
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    enable_thinking: bool = True
    reasoning_budget: int | None = 512

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        n: int = 1,
        enable_thinking: bool | None = None,
        reasoning_budget: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Return `n` completion texts. Raises on API error — callers decide retry policy."""
        ctk: dict[str, Any] = {
            "enable_thinking": self.enable_thinking if enable_thinking is None else enable_thinking,
        }
        budget = self.reasoning_budget if reasoning_budget is None else reasoning_budget
        if budget is not None:
            ctk["reasoning_budget"] = budget

        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=n,
            extra_body={"chat_template_kwargs": ctk},
        )
        return [c.message.content or "" for c in rsp.choices]


def from_env() -> NemotronClient:
    return NemotronClient(
        base_url=os.environ.get("NEMOTRON_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("NEMOTRON_API_KEY", "EMPTY"),
        model=os.environ.get("NEMOTRON_MODEL_ALIAS", "nemotron-super"),
    )
