"""Lightweight YAML loader with ${VAR:-default} env-var expansion.

Keeps the env server dependency-free beyond PyYAML + pydantic — no Hydra.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand(val: Any) -> Any:
    if isinstance(val, str):
        def sub(m):
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_PATTERN.sub(sub, val)
    if isinstance(val, dict):
        return {k: _expand(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_expand(v) for v in val]
    return val


class EnvConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18200
    cache_dir: str = "demo/cache"
    live_scraping: bool = False
    use_system_chrome: bool = True
    platforms: list[str] = Field(
        default_factory=lambda: [
            "naver", "ohouse", "eleventh_street",
            "emartinternetshopping", "shinsegaemall", "lotteon",
        ]
    )


class JudgeConfig(BaseModel):
    base_url: str = "http://localhost:8001/v1"
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    api_key: str = "dummy"
    timeout_s: int = 60
    max_tokens: int = 512
    temperature: float = 0.0
    mock: bool = False
    concurrency: int = 8


class PolicyConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "nemotron-match"
    api_key: str = "dummy"
    timeout_s: int = 120
    max_tokens: int = 1024
    temperature: float = 1.0


class RewardConfig(BaseModel):
    alpha: float = 0.5        # weight on specificity (TN rate)
    beta: float = 0.25        # false-positive penalty
    gamma: float = 0.25       # false-negative penalty
    delta: float = 0.05       # per-call budget overflow penalty
    min_positive_bonus: float = 0.1   # bonus for landing ≥1 true-positive
    no_submit_penalty: float = 0.0    # penalty when submissions=0


class EpisodeConfig(BaseModel):
    max_turns: int = 16
    max_tool_calls_per_turn: int = 4
    max_output_chars: int = 6000   # per-tool-result truncation before feeding back


class EnvRootConfig(BaseModel):
    env: EnvConfig = EnvConfig()
    judge: JudgeConfig = JudgeConfig()
    policy: PolicyConfig = PolicyConfig()
    reward: RewardConfig = RewardConfig()
    episode: EpisodeConfig = EpisodeConfig()


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "product_matching.yaml"


def load_config(path: Optional[str | Path] = None) -> EnvRootConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        return EnvRootConfig()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = _expand(raw)
    return EnvRootConfig.model_validate(raw)


def dump_config(cfg: EnvRootConfig) -> Dict[str, Any]:
    return cfg.model_dump()
