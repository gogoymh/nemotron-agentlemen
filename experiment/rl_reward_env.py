"""Reward environment + data processor for product-match GRPO (copy_lr.md).

Reward design (three components, summed into a scalar total):
  (1) Structured-output tag — 0.5 per tag:
        +0.5  if the response contains a non-empty <reason>…</reason>
        +0.5  if the response contains a valid <label>0|1</label>
      Max: 1.0
  (2) Binary classification — 2.0 if the parsed label matches the gold
      label, else 0.0.
      Max: 2.0
  (3) LLM-as-judge — the 3 prompts under prompt/reward_agent/ are sent
      to a vLLM OpenAI-compatible judge server (120B, GPUs 4–7). Each
      returns a float in [0,1]; we take the mean across the 3 agents.
      Max: 1.0

Total ∈ [0, 4].

The env returns rewards with shape [B, 3] so each component is tracked
independently in wandb / tensorboard; NeMo-RL sums them to get the
optimization signal.

Environment config (passed from YAML → __init__):
    judge_url          : OpenAI-compatible base URL (e.g. http://localhost:8000/v1)
    judge_model        : model name to send in `model` field
    judge_max_tokens   : 8 is enough for a single float
    judge_timeout_s    : http timeout per request
    log_dir            : directory for per-step generation dumps
                          (rolled via append on <log_dir>/generations.jsonl)

Every step() writes one JSON line per rollout to `generations.jsonl` so the
user can eyeball outputs during training.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, cast

import ray
import torch

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_AGENT_DIR = REPO_ROOT / "prompt" / "reward_agent"
JUDGE_PROMPT_FILES = [
    "core_identity_agent_prompt.txt",
    "model_identifier_agent_prompt.txt",
    "variant_conflict_agent_prompt.txt",
]


# ─── Data processor ──────────────────────────────────────────────────────────
def product_match_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """SFT-compatible `[system]` prompt with generation prefix.

    `extra_env_info` carries through to the reward env — we pass p1/p2 names
    so the judge prompts can be rendered in-env.
    """
    system_content = datum_dict["input"]
    gold_label = str(datum_dict["output"]).strip()

    extra_env_info: dict[str, Any] = {
        "ground_truth": gold_label,
        "p1_name": datum_dict.get("p1_name", ""),
        "p2_name": datum_dict.get("p2_name", ""),
        "decision": datum_dict.get("decision", ""),
    }

    system_msg: dict[str, str | torch.Tensor] = {
        "role": "system",
        "content": system_content,
    }
    rendered = tokenizer.apply_chat_template(
        [cast(dict[str, str], system_msg)],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    system_msg["token_ids"] = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    system_msg["content"] = rendered
    message_log: LLMMessageLogType = [system_msg]

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        for m in message_log:
            m["token_ids"] = m["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    out: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        out["task_name"] = datum_dict["task_name"]
    return out


# ─── Reward primitives ───────────────────────────────────────────────────────
_LABEL_RE = re.compile(r"<label>\s*([01])\s*</label>")
_REASON_RE = re.compile(r"<reason>(.+?)</reason>", re.DOTALL)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _format_reward(response: str) -> tuple[float, str | None]:
    """Returns (format_score, reason_text). Score is 0.5 per tag, max 1.0."""
    score = 0.0
    rm = _REASON_RE.search(response)
    reason_text: str | None = None
    if rm is not None and rm.group(1).strip():
        score += 0.5
        reason_text = rm.group(1).strip()
    if _LABEL_RE.search(response):
        score += 0.5
    return score, reason_text


def _correctness_reward(response: str, gold: str) -> tuple[float, str | None]:
    """Returns (score, parsed_label_or_None). 2.0 if match, else 0.0."""
    m = _LABEL_RE.search(response)
    if m is None:
        return 0.0, None
    pred = m.group(1)
    return (2.0 if pred == gold.strip() else 0.0), pred


# ─── Judge prompt loading ────────────────────────────────────────────────────
def _load_judge_prompts() -> list[tuple[str, str]]:
    """Returns list of (agent_name, prompt_template) for the 3 reward agents."""
    out: list[tuple[str, str]] = []
    for fn in JUDGE_PROMPT_FILES:
        path = REWARD_AGENT_DIR / fn
        if not path.exists():
            raise FileNotFoundError(f"missing judge prompt: {path}")
        name = fn.removesuffix("_prompt.txt")
        out.append((name, path.read_text(encoding="utf-8")))
    return out


def _render_judge_prompt(
    template: str, p1_name: str, p2_name: str, reason_text: str
) -> str:
    return (
        template.replace("{p1_name}", p1_name)
        .replace("{p2_name}", p2_name)
        .replace("{reason_text}", reason_text)
    )


def _parse_judge_float(text: str) -> float:
    """Prompts specify a bare float reply; take the first match and clamp."""
    m = _FLOAT_RE.search(text)
    if m is None:
        return 0.0
    try:
        v = float(m.group(0))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, v))


# ─── Reward environment ──────────────────────────────────────────────────────
@ray.remote(max_restarts=-1, max_task_retries=-1)
class ProductMatchRewardEnvironment(EnvironmentInterface):
    """Multi-reward env (copy_lr.md §6.3 pattern).

    Returns rewards shape [B, 3] — [format, correctness, judge_avg]. NeMo-RL
    sums them to produce the GRPO advantage signal; individual components
    stay visible in metrics.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.judge_url = cfg.get("judge_url") or os.environ.get(
            "JUDGE_URL", "http://localhost:8000/v1"
        )
        self.judge_model = cfg.get("judge_model") or os.environ.get(
            "JUDGE_MODEL", "nemotron-3-super-120b"
        )
        self.judge_max_tokens = int(cfg.get("judge_max_tokens", 16))
        self.judge_timeout_s = float(cfg.get("judge_timeout_s", 60.0))
        self.judge_temperature = float(cfg.get("judge_temperature", 0.0))
        # On transient server error, fail soft — don't kill training.
        self.judge_fail_soft = bool(cfg.get("judge_fail_soft", True))

        log_dir = cfg.get("log_dir") or os.environ.get(
            "RL_GENERATIONS_DIR",
            str(REPO_ROOT / "artifacts" / "rl" / "nemotron-30b" / "generations"),
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "generations.jsonl"
        self._log_lock = threading.Lock()
        self._call_idx = 0

        self.judge_prompts = _load_judge_prompts()

        # Lazy-construct the HTTP client (httpx is importable inside the
        # NeMo-RL container).
        import httpx  # noqa: WPS433

        self._http = httpx.Client(timeout=self.judge_timeout_s)

    # ── Judge call ──────────────────────────────────────────────────────────
    def _score_with_agent(
        self,
        template: str,
        p1_name: str,
        p2_name: str,
        reason_text: str,
    ) -> float:
        prompt = _render_judge_prompt(template, p1_name, p2_name, reason_text)
        payload = {
            "model": self.judge_model,
            "prompt": prompt,
            "temperature": self.judge_temperature,
            "max_tokens": self.judge_max_tokens,
            "stream": False,
        }
        url = self.judge_url.rstrip("/") + "/completions"
        try:
            r = self._http.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["text"]
        except Exception as e:
            if not self.judge_fail_soft:
                raise
            print(f"[reward] judge call failed ({type(e).__name__}: {e}); "
                  f"returning 0.0 for this agent")
            return 0.0
        return _parse_judge_float(text)

    def _judge_batch(
        self,
        p1_names: list[str],
        p2_names: list[str],
        reason_texts: list[str | None],
    ) -> tuple[list[float], list[list[float]]]:
        """Per-sample mean score across 3 agents + per-agent breakdown."""
        agent_scores: list[list[float]] = [
            [0.0 for _ in p1_names] for _ in self.judge_prompts
        ]
        for ai, (_, template) in enumerate(self.judge_prompts):
            for bi, (p1, p2, rt) in enumerate(
                zip(p1_names, p2_names, reason_texts)
            ):
                if not rt:
                    agent_scores[ai][bi] = 0.0
                    continue
                agent_scores[ai][bi] = self._score_with_agent(
                    template, p1, p2, rt
                )

        mean_scores: list[float] = []
        per_sample_breakdown: list[list[float]] = []
        for bi in range(len(p1_names)):
            per_agent = [agent_scores[ai][bi] for ai in range(len(self.judge_prompts))]
            per_sample_breakdown.append(per_agent)
            mean_scores.append(sum(per_agent) / max(1, len(per_agent)))
        return mean_scores, per_sample_breakdown

    # ── Step-by-step log ────────────────────────────────────────────────────
    def _dump_step(
        self,
        step_id: int,
        responses: list[str],
        metadata: list[dict],
        format_scores: list[float],
        correct_scores: list[float],
        judge_means: list[float],
        judge_breakdown: list[list[float]],
        preds: list[str | None],
        reasons: list[str | None],
    ) -> None:
        agent_names = [n for n, _ in self.judge_prompts]
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._log_lock, open(self.log_path, "a", encoding="utf-8") as f:
            for i, (resp, meta) in enumerate(zip(responses, metadata)):
                line = {
                    "ts": ts,
                    "call_idx": step_id,
                    "batch_idx": i,
                    "p1_name": meta.get("p1_name", ""),
                    "p2_name": meta.get("p2_name", ""),
                    "gold_label": meta.get("ground_truth", ""),
                    "pred_label": preds[i],
                    "reason_text": reasons[i],
                    "format_reward": format_scores[i],
                    "correctness_reward": correct_scores[i],
                    "judge_mean": judge_means[i],
                    "judge_per_agent": dict(zip(agent_names, judge_breakdown[i])),
                    "total_reward": (
                        format_scores[i] + correct_scores[i] + judge_means[i]
                    ),
                    "response": resp,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # ── EnvironmentInterface ────────────────────────────────────────────────
    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn:
        self._call_idx += 1
        step_id = self._call_idx

        responses = [
            "".join(
                str(m["content"])
                for m in conv
                if m["role"] == "assistant"
            )
            for conv in message_log_batch
        ]

        # (1) format + (2) correctness are pure-string; do them in one pass.
        format_scores: list[float] = []
        correct_scores: list[float] = []
        preds: list[str | None] = []
        reasons: list[str | None] = []
        for resp, meta in zip(responses, metadata):
            fmt, reason_text = _format_reward(resp)
            corr, pred = _correctness_reward(resp, str(meta["ground_truth"]))
            format_scores.append(fmt)
            correct_scores.append(corr)
            preds.append(pred)
            reasons.append(reason_text)

        # (3) judge — needs HTTP per (agent × sample). Batch by batch-index so
        # we can reuse one request per prompt.
        p1_names = [str(m.get("p1_name", "")) for m in metadata]
        p2_names = [str(m.get("p2_name", "")) for m in metadata]
        judge_means, judge_breakdown = self._judge_batch(p1_names, p2_names, reasons)

        self._dump_step(
            step_id,
            responses=responses,
            metadata=metadata,
            format_scores=format_scores,
            correct_scores=correct_scores,
            judge_means=judge_means,
            judge_breakdown=judge_breakdown,
            preds=preds,
            reasons=reasons,
        )

        # Shape [B, 3] — NeMo-RL sums columns to get the scalar reward.
        rewards = torch.tensor(
            list(zip(format_scores, correct_scores, judge_means)),
            dtype=torch.float32,
        )
        done = torch.ones(len(responses))

        return EnvironmentReturn(
            observations=[
                {"role": "environment", "content": ""} for _ in responses
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(responses),
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(self, batch):
        return batch, {}
