"""TRL-based GRPO for Nemotron-3-Nano-30B with LoRA.

NeMo-RL v0.4.0 (the image we were using) has no usable LoRA path on either
DTensor or Megatron backends for the vLLM-colocated GRPO entry. We therefore
switch to HuggingFace TRL's `GRPOTrainer`, which natively accepts a
`peft_config=LoraConfig(...)`.

Architecture:
  - Base model  : artifacts/sft/nemotron-30b/hf-iter_0001875 (SFT-merged HF)
  - PEFT        : LoRA r=32 α=64 dropout=0.05, target = q/k/v_proj
  - Dataset     : data/rl/{training,validation}.jsonl → HF Dataset(prompt=input)
  - Reward      : format(1.0) + correctness(2.0) + judge_mean(1.0) → scalar
  - Rollouts    : vLLM rollout server (TRL vllm-serve, TP=4) on GPUs 4-7.
                   Trainer pushes LoRA deltas between optimizer steps via TRL's
                   server weight-sync path. Replaces HF generate() which on a
                   device_map pipeline was the bottleneck (every token hopped
                   through all training stages serially).
  - Sharding    : single process + HF `device_map="auto"` splits the 30B bf16
                   base across GPUs 0-3 (~15 GB/GPU). LoRA adapters are the
                   only trainables, so no DDP/ZeRO. ZeRO-3 and QLoRA were both
                   tried and ruled out — see build_model() below.
  - Judge       : Azure OpenAI chat completions (gpt-5.4 via APIM). Creds in
                   .env (AZURE_ENDPOINT/AZURE_API_KEY/AZURE_MODEL/AZURE_API_VERSION).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

# TRL 0.19 GRPOTrainer requires `import vllm` to succeed even when
# vllm_mode="server" (rollouts go over HTTP, but the availability gate at
# grpo_trainer.py:625 still checks). vllm lives only in the per-Ray-worker
# venv in the NeMo-RL container. Append that venv's site-packages so vllm
# (and its transitive deps) become importable; APPEND — so main venv's
# transformers/torch/etc. keep priority and aren't shadowed.
_RAY_VLLM_SITE = (
    "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker."
    "VllmGenerationWorker/lib/python3.12/site-packages"
)
if os.path.isdir(_RAY_VLLM_SITE) and _RAY_VLLM_SITE not in sys.path:
    sys.path.append(_RAY_VLLM_SITE)

import httpx
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# `device_map="auto"` pipeline-shards the 30B base across cuda:0-3, so LoRA-
# adapted params (and the merged base weights during merge_adapter()) live on
# multiple GPUs. TRL's VLLMClient initializes its pynccl communicator on
# cuda:0 only; broadcasting a tensor whose `.device.index != 0` trips an
# assertion in pynccl.broadcast. Stage each tensor onto cuda:0 before the
# broadcast — the copy is freed right after, so peak extra VRAM is one
# parameter at a time.
from trl.extras.vllm_client import VLLMClient as _VLLMClient

_orig_update_named_param = _VLLMClient.update_named_param

def _update_named_param_cuda0(self, name, weights):
    if weights.device.type == "cuda" and weights.device.index != 0:
        weights = weights.to("cuda:0", non_blocking=False)
    return _orig_update_named_param(self, name, weights)

_VLLMClient.update_named_param = _update_named_param_cuda0


# Nemotron-H's HF modeling code ignores the `logits_to_keep` kwarg and always
# returns logits for the full sequence. TRL's `_get_per_token_logps` assumes
# the model honors `logits_to_keep`: it slices `input_ids` to the last N
# tokens but leaves logits at full seqlen, then errors on the subtract in
# `selective_log_softmax` (shape (B, full_len) vs (B, N) mismatch). Patch the
# method to slice logits down to `logits_to_keep` before the softmax.
from trl.trainer.grpo_trainer import GRPOTrainer as _GRPOTrainer
from trl.trainer.utils import selective_log_softmax as _selective_log_softmax

def _patched_get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
    batch_size = batch_size or input_ids.size(0)
    all_logps = []
    for i in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[i:i + batch_size]
        attention_mask_batch = attention_mask[i:i + batch_size]
        logits = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]
        if logits.size(1) > logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        input_ids_batch = input_ids_batch[:, -logits_to_keep:]
        logits = logits / self.temperature
        logps = _selective_log_softmax(logits, input_ids_batch)
        all_logps.append(logps)
    return torch.cat(all_logps, dim=0)

_GRPOTrainer._get_per_token_logps = _patched_get_per_token_logps


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_AGENT_DIR = REPO_ROOT / "prompt" / "reward_agent"
JUDGE_PROMPT_FILES = [
    "core_identity_agent_prompt.txt",
    "model_identifier_agent_prompt.txt",
    "variant_conflict_agent_prompt.txt",
]


# ─── Reward parsing primitives ────────────────────────────────────────────────
_LABEL_RE = re.compile(r"<label>\s*([01])\s*</label>")
_REASON_RE = re.compile(r"<reason>(.+?)</reason>", re.DOTALL)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _format_reward(response: str) -> tuple[float, str | None]:
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
    m = _LABEL_RE.search(response)
    if m is None:
        return 0.0, None
    pred = m.group(1)
    return (2.0 if pred == gold.strip() else 0.0), pred


def _parse_judge_float(text: str) -> float:
    m = _FLOAT_RE.search(text)
    if m is None:
        return 0.0
    try:
        v = float(m.group(0))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, v))


def _load_judge_prompts() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for fn in JUDGE_PROMPT_FILES:
        path = REWARD_AGENT_DIR / fn
        if not path.exists():
            raise FileNotFoundError(f"missing judge prompt: {path}")
        out.append((fn.removesuffix("_prompt.txt"), path.read_text("utf-8")))
    return out


def _render_judge_prompt(tmpl: str, p1: str, p2: str, reason_text: str) -> str:
    return (
        tmpl.replace("{p1_name}", p1)
        .replace("{p2_name}", p2)
        .replace("{reason_text}", reason_text)
    )


# ─── Judge HTTP client (Azure OpenAI chat completions) ────────────────────────
class JudgeClient:
    """Azure OpenAI chat-completions judge.

    The judge prompt templates are instruction-style text, so we wrap them as a
    single user message. Response is a short numeric string; `_parse_judge_float`
    extracts the float.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str,
        max_tokens: int,
        timeout_s: float,
        temperature: float,
        fail_soft: bool,
    ) -> None:
        self.url = f"{endpoint.rstrip('/')}/deployments/{deployment}/chat/completions"
        self.params = {"api-version": api_version}
        self.headers = {"api-key": api_key, "content-type": "application/json"}
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fail_soft = fail_soft
        self._http = httpx.Client(timeout=timeout_s)

    def score_one(self, prompt: str) -> float:
        # `max_tokens` is rejected on gpt-5.x deployments (returns 400 with
        # "Unsupported parameter"). Use `max_completion_tokens` exclusively.
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            r = self._http.post(
                self.url, params=self.params, headers=self.headers, json=payload
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return _parse_judge_float(text)
        except Exception as e:
            if not self.fail_soft:
                raise
            print(f"[reward] judge failed ({type(e).__name__}: {e}) → 0.0")
            return 0.0


# ─── Per-step JSONL dump (for eyeballing during training) ─────────────────────
class StepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._call_idx = 0

    def log_batch(self, rows: Iterable[dict]) -> None:
        self._call_idx += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                f.write(
                    json.dumps(
                        {"ts": ts, "call_idx": self._call_idx, "batch_idx": i, **row},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ─── Reward function TRL will call every step ─────────────────────────────────
def build_reward_fn(judge: JudgeClient, step_logger: StepLogger, agent_names: list[str],
                    judge_templates: list[str], judge_max_workers: int = 32):
    """Returns a TRL-style reward_fn(prompts, completions, **kwargs) -> list[float].

    Judge calls (n_samples × n_agents) are fanned out through a ThreadPoolExecutor
    so step time isn't dominated by serial HTTP latency.
    """

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        p1_name: list[str] | None = None,
        p2_name: list[str] | None = None,
        output: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        n = len(completions)
        p1s = p1_name or [""] * n
        p2s = p2_name or [""] * n
        golds = output or [""] * n

        fmts: list[float] = [0.0] * n
        corrs: list[float] = [0.0] * n
        preds: list[str | None] = [None] * n
        reasons: list[str | None] = [None] * n
        for i in range(n):
            fmts[i], reasons[i] = _format_reward(completions[i])
            corrs[i], preds[i] = _correctness_reward(completions[i], str(golds[i]))

        n_agents = len(judge_templates)
        per_agent: list[list[float]] = [[0.0] * n_agents for _ in range(n)]

        jobs: list[tuple[int, int, str]] = []
        for i in range(n):
            if not reasons[i]:
                continue
            for ai, tmpl in enumerate(judge_templates):
                jobs.append((i, ai, _render_judge_prompt(tmpl, p1s[i], p2s[i], reasons[i])))

        if jobs:
            workers = min(judge_max_workers, len(jobs))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(judge.score_one, p): (i, ai) for i, ai, p in jobs}
                for fut in futures:
                    i, ai = futures[fut]
                    per_agent[i][ai] = fut.result()

        rows: list[dict] = []
        totals: list[float] = []
        for i in range(n):
            judge_mean = sum(per_agent[i]) / max(1, n_agents)
            total = fmts[i] + corrs[i] + judge_mean
            totals.append(total)
            rows.append(
                {
                    "p1_name": p1s[i],
                    "p2_name": p2s[i],
                    "gold_label": golds[i],
                    "pred_label": preds[i],
                    "reason_text": reasons[i],
                    "format_reward": fmts[i],
                    "correctness_reward": corrs[i],
                    "judge_mean": judge_mean,
                    "judge_per_agent": dict(zip(agent_names, per_agent[i])),
                    "total_reward": total,
                    "response": completions[i],
                }
            )

        step_logger.log_batch(rows)
        return totals

    return reward_fn


# ─── Dataset loading ──────────────────────────────────────────────────────────
def load_jsonl_as_dataset(path: Path, tokenizer) -> Dataset:
    """Read JSONL and render the chat prompt up-front.

    TRL expects a 'prompt' column (string). Other columns pass through to the
    reward fn via **kwargs.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            system_content = d["input"]
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append(
                {
                    "prompt": rendered,
                    "output": str(d["output"]).strip(),
                    "p1_name": d.get("p1_name", ""),
                    "p2_name": d.get("p2_name", ""),
                }
            )
    return Dataset.from_list(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO_ROOT / "artifacts/sft/nemotron-30b/hf-iter_0001875"))
    ap.add_argument("--train-jsonl", default=str(REPO_ROOT / "data/rl/training.jsonl"))
    ap.add_argument("--val-jsonl", default=str(REPO_ROOT / "data/rl/validation.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/rl/nemotron-30b-trl"))

    ap.add_argument("--num-prompts-per-step", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--kl", type=float, default=0.01)
    ap.add_argument("--save-period", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=10)

    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj",
        help="Comma-separated HF linear-module names",
    )

    ap.add_argument("--azure-endpoint", default=os.environ.get("AZURE_ENDPOINT", ""))
    ap.add_argument("--azure-api-key", default=os.environ.get("AZURE_API_KEY", ""))
    ap.add_argument("--azure-deployment", default=os.environ.get("AZURE_MODEL", ""))
    ap.add_argument("--azure-api-version", default=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"))
    ap.add_argument("--judge-max-tokens", type=int, default=16)
    ap.add_argument("--judge-timeout-s", type=float, default=60.0)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-max-workers", type=int, default=32,
                    help="Max concurrent Azure judge HTTP calls per reward_fn invocation")

    ap.add_argument("--vllm-server-host", default=os.environ.get("VLLM_SERVER_HOST", "127.0.0.1"))
    ap.add_argument("--vllm-server-port", type=int, default=int(os.environ.get("VLLM_SERVER_PORT", "8001")))
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "nemotron-agentlemen-rl"))
    ap.add_argument("--wandb-run-name", default=os.environ.get("WANDB_EXP_NAME", "rl-from-sft-trl"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-process run — 30B bf16 is sharded across the 4 visible GPUs by
    # HF's `device_map="auto"` (pipeline parallel, layer-wise split). world_size
    # stays 1 for gradient accumulation math.
    world_size = 1
    # TRL's effective_train_batch_size is measured in SAMPLES (post-generation),
    # and must be divisible by num_generations. Compute grad_accum so the total
    # samples/step == num_prompts × num_gens.
    samples_per_step = args.num_prompts_per_step * args.num_generations
    grad_accum = max(1, samples_per_step // max(1, args.per_device_bs * world_size))
    effective_bs = world_size * args.per_device_bs * grad_accum

    print(f"[trl-rl] model={args.model}")
    print(f"[trl-rl] world_size={world_size} per_dev_bs={args.per_device_bs} "
          f"grad_accum={grad_accum} num_gens={args.num_generations} "
          f"→ effective_bs={effective_bs} samples/step "
          f"(= {args.num_prompts_per_step} prompts × {args.num_generations} gens)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_jsonl_as_dataset(Path(args.train_jsonl), tokenizer)
    val_ds = load_jsonl_as_dataset(Path(args.val_jsonl), tokenizer)
    print(f"[trl-rl] train={len(train_ds)}  val={len(val_ds)}")

    agents = _load_judge_prompts()
    agent_names = [n for n, _ in agents]
    agent_templates = [t for _, t in agents]
    if not (args.azure_endpoint and args.azure_api_key and args.azure_deployment):
        raise SystemExit(
            "Azure judge config missing. Set AZURE_ENDPOINT / AZURE_API_KEY / "
            "AZURE_MODEL (and optionally AZURE_API_VERSION) in .env, or pass "
            "--azure-endpoint / --azure-api-key / --azure-deployment."
        )
    judge = JudgeClient(
        endpoint=args.azure_endpoint,
        api_key=args.azure_api_key,
        deployment=args.azure_deployment,
        api_version=args.azure_api_version,
        max_tokens=args.judge_max_tokens,
        timeout_s=args.judge_timeout_s,
        temperature=args.judge_temperature,
        fail_soft=True,
    )
    step_logger = StepLogger(out_dir / "generations" / "generations.jsonl")
    reward_fn = build_reward_fn(
        judge, step_logger, agent_names, agent_templates,
        judge_max_workers=args.judge_max_workers,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Base model loading. We ruled out the obvious alternatives:
    #   - QLoRA (bnb 4-bit): Nemotron-H's fused `mamba_split_conv1d_scan_combined`
    #     kernel calls F.linear on `outproj_weight` directly, expecting a 2-D
    #     tensor. bnb stores it as a 1-D uint8 pack → shape mismatch crash.
    #   - DeepSpeed ZeRO-3: its `unwrap_model` gathers the full 30B onto each
    #     rank during generate() → CUBLAS_STATUS_ALLOC_FAILED on H100-80GB.
    #   - DDP w/ full bf16 replica: 60 GB × 4 ranks doesn't fit.
    # So we load bf16 once and let HF `device_map="auto"` layer-split the 30B
    # across the 4 visible GPUs (~15 GB/GPU). LoRA adapters are the only
    # trainable params, so no all-reduce is needed — single-process is fine.
    #
    # Attn: NemotronH has neither FA2 nor SDPA in the transformers dispatch
    # table, so eager is the only option. Mamba blocks still use the fused
    # causal-conv1d / mamba-ssm CUDA kernels.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="auto",
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        overwrite_output_dir=True,

        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_generations=args.num_generations,

        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        max_steps=args.max_steps,
        num_train_epochs=1,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.grad_clip,
        beta=args.kl,

        bf16=True,
        gradient_checkpointing=False,

        # vLLM rollout server runs out-of-process on GPUs 4-7 (TP=4). HF
        # `generate()` through the training pipeline was the bottleneck —
        # every rollout token had to hop through all training stages serially.
        # Server mode lets TRL push LoRA adapter deltas after each optimizer
        # step; inference happens with PagedAttention + continuous batching
        # on its own GPU pool.
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,

        temperature=1.0,
        top_p=1.0,

        save_strategy="steps",
        save_steps=args.save_period,
        save_total_limit=3,
        logging_steps=1,
        logging_first_step=True,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        remove_unused_columns=False,  # keep p1_name/p2_name/output for reward_fn
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    trainer.train()
    trainer.save_model(str(out_dir / "checkpoints" / "final"))


if __name__ == "__main__":
    main()
