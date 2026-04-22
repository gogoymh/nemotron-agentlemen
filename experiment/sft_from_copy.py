"""Minimal Nemotron-3-Nano SFT script, following copy.md.

Entry: torchrun --nproc_per_node=8 experiment/sft_from_copy.py [--peft lora]

Uses Megatron-Bridge's Nano recipe as in copy.md, swaps the default SQuAD
dataset for our HF-chat JSONL under `data/sft/`, and calls `finetune(...)`.

Expects a pre-converted Megatron dist_checkpoint (see
`experiment/convert_hf_to_megatron.py`). If the converted dir is not present,
it falls back to the HF id; Bridge will then try to import at load time.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_finetune_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

import sys as _sys
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from tokenizer_patch import save_patched_tokenizer, verify_mask  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "sft"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "sft" / "nemotron-30b"
DEFAULT_PRETRAINED = str(REPO_ROOT / "megatron_checkpoints" / "nemotron_nano")
HF_FALLBACK = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def count_lines(p: Path) -> int:
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def bind_local_rank() -> None:
    if not torch.cuda.is_available():
        return
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        torch.cuda.set_device(int(lr))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--pretrained", type=str, default=None,
                   help="Megatron dist_checkpoint dir. "
                        f"Default: {DEFAULT_PRETRAINED} (else HF id {HF_FALLBACK})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 → derive from epochs × ceil(N/GBS)")
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--lr-warmup-iters", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--peft", type=str, default="none",
                   help="'none' for full SFT, 'lora' or 'dora' for PEFT")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-target-modules", type=str, default="linear_qkv")
    p.add_argument("--wandb-project", type=str, default="nemotron-agentlemen-sft")
    p.add_argument("--wandb-exp-name", type=str, default="sft-from-copy")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--completion-only-loss", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Patch chat_template so loss is computed only on assistant "
                        "tokens (--no-completion-only-loss = legacy behavior: loss on all tokens).")
    args = p.parse_args()

    bind_local_rank()

    train_path = args.data_root / "training.jsonl"
    val_path = args.data_root / "validation.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"missing {train_path} or {val_path}")

    # Epoch → iter derivation.
    if args.max_steps > 0:
        train_iters = args.max_steps
    else:
        n = count_lines(train_path)
        iters_per_epoch = max(1, math.ceil(n / args.global_batch_size))
        train_iters = args.num_epochs * iters_per_epoch
        print(f"[sft] n_train={n} gbs={args.global_batch_size} "
              f"→ iters/epoch={iters_per_epoch} × epochs={args.num_epochs} "
              f"= train_iters={train_iters}")

    eval_interval = args.eval_interval or max(1, train_iters // max(args.num_epochs, 1))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10.0

    pretrained = args.pretrained or (
        DEFAULT_PRETRAINED if Path(DEFAULT_PRETRAINED).is_dir() else HF_FALLBACK
    )
    wandb_project = None if args.no_wandb or not os.environ.get("WANDB_API_KEY") else args.wandb_project

    # 1) Build config from the official Nano SFT recipe (copy.md snippet).
    cfg = nemotron_3_nano_finetune_config(
        peft=args.peft,
        seq_length=args.seq_length,
        train_iters=train_iters,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        finetune_lr=args.lr,
        min_lr=min_lr,
        lr_warmup_iters=args.lr_warmup_iters,
        lr_decay_iters=train_iters,
        eval_interval=eval_interval,
        save_interval=args.save_interval,
        dir=str(args.output_dir),
        name="run",
        wandb_project=wandb_project,
        wandb_exp_name=args.wandb_exp_name,
        # H100 PCIe: no cross-pair NVLink mesh, so replicate experts and keep
        # the simple alltoall dispatcher.
        expert_model_parallelism=1,
        enable_deepep=False,
        comm_overlap_config=CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=False,
        ),
    )

    cfg.model.seq_length = args.seq_length
    cfg.logger.log_interval = args.log_interval
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    if hasattr(cfg.model, "moe_flex_dispatcher_backend"):
        cfg.model.moe_flex_dispatcher_backend = None

    # Completion-only loss: patch the tokenizer's chat_template to wrap the
    # assistant emit with {% generation %}...{% endgeneration %}. Megatron-
    # Bridge's chat preprocessor (data/datasets/utils.py::_chat_preprocess)
    # then picks up the assistant-tokens mask from HF and uses it as loss_mask.
    # Without this, Nemotron-3-Nano's shipped template has no generation
    # marker, so mask falls back to all-ones and loss covers the system prompt.
    if args.completion_only_loss:
        # Write on rank 0 only, others wait via filesystem sentinel. Distributed
        # init has not happened yet at this point (finetune() does it later),
        # so we can't use torch.distributed.barrier.
        import time as _time
        base_tok = cfg.tokenizer.tokenizer_model  # HF repo id or dir
        patched_dir = args.output_dir / "tokenizer_completion_only"
        sentinel = patched_dir / ".ready"
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            patched_dir.parent.mkdir(parents=True, exist_ok=True)
            save_patched_tokenizer(base_tok, patched_dir)
            try:
                import json
                with open(args.data_root / "training.jsonl", encoding="utf-8") as f:
                    first = json.loads(f.readline())
                stats = verify_mask(patched_dir, first["messages"])
                print(f"[completion-only] mask stats on 1st train row: "
                      f"n_tokens={stats['n_tokens']} "
                      f"assistant={stats['n_assistant']} "
                      f"context={stats['n_context']} "
                      f"assistant_ratio={stats['assistant_ratio']:.3f}")
            except Exception as e:
                print(f"[completion-only] WARN: mask verify failed: {e}")
            sentinel.touch()
        else:
            deadline = _time.time() + 120
            while not sentinel.exists():
                if _time.time() > deadline:
                    raise RuntimeError(
                        f"[completion-only] rank {rank} timed out waiting for "
                        f"rank-0 to write {patched_dir}"
                    )
                _time.sleep(0.5)
        cfg.tokenizer.tokenizer_model = str(patched_dir)
        if rank == 0:
            print(f"[completion-only] using patched tokenizer at {patched_dir}")

    # 2) Swap default SQuAD → our HF-chat JSONL at data/sft/ (copy.md snippet).
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=str(args.data_root),
        seq_length=args.seq_length,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            # Chat collate only emits `attention_mask` when this is False.
            "get_attention_mask_from_fusion": False,
        },
        do_test=False,
    )

    # 3) Optional LoRA (copy.md snippet).
    if args.peft.lower() == "lora":
        targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        cfg.peft = LoRA(target_modules=targets, dim=args.lora_r, alpha=args.lora_alpha)

    # 4) Checkpoint wiring.
    cfg.checkpoint.pretrained_checkpoint = pretrained
    cfg.checkpoint.save = str(args.output_dir / "checkpoints")
    cfg.checkpoint.load = str(args.output_dir / "checkpoints")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("========== SFT (copy.md recipe) ==========")
    print(f"  pretrained:   {pretrained}")
    print(f"  data_root:    {args.data_root}")
    print(f"  output:       {args.output_dir}")
    print(f"  epochs:       {args.num_epochs} → train_iters={train_iters}")
    print(f"  lr:           {args.lr} (warmup={args.lr_warmup_iters}, min={min_lr})")
    print(f"  gbs/mbs:      {args.global_batch_size}/{args.micro_batch_size}")
    print(f"  seq_length:   {args.seq_length}")
    print(f"  peft:         {args.peft}")
    print(f"  wandb:        {wandb_project or '(off)'} exp={args.wandb_exp_name}")
    print("==========================================")

    # 5) Launch training (copy.md snippet).
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
