"""Full SFT entrypoint for Nemotron-3-Nano on data/sft — Megatron-Bridge nano-v3.

Replaces the older `run_recipe.py` path (removed on the nano-v3 branch). Calls
`nemotron_3_nano_finetune_config(...)` directly, then:

  - swaps the default SQuAD dataset for our HF-chat-format JSONL under `data/sft/`
  - forces the MoE token dispatcher back to "alltoall" so DeepEP (which needs a
    full NVLink mesh) is not triggered on H100 PCIe
  - derives train_iters from num_epochs × ceil(N_train / GBS)
  - wires WandB if WANDB_API_KEY is set in .env

Run under torchrun via experiment/sft.sh, or directly:

    torchrun --nproc_per_node=8 src/train_sft.py --num-epochs 5 --lr 1e-4
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import (  # noqa: E402
    MODELS,
    SFT_DATA_DIR,
    SFT_OUTPUT_DIR,
    load_dotenv,
    resolve_model,
)

from megatron.bridge.peft.lora import LoRA  # noqa: E402
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (  # noqa: E402
    nemotron_3_nano_finetune_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig  # noqa: E402
from megatron.bridge.training.config import FinetuningDatasetConfig  # noqa: E402
from megatron.bridge.training.finetune import finetune  # noqa: E402
from megatron.bridge.training.gpt_step import forward_step  # noqa: E402


logger = logging.getLogger(__name__)


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def bind_cuda_device_from_local_rank() -> None:
    """Pin each torchrun worker to its assigned GPU before NCCL barriers."""
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        torch.cuda.set_device(int(local_rank))
    except ValueError:
        logger.warning("ignoring non-integer LOCAL_RANK=%r", local_rank)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nemotron-30b", choices=list(MODELS))
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=None,
                   help="Cosine-decay floor (default: lr/10)")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--lr-warmup-iters", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 → derive train_iters from epochs × ceil(N_train/GBS)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Checkpoint/log output dir (default: $EPHEMERAL/.../sft/<key>)")
    p.add_argument("--pretrained-checkpoint", type=str, default=None,
                   help="Megatron dist_checkpoint dir (converted from HF via "
                        "experiment/convert_hf_to_megatron.py). Overrides spec.local_path.")
    p.add_argument("--data-root", type=Path, default=SFT_DATA_DIR,
                   help="Dir containing training.jsonl / validation.jsonl")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=None,
                   help="Default: once per epoch")
    p.add_argument("--log-interval", type=int, default=1,
                   help="Log (stdout + wandb) every N steps (default: 1)")
    p.add_argument("--peft", type=str, default="lora",
                   help="'none' → full SFT, 'lora'/'dora' for PEFT")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank (dim)")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    p.add_argument("--lora-target-modules", type=str, default="linear_qkv",
                   help="Comma-separated LoRA target modules. "
                        "'linear_qkv' covers Q/K/V (Megatron fuses them).")
    p.add_argument("--wandb-project", type=str, default="nemotron-agentlemen-sft")
    p.add_argument("--wandb-exp-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    bind_cuda_device_from_local_rank()

    train_path = args.data_root / "training.jsonl"
    val_path = args.data_root / "validation.jsonl"
    if not train_path.exists() or not val_path.exists():
        sys.exit(f"ERROR: missing {train_path} or {val_path}")

    load_dotenv()

    spec = resolve_model(args.model)
    out = Path(args.output_dir or (SFT_OUTPUT_DIR / spec.key))
    out.mkdir(parents=True, exist_ok=True)

    if args.pretrained_checkpoint:
        pretrained = args.pretrained_checkpoint
    else:
        pretrained = spec.local_path if Path(spec.local_path).is_dir() else spec.hf_id

    # Derive train_iters from epoch budget.
    if args.max_steps > 0:
        train_iters = args.max_steps
    else:
        n_train = count_jsonl_lines(train_path)
        iters_per_epoch = max(1, math.ceil(n_train / args.global_batch_size))
        train_iters = args.num_epochs * iters_per_epoch
        print(
            f"  (derived) n_train={n_train} gbs={args.global_batch_size} "
            f"→ iters_per_epoch={iters_per_epoch} × epochs={args.num_epochs} "
            f"= train_iters={train_iters}"
        )

    eval_interval = args.eval_interval or max(1, train_iters // max(args.num_epochs, 1))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10.0

    wandb_project = None if args.no_wandb or not os.environ.get("WANDB_API_KEY") else args.wandb_project
    wandb_exp_name = args.wandb_exp_name or f"sft-{spec.key}"

    # H100 PCIe only has NVLink within pairs (0-1, 2-3, 4-5, 6-7); cross-pair
    # peer access triggers `cudaErrorContained`. Kill every path that touches
    # peer memory:
    #   - EP=1 so MoE experts are replicated, no expert-parallel alltoall
    #     handshake (that was the late-barrier crash).
    #   - tp_comm_overlap=False so TE skips `initialize_ub()` NVLink userbuffer
    #     allocation on init.
    # TP=1, DP=8 over NCCL SHM/direct (already confirmed working by prior
    # NCCL_DEBUG=INFO logs).
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
        dir=str(out),
        name="run",
        wandb_project=wandb_project,
        wandb_exp_name=wandb_exp_name,
        expert_model_parallelism=1,
        enable_deepep=False,
        comm_overlap_config=CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=False,
        ),
    )

    # Model's seq_length must match dataset's (the recipe doesn't sync it).
    cfg.model.seq_length = args.seq_length

    # Recipe defaults log_interval=10; surface every step to wandb/stdout.
    cfg.logger.log_interval = args.log_interval

    # Swap default SQuAD for our HF-chat JSONL at data/sft/.
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=str(args.data_root),
        seq_length=args.seq_length,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            # Chat collate only emits the `attention_mask` key when this is
            # False. get_batch() in gpt_step.py reads it unconditionally, so
            # leaving the default (True) triggers KeyError: 'attention_mask'.
            "get_attention_mask_from_fusion": False,
        },
        do_test=False,
    )

    # With EP=1 there is no expert-alltoall to perform, so force the simplest
    # dispatcher and turn off any overlap that could lazy-allocate NVLink IPC
    # buffers.
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    if hasattr(cfg.model, "moe_flex_dispatcher_backend"):
        cfg.model.moe_flex_dispatcher_backend = None

    # Override the recipe's default LoRA (all linear layers incl. mamba
    # in/out_proj) with user-requested shape.
    if isinstance(args.peft, str) and args.peft.lower() == "lora":
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        cfg.peft = LoRA(
            target_modules=target_modules,
            dim=args.lora_r,
            alpha=args.lora_alpha,
        )

    # Point at the pre-downloaded snapshot if present.
    cfg.checkpoint.pretrained_checkpoint = pretrained
    cfg.checkpoint.save = str(out / "checkpoints")
    cfg.checkpoint.load = str(out / "checkpoints")

    print("========== Full SFT (Megatron-Bridge, nano-v3) ==========")
    print(f"  model:        {spec.key}")
    print(f"  pretrained:   {pretrained}")
    print(f"  data_root:    {args.data_root}")
    print(f"  output:       {out}")
    print(f"  epochs:       {args.num_epochs}  → train_iters={train_iters}")
    print(f"  lr:           {args.lr} (warmup={args.lr_warmup_iters}, decay→{min_lr})")
    print(f"  gbs/mbs:      {args.global_batch_size}/{args.micro_batch_size}")
    print(f"  seq_length:   {args.seq_length}")
    print(f"  dispatcher:   alltoall (DeepEP disabled for PCIe)")
    print(f"  peft:         {args.peft}"
          + (f" (r={args.lora_r}, alpha={args.lora_alpha}, targets={args.lora_target_modules})"
             if isinstance(args.peft, str) and args.peft.lower() == 'lora' else ''))
    print(f"  wandb:        {wandb_project or '(off)'} exp={wandb_exp_name}")
    print("=========================================================")

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
