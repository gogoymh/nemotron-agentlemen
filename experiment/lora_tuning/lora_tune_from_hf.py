"""LoRA tuning entrypoint that mirrors convert_hf_to_megatron + sft.sh defaults.

This script intentionally follows the same conversion behavior as
`experiment/convert_hf_to_megatron.py`:

1) Ensure an HF snapshot exists.
2) Convert HF snapshot -> Megatron dist checkpoint with AutoBridge.import_ckpt.
3) Launch LoRA finetuning from that Megatron checkpoint.

Defaults match the current Docker SFT flow:
  - model: NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  - dataset: data/sft/{training,validation}.jsonl
  - hyperparameters: seq=1024, gbs=8, mbs=1, epochs=5, lr=1e-5,
    lora_r=32, lora_alpha=64, lora_target_modules=linear_qkv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_finetune_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


DEFAULT_HF_CKPT = Path("/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
DEFAULT_MEGATRON_CKPT = Path("/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-megatron")
DEFAULT_OUTPUT_DIR = Path("/ephemeral/nemotron-agentlemen_artifacts/sft/nemotron-30b")


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def bind_cuda_device_from_local_rank() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        torch.cuda.set_device(int(local_rank))
    except ValueError:
        pass


def ensure_megatron_checkpoint(hf_path: Path, megatron_path: Path) -> None:
    marker = megatron_path / "latest_checkpointed_iteration.txt"
    if marker.exists():
        print(f"using existing Megatron dist_checkpoint: {megatron_path}")
        return

    if not hf_path.is_dir():
        raise SystemExit(f"HF snapshot not found: {hf_path}")

    megatron_path.mkdir(parents=True, exist_ok=True)
    print("converting HF snapshot -> Megatron dist_checkpoint")
    print(f"  hf:       {hf_path}")
    print(f"  megatron: {megatron_path}")
    AutoBridge.import_ckpt(
        str(hf_path),
        str(megatron_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print("conversion done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=Path, default=DEFAULT_HF_CKPT)
    parser.add_argument("--megatron-path", type=Path, default=DEFAULT_MEGATRON_CKPT)
    parser.add_argument("--data-root", type=Path, default=Path("data/sft"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--lr-warmup-iters", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1)

    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=1)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-target-modules", type=str, default="linear_qkv")

    parser.add_argument("--wandb-project", type=str, default="nemotron-agentlemen-sft")
    parser.add_argument("--wandb-exp-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bind_cuda_device_from_local_rank()

    train_path = args.data_root / "training.jsonl"
    val_path = args.data_root / "validation.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"missing training/validation JSONL under: {args.data_root}")

    ensure_megatron_checkpoint(args.hf_path, args.megatron_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_steps > 0:
        train_iters = args.max_steps
    else:
        n_train = count_jsonl_lines(train_path)
        iters_per_epoch = max(1, math.ceil(n_train / args.global_batch_size))
        train_iters = args.num_epochs * iters_per_epoch
        print(
            f"(derived) n_train={n_train} gbs={args.global_batch_size} "
            f"-> iters_per_epoch={iters_per_epoch} x epochs={args.num_epochs} "
            f"= train_iters={train_iters}"
        )

    eval_interval = args.eval_interval or max(1, train_iters // max(args.num_epochs, 1))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10.0

    wandb_project = None if args.no_wandb or not os.environ.get("WANDB_API_KEY") else args.wandb_project
    wandb_exp_name = args.wandb_exp_name or "sft-nemotron-30b"

    cfg = nemotron_3_nano_finetune_config(
        peft="lora",
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
        dir=str(output_dir),
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

    cfg.model.seq_length = args.seq_length
    cfg.logger.log_interval = args.log_interval
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=str(args.data_root),
        seq_length=args.seq_length,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "get_attention_mask_from_fusion": False,
        },
        do_test=False,
    )

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = True
    if hasattr(cfg.model, "moe_flex_dispatcher_backend"):
        cfg.model.moe_flex_dispatcher_backend = None

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    cfg.peft = LoRA(
        target_modules=target_modules,
        dim=args.lora_r,
        alpha=args.lora_alpha,
    )

    cfg.checkpoint.pretrained_checkpoint = str(args.megatron_path)
    cfg.checkpoint.save = str(output_dir / "checkpoints")
    cfg.checkpoint.load = str(output_dir / "checkpoints")

    print("========== LoRA SFT (HF -> Megatron -> finetune) ==========")
    print(f"hf checkpoint:      {args.hf_path}")
    print(f"megatron checkpoint:{args.megatron_path}")
    print(f"data root:          {args.data_root}")
    print(f"output:             {output_dir}")
    print(f"epochs/iters:       {args.num_epochs}/{train_iters}")
    print(f"lr:                 {args.lr} (warmup={args.lr_warmup_iters}, min={min_lr})")
    print(f"seq/gbs/mbs:        {args.seq_length}/{args.global_batch_size}/{args.micro_batch_size}")
    print(f"lora:               r={args.lora_r}, alpha={args.lora_alpha}, targets={target_modules}")
    print(f"wandb:              {wandb_project or '(off)'} exp={wandb_exp_name}")
    print("============================================================")

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
