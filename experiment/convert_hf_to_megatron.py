"""One-off: convert an HF snapshot to a Megatron-Bridge distributed checkpoint.

Bridge's `pretrained_checkpoint` path expects a native Megatron dist_checkpoint
(looks for `latest_checkpointed_iteration.txt`), not raw HF safetensors, so the
first training launch would otherwise hit `ValueError: Invalid pretrained
checkpoint directory`.

`--hf-path` can be either:
  * a local directory with an HF snapshot, or
  * an HF repo id (e.g. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`), in which
    case AutoBridge will download it using `HF_HOME`/`HF_TOKEN`.

Run inside the NeMo container (same image as SFT):

    torchrun --nproc_per_node=1 experiment/convert_hf_to_megatron.py \
        --hf-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --megatron-path ./megatron_checkpoints/nemotron_nano
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from megatron.bridge.models.conversion.auto_bridge import AutoBridge


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-path", type=str, required=True,
                   help="Local HF snapshot dir OR HF repo id (e.g. "
                        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).")
    p.add_argument("--megatron-path", type=Path, required=True)
    args = p.parse_args()

    hf_path_obj = Path(args.hf_path)
    if hf_path_obj.is_dir():
        hf_arg = str(hf_path_obj)
        kind = "local dir"
    elif "/" in args.hf_path and not args.hf_path.startswith("/"):
        # Looks like an HF repo id (`org/name`). Let AutoBridge pull it.
        hf_arg = args.hf_path
        kind = "HF repo id (will download)"
    else:
        raise SystemExit(
            f"HF path not found and does not look like a repo id: {args.hf_path}"
        )

    args.megatron_path.mkdir(parents=True, exist_ok=True)

    print(f"converting HF snapshot → Megatron dist_checkpoint")
    print(f"  hf:       {hf_arg}  ({kind})")
    print(f"  megatron: {args.megatron_path}")

    AutoBridge.import_ckpt(
        hf_arg,
        str(args.megatron_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("done.")


if __name__ == "__main__":
    main()
