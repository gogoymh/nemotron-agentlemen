#!/usr/bin/env bash
# Full SFT on data/sft with Nemotron-3-Nano-30B-A3B-BF16 inside the official
# NVIDIA NeMo container for Nemotron 3 Nano.
#
# Image: nvcr.io/nvidia/nemo:25.11.nemotron_3_nano (paired with Megatron-Bridge
# nano-v3 branch). Our Megatron-Bridge/ submodule is on origin/nano-v3 and
# mounted in via PYTHONPATH override so we use the exact API the recipe was
# written against.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.11.nemotron_3_nano}"
NPROC="${NPROC:-8}"
SEQ_LEN=1024
GBS="${GBS:-8}"
MBS="${MBS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-1e-5}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-sft}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-sft-nano30b-$(date +%Y%m%d-%H%M%S)}"

OUT_DIR="${OUT_DIR:-/ephemeral/nemotron-agentlemen_artifacts/sft/nemotron-30b}"
HF_CKPT="${HF_CKPT:-/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-megatron}"
mkdir -p "$OUT_DIR"

ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Pull it first:"
    echo "   docker pull $IMAGE"
    exit 1
fi

# Override the container-bundled megatron.bridge with our nano-v3 submodule
# (same API but the submodule is the ground truth for local edits/debug).
BRIDGE_PYTHONPATH="/workspace/project/Megatron-Bridge/src"

# Convert HF snapshot → Megatron dist_checkpoint once (Bridge won't load HF
# safetensors from `pretrained_checkpoint` directly).
if [[ ! -f "$MEGATRON_CKPT/latest_checkpointed_iteration.txt" ]]; then
    echo "converting HF → Megatron dist_checkpoint at $MEGATRON_CKPT (one-time, slow)"
    docker run --rm --gpus all --ipc=host --privileged \
        --shm-size=64g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$REPO_ROOT":/workspace/project \
        -v /ephemeral:/ephemeral \
        -e "PYTHONPATH=${BRIDGE_PYTHONPATH}:/workspace/project" \
        -e HF_HOME=/ephemeral/hf_cache \
        "${ENV_FILE_ARG[@]}" \
        -w /workspace/project \
        "$IMAGE" \
        torchrun --nproc_per_node=1 experiment/convert_hf_to_megatron.py \
            --hf-path "$HF_CKPT" \
            --megatron-path "$MEGATRON_CKPT"
fi

echo "launching SFT (${NUM_EPOCHS} epochs, lr=${LR}) with torchrun --nproc_per_node=${NPROC}"
# H100 PCIe topology (nvidia-smi topo -p2p r): P2P is only `OK` within NVLinked
# pairs (0-1, 2-3, 4-5, 6-7); every cross-pair slot is `NS` (Not Supported at
# the driver level). NCCL 2.28's CUmem/VMM allocator on CUDA 13 maps peer
# memory opportunistically even with `NCCL_P2P_DISABLE=1` and triggers
# `cudaErrorContained: Invalid access of peer GPU memory over nvlink` at the
# first cross-pair reduction. NCCL_CUMEM_ENABLE=0 forces the legacy allocator
# path that does not require peer mapping. DEBUG=INFO dumps topology so we can
# still see what NCCL chose if this run fails too.
docker run --rm --gpus all --ipc=host --privileged \
    --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO_ROOT":/workspace/project \
    -v /ephemeral:/ephemeral \
    -e "PYTHONPATH=${BRIDGE_PYTHONPATH}:/workspace/project" \
    -e HF_HOME=/ephemeral/hf_cache \
    -e NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
    -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    -e NCCL_CUMEM_ENABLE="$NCCL_CUMEM_ENABLE" \
    -e NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
    -e NCCL_PXN_DISABLE="$NCCL_PXN_DISABLE" \
    -e NCCL_COLLNET_ENABLE="$NCCL_COLLNET_ENABLE" \
    -e NCCL_DEBUG=INFO \
    -e NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING \
    "${ENV_FILE_ARG[@]}" \
    -w /workspace/project \
    "$IMAGE" \
    torchrun --nproc_per_node="$NPROC" src/train_sft.py \
        --num-epochs "$NUM_EPOCHS" \
        --lr "$LR" \
        --global-batch-size "$GBS" \
        --micro-batch-size "$MBS" \
        --seq-length "$SEQ_LEN" \
        --output-dir "$OUT_DIR" \
        --pretrained-checkpoint "$MEGATRON_CKPT" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-exp-name "$WANDB_EXP_NAME" \
        --peft lora \
        --lora-r 32 \
        --lora-alpha 64 \
        --lora-target-modules linear_qkv \
        "$@"
