#!/usr/bin/env bash
# Run LoRA tuning in Docker with the same model, dataset, and hyperparameters
# as experiment/sft.sh, while performing HF -> Megatron checkpoint conversion
# inside the Python entrypoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.11.nemotron_3_nano}"
NPROC="${NPROC:-8}"

HF_CKPT="${HF_CKPT:-/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-megatron}"
DATA_ROOT="${DATA_ROOT:-/workspace/project/data/sft}"
OUT_DIR="${OUT_DIR:-/ephemeral/nemotron-agentlemen_artifacts/sft/nemotron-30b}"

NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-1e-5}"
SEQ_LEN="${SEQ_LEN:-1024}"
GBS="${GBS:-8}"
MBS="${MBS:-1}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGETS="${LORA_TARGETS:-linear_qkv}"

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-sft}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-sft-nano30b-$(date +%Y%m%d-%H%M%S)}"

ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Pull it first:"
    echo "   docker pull $IMAGE"
    exit 1
fi

BRIDGE_PYTHONPATH="/workspace/project/Megatron-Bridge/src"

docker run --rm --gpus all --ipc=host --privileged \
    --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO_ROOT":/workspace/project \
    -v /ephemeral:/ephemeral \
    -e "PYTHONPATH=${BRIDGE_PYTHONPATH}:/workspace/project" \
    -e HF_HOME=/ephemeral/hf_cache \
    -e NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
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
    torchrun --nproc_per_node="$NPROC" experiment/lora_tuning/lora_tune_from_hf.py \
        --hf-path "$HF_CKPT" \
        --megatron-path "$MEGATRON_CKPT" \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUT_DIR" \
        --num-epochs "$NUM_EPOCHS" \
        --lr "$LR" \
        --seq-length "$SEQ_LEN" \
        --global-batch-size "$GBS" \
        --micro-batch-size "$MBS" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-target-modules "$LORA_TARGETS" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-exp-name "$WANDB_EXP_NAME" \
        "$@"
