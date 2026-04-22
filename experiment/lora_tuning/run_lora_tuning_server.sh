#!/usr/bin/env bash
# Run LoRA tuning directly on a server (no Docker).
# Uses the same defaults as experiment/sft.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

NPROC="${NPROC:-8}"

HF_CKPT="${HF_CKPT:-/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-megatron}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/sft}"
OUT_DIR="${OUT_DIR:-/ephemeral/nemotron-agentlemen_artifacts/sft/nemotron-30b}"

NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-1e-5}"
SEQ_LEN="${SEQ_LEN:-1024}"
GBS="${GBS:-8}"
MBS="${MBS:-1}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGETS="${LORA_TARGETS:-linear_qkv}"

WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-sft}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-sft-nano30b-$(date +%Y%m%d-%H%M%S)}"

# Keep this aligned with the existing training flow.
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,TUNING}"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

export PYTHONPATH="$REPO_ROOT/Megatron-Bridge/src:$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/ephemeral/hf_cache}"
export NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE
export NCCL_PXN_DISABLE NCCL_COLLNET_ENABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS

mkdir -p "$OUT_DIR"

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
