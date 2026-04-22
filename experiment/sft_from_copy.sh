#!/usr/bin/env bash
# Launcher for experiment/sft_from_copy.py — copy.md flow (convert + torchrun).
#
# Auto-detects execution context:
#   * Host  → wraps both steps in `docker run ... nvcr.io/nvidia/nemo:...`
#   * Inside NeMo container (no docker binary, or /.dockerenv present)
#             → runs conversion + torchrun directly in the current env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.11.nemotron_3_nano}"
NPROC="${NPROC:-8}"
GBS="${GBS:-16}"
MBS="${MBS:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-1e-5}"
PEFT="${PEFT:-none}"   # 'none' | 'lora' | 'dora'

HF_REPO_ID="${HF_REPO_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"

# All paths live under the repo (copy.md style: ./megatron_checkpoints/...).
HF_CKPT="${HF_CKPT:-$REPO_ROOT/.cache/hf_snapshot/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MEGATRON_CKPT="${MEGATRON_CKPT:-$REPO_ROOT/megatron_checkpoints/nemotron_nano}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/sft/nemotron-30b}"
HF_HOME_DEFAULT="$REPO_ROOT/.cache/hf"

# Convert source: local HF snapshot dir if present, else HF repo id (copy.md
# AutoBridge.import_ckpt("nvidia/...") path — download via HF_TOKEN).
if [[ -d "$HF_CKPT" ]]; then
    CONVERT_SRC="$HF_CKPT"
else
    CONVERT_SRC="$HF_REPO_ID"
fi

WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-sft}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-sft-copy-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT_DIR"

# ── Detect: are we already inside the NeMo container? ────────────────────────
# `IN_CONTAINER=1` to force, `IN_CONTAINER=0` to force host mode.
if [[ -n "${IN_CONTAINER:-}" ]]; then
    IN_CTR="$IN_CONTAINER"
elif [[ -f /.dockerenv ]] || ! command -v docker >/dev/null 2>&1; then
    IN_CTR=1
else
    IN_CTR=0
fi

# ── Inside-container mode: run directly, no docker wrapping. ─────────────────
if [[ "$IN_CTR" == "1" ]]; then
    echo "[sft] detected in-container execution (skipping docker run)"

    # Load .env into current shell if present.
    if [[ -f "$REPO_ROOT/.env" ]]; then
        set -a; . "$REPO_ROOT/.env"; set +a
    fi

    # Point PYTHONPATH at our Megatron-Bridge submodule (if mounted) before
    # the container-bundled copy. If not present, just fall back to repo root.
    BRIDGE_SRC="$REPO_ROOT/Megatron-Bridge/src"
    if [[ -d "$BRIDGE_SRC" ]]; then
        export PYTHONPATH="${BRIDGE_SRC}:${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    else
        export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    fi
    export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
    mkdir -p "$HF_HOME"

    # NCCL guards for H100 PCIe (no cross-pair NVLink mesh).
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
    export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"

    # Step 1: one-time HF → Megatron dist_checkpoint.
    if [[ ! -f "$MEGATRON_CKPT/latest_checkpointed_iteration.txt" ]]; then
        echo "[convert] $CONVERT_SRC → $MEGATRON_CKPT (one-time)"
        torchrun --nproc_per_node=1 experiment/convert_hf_to_megatron.py \
            --hf-path "$CONVERT_SRC" \
            --megatron-path "$MEGATRON_CKPT"
    fi

    echo "[sft] launching ${NUM_EPOCHS} epochs, peft=${PEFT}, lr=${LR}, gbs=${GBS}"
    exec torchrun --nproc_per_node="$NPROC" experiment/sft_from_copy.py \
        --data-root data/sft \
        --pretrained "$MEGATRON_CKPT" \
        --output-dir "$OUT_DIR" \
        --num-epochs "$NUM_EPOCHS" \
        --global-batch-size "$GBS" \
        --micro-batch-size "$MBS" \
        --seq-length "$SEQ_LEN" \
        --lr "$LR" \
        --peft "$PEFT" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-exp-name "$WANDB_EXP_NAME" \
        "$@"
fi

# ── Host mode: wrap everything in docker run. ────────────────────────────────
ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Run: docker pull $IMAGE" >&2
    exit 1
fi

BRIDGE_PYTHONPATH="/workdir/Megatron-Bridge/src"

if [[ ! -f "$MEGATRON_CKPT/latest_checkpointed_iteration.txt" ]]; then
    echo "[convert] $CONVERT_SRC → $MEGATRON_CKPT (one-time)"
    docker run --rm --gpus all --ipc=host --privileged \
        --shm-size=64g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$REPO_ROOT":/workdir \
        -e "PYTHONPATH=${BRIDGE_PYTHONPATH}:/workdir" \
        -e HF_HOME=/workdir/.cache/hf \
        "${ENV_FILE_ARG[@]}" \
        -w /workdir \
        "$IMAGE" \
        torchrun --nproc_per_node=1 experiment/convert_hf_to_megatron.py \
            --hf-path "$CONVERT_SRC" \
            --megatron-path "$MEGATRON_CKPT"
fi

echo "[sft] launching ${NUM_EPOCHS} epochs, peft=${PEFT}, lr=${LR}, gbs=${GBS}"
docker run --rm --gpus all --ipc=host --privileged \
    --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO_ROOT":/workdir \
    -e "PYTHONPATH=${BRIDGE_PYTHONPATH}:/workdir" \
    -e HF_HOME=/workdir/.cache/hf \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_PXN_DISABLE=1 \
    -e NCCL_COLLNET_ENABLE=0 \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    "$IMAGE" \
    torchrun --nproc_per_node="$NPROC" experiment/sft_from_copy.py \
        --data-root data/sft \
        --pretrained "$MEGATRON_CKPT" \
        --output-dir "$OUT_DIR" \
        --num-epochs "$NUM_EPOCHS" \
        --global-batch-size "$GBS" \
        --micro-batch-size "$MBS" \
        --seq-length "$SEQ_LEN" \
        --lr "$LR" \
        --peft "$PEFT" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-exp-name "$WANDB_EXP_NAME" \
        "$@"
