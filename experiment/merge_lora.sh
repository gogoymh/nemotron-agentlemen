#!/usr/bin/env bash
# Merge a Megatron-Bridge LoRA checkpoint back into the base model, then
# (by default) export to HuggingFace safetensors so it can be served/evaluated.
#
# Two-stage pipeline (copy.md L146-149 + L23-26):
#   stage 1: merge_lora.py   LoRA + base   → merged Megatron dist_checkpoint
#   stage 2: convert_checkpoints.py export   Megatron           → HF safetensors
#
# Inputs  : ./artifacts/sft/nemotron-30b/checkpoints/iter_XXXXXXX  (torch_dist)
# Outputs : ./artifacts/sft/nemotron-30b/merged-iter_XXXXXXX   (Megatron, intermediate)
#           ./artifacts/sft/nemotron-30b/hf-iter_XXXXXXX       (HF safetensors, final)
#
# Usage:
#   ITER=1875 bash experiment/merge_lora.sh          # merge + export (default)
#   ITER=1875 NO_EXPORT=1 bash experiment/merge_lora.sh   # merge only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.11.nemotron_3_nano}"
HF_MODEL="${HF_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MERGE_SCRIPT="${MERGE_SCRIPT:-/opt/Megatron-Bridge/examples/peft/merge_lora.py}"

ITER="${ITER:-500}"
ITER_PAD="$(printf 'iter_%07d' "$ITER")"
LORA_CKPT="${LORA_CKPT:-$REPO_ROOT/artifacts/sft/nemotron-30b/checkpoints/$ITER_PAD}"
OUT="${OUT:-$REPO_ROOT/artifacts/sft/nemotron-30b/merged-$ITER_PAD}"

if [[ ! -d "$LORA_CKPT" ]]; then
    echo "ERROR: LoRA checkpoint not found: $LORA_CKPT" >&2
    echo "Available iter_ dirs:" >&2
    ls "$REPO_ROOT/artifacts/sft/nemotron-30b/checkpoints/" 2>/dev/null | grep "^iter_" >&2 || true
    exit 1
fi

mkdir -p "$OUT"

# Detect host vs in-container.
if [[ -n "${IN_CONTAINER:-}" ]]; then
    IN_CTR="$IN_CONTAINER"
elif [[ -f /.dockerenv ]] || ! command -v docker >/dev/null 2>&1; then
    IN_CTR=1
else
    IN_CTR=0
fi

if [[ "$IN_CTR" == "1" ]]; then
    [[ -f "$REPO_ROOT/.env" ]] && { set -a; . "$REPO_ROOT/.env"; set +a; }
    export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

    echo "[merge] base=$HF_MODEL"
    echo "[merge] lora=$LORA_CKPT"
    echo "[merge] out =$OUT"
    # merge_lora.py calls torch.distributed.init_process_group("nccl") which
    # needs RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT, so always go through
    # torchrun (single proc is fine).
    if [[ ! -f "$OUT/iter_0000000/__0_0.distcp" ]]; then
        torchrun --nproc_per_node=1 "$MERGE_SCRIPT" \
            --hf-model-path "$HF_MODEL" \
            --lora-checkpoint "$LORA_CKPT" \
            --output "$OUT"
    else
        echo "[merge] skipping — merged checkpoint already present"
    fi
    if [[ "${NO_EXPORT:-0}" != "1" ]]; then
        ITER="$ITER" MEGATRON_DIR="$OUT" \
            HF_OUT="${HF_OUT:-$REPO_ROOT/artifacts/sft/nemotron-30b/hf-$ITER_PAD}" \
            IN_CONTAINER=1 \
            bash "$REPO_ROOT/experiment/export_to_hf.sh"
    fi
    exit 0
fi

# Host mode.
ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. docker pull $IMAGE" >&2
    exit 1
fi

echo "[merge] base=$HF_MODEL"
echo "[merge] lora=$LORA_CKPT"
echo "[merge] out =$OUT"

# Translate host paths under $REPO_ROOT to the container's /workdir mount.
_to_container() { echo "${1/#$REPO_ROOT/\/workdir}"; }
LORA_CKPT_C="$(_to_container "$LORA_CKPT")"
OUT_C="$(_to_container "$OUT")"

if [[ ! -f "$OUT/iter_0000000/__0_0.distcp" ]]; then
    docker run --rm --gpus all --ipc=host --shm-size=16g \
        -v "$REPO_ROOT":/workdir \
        -e HF_HOME=/workdir/.cache/hf \
        "${ENV_FILE_ARG[@]}" \
        -w /workdir \
        "$IMAGE" \
        torchrun --nproc_per_node=1 "$MERGE_SCRIPT" \
            --hf-model-path "$HF_MODEL" \
            --lora-checkpoint "$LORA_CKPT_C" \
            --output "$OUT_C"
else
    echo "[merge] skipping — merged checkpoint already present"
fi

echo "[merge] merged Megatron checkpoint at $OUT"

if [[ "${NO_EXPORT:-0}" != "1" ]]; then
    HF_OUT_DEFAULT="$REPO_ROOT/artifacts/sft/nemotron-30b/hf-$ITER_PAD"
    ITER="$ITER" MEGATRON_DIR="$OUT" HF_OUT="${HF_OUT:-$HF_OUT_DEFAULT}" \
        bash "$REPO_ROOT/experiment/export_to_hf.sh"
fi
