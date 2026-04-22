#!/usr/bin/env bash
# Serve a merged RL checkpoint (bf16 HF safetensors) via vLLM so eval.sh can
# hit it over OpenAI-compatible HTTP.
#
# Inputs:
#   HF_MODEL_DIR : merged HF model dir from merge_rl_lora.sh
#                  default: /mnt/data/artifacts/rl/nemotron-30b-trl/hf-checkpoint-$STEP
#   VLLM_TP      : tensor-parallel size (default 4)
#   VLLM_PORT    : server port (default 8000 — matches eval.sh default)
#   VLLM_GPUS    : which GPUs to use (default 0,1,2,3)
#
# Usage:
#   STEP=200 bash experiment/serve_rl.sh            # foreground
#   STEP=200 bash experiment/serve_rl.sh -d         # background (writes pid)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano}"

STEP="${STEP:-200}"
HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/data/artifacts/rl/nemotron-30b-trl/hf-checkpoint-$STEP}"
MODEL_ID="${MODEL_ID:-nemotron-30b-rl-step$STEP}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/artifacts/rl/nemotron-30b-trl/logs}"
LOG_FILE="$LOG_DIR/vllm_serve_step$STEP.log"
LOG_FILE_C="${LOG_FILE/#$REPO_ROOT/\/workdir}"  # container-side path
PID_FILE="$LOG_DIR/vllm_serve_step$STEP.pid"
CONTAINER_NAME="${CONTAINER_NAME:-rl-vllm-serve-step$STEP}"

if [[ ! -f "$HF_MODEL_DIR/config.json" ]]; then
    echo "ERROR: merged HF model not found at $HF_MODEL_DIR" >&2
    echo "       run 'STEP=$STEP bash experiment/merge_rl_lora.sh' first" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

# Detached flag.
DETACH=""
if [[ "${1:-}" == "-d" || "${1:-}" == "--detach" ]]; then
    DETACH="--detach"
fi

echo "[serve-rl] model    = $HF_MODEL_DIR"
echo "[serve-rl] served-id= $MODEL_ID"
echo "[serve-rl] GPUs     = $VLLM_GPUS  (TP=$VLLM_TP)"
echo "[serve-rl] port     = $VLLM_PORT"
echo "[serve-rl] log      = $LOG_FILE"

# vLLM lives in the per-Ray-worker venv of the NeMo-RL image; reuse that binary.
VLLM_BIN="/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/vllm"

CMD=(
    "$VLLM_BIN" serve "$HF_MODEL_DIR"
    --served-model-name "$MODEL_ID"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --tensor-parallel-size "$VLLM_TP"
    --gpu-memory-utilization "$VLLM_GPU_MEM"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --dtype bfloat16
    --trust-remote-code
    --enable-prefix-caching
)

if [[ -n "$DETACH" ]]; then
    docker run -d --gpus all --ipc=host --privileged \
        --shm-size=64g --network host --name "$CONTAINER_NAME" \
        -v "$REPO_ROOT":/workdir \
        -v /mnt/data:/mnt/data \
        -e HF_HOME=/workdir/.cache/hf \
        -e CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
        "${ENV_FILE_ARG[@]}" \
        -w /workdir \
        "$IMAGE" \
        bash -c "${CMD[*]} > $LOG_FILE_C 2>&1"
    echo "[serve-rl] detached — tail with: sudo docker logs -f $CONTAINER_NAME"
    echo "[serve-rl] stop    with: sudo docker stop $CONTAINER_NAME"
else
    docker run --rm --gpus all --ipc=host --privileged \
        --shm-size=64g --network host --name "$CONTAINER_NAME" \
        -v "$REPO_ROOT":/workdir \
        -v /mnt/data:/mnt/data \
        -e HF_HOME=/workdir/.cache/hf \
        -e CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
        "${ENV_FILE_ARG[@]}" \
        -w /workdir \
        "$IMAGE" \
        "${CMD[@]}"
fi
