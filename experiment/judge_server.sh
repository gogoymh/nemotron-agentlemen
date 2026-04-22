#!/usr/bin/env bash
# vLLM OpenAI-compatible judge server for product-match RL rewards.
#
# Hosts Nemotron-3-Super-120B on GPUs 4–7 (TP=4). The training run on 0–3
# queries this server via HTTP for LLM-as-judge scoring — see
# experiment/rl_reward_env.py.
#
# Served model name (what clients send in the `model` field):
#   nemotron-3-super-120b
# This alias must match the JUDGE_MODEL env/config the RL launcher sends;
# both default to "nemotron-3-super-120b".
#
# Usage:
#   bash experiment/judge_server.sh              # detached background container
#   JUDGE_PORT=8001 bash experiment/judge_server.sh
#
# Health check:
#   curl -s http://localhost:${JUDGE_PORT:-8000}/v1/models | jq
#
# Stop:
#   docker stop nemotron-vllm && docker rm nemotron-vllm
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-vllm}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
JUDGE_MODEL="${JUDGE_MODEL:-nemotron-3-super-120b}"   # --served-model-name alias
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_GPUS="${JUDGE_GPUS:-4,5,6,7}"
JUDGE_TP="${JUDGE_TP:-4}"

# Load HF_TOKEN / WANDB_API_KEY / etc from .env so private HF gated models load.
[[ -f "$REPO_ROOT/.env" ]] && { set -a; . "$REPO_ROOT/.env"; set +a; }

# Idempotent: if the container already runs, just print its state.
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[judge] '$CONTAINER_NAME' already running."
    docker ps --filter "name=^${CONTAINER_NAME}$"
    exit 0
fi
# If a stopped container with the same name exists, remove it first.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[judge] removing stale container '$CONTAINER_NAME'"
    docker rm "$CONTAINER_NAME" >/dev/null
fi

echo "[judge] launching $CONTAINER_NAME"
echo "[judge]   weights  : $JUDGE_MODEL_PATH"
echo "[judge]   alias    : $JUDGE_MODEL"
echo "[judge]   gpus     : $JUDGE_GPUS  (TP=$JUDGE_TP)"
echo "[judge]   port     : $JUDGE_PORT"

docker run -d --gpus "\"device=$JUDGE_GPUS\"" \
    --name "$CONTAINER_NAME" \
    --init \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -p "$JUDGE_PORT:8000" \
    --ipc=host \
    --shm-size=16g \
    "$IMAGE" \
    --model "$JUDGE_MODEL_PATH" \
    --tensor-parallel-size "$JUDGE_TP" \
    --served-model-name "$JUDGE_MODEL"

echo
echo "[judge] container started. Tail logs with:"
echo "        docker logs -f $CONTAINER_NAME"
echo "[judge] health check:"
echo "        curl -s http://localhost:$JUDGE_PORT/v1/models | jq"
