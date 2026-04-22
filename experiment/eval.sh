#!/usr/bin/env bash
# Evaluate the SFT checkpoint on data/multi-platform/test.csv via nemo-evaluator,
# run inside the nemotron-agentlemen Docker image.
#
# Prereqs:
#   - `experiment/build_docker.sh` has been run at least once.
#   - A trained checkpoint exported as HF weights.
#   - vLLM (or any OpenAI-compatible server) serving it at $MODEL_URL on the
#     host (the container is launched with --network host to reach it).
#
# Flow:
#   1) stage test.csv → BYOB test.jsonl
#   2) compile BYOB benchmark
#   3) nemo-evaluator run_eval → results/SFT/
#   4) report.py → results/SFT/summary.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nemotron-agentlemen:latest}"
MODEL_URL="${MODEL_URL:-http://localhost:8000}"
MODEL_ID="${MODEL_ID:-nemotron-30b-sft}"
MODEL_TYPE="${MODEL_TYPE:-chat}"
API_KEY_NAME="${API_KEY_NAME:-OPENAI_API_KEY}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/SFT}"

ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Build it first:"
    echo "   experiment/build_docker.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[1/4] staging test.csv → BYOB test.jsonl ..."
docker run --rm \
    -v "$REPO_ROOT":/workspace/project \
    "${ENV_FILE_ARG[@]}" \
    -w /workspace/project \
    "$IMAGE" \
    python -m src.evals.product_matching.prepare_data

echo "[2/4] compiling BYOB benchmark ..."
docker run --rm \
    -v "$REPO_ROOT":/workspace/project \
    "${ENV_FILE_ARG[@]}" \
    -w /workspace/project \
    "$IMAGE" \
    nemo-evaluator-byob src/evals/product_matching/benchmark.py

echo "[3/4] running nemo-evaluator against ${MODEL_URL} ..."
docker run --rm --network host \
    -v "$REPO_ROOT":/workspace/project \
    -v /ephemeral:/ephemeral \
    "${ENV_FILE_ARG[@]}" \
    -w /workspace/project \
    "$IMAGE" \
    nemo-evaluator run_eval \
        --eval_type byob_product_matching.product_matching \
        --model_url "$MODEL_URL" \
        --model_id "$MODEL_ID" \
        --model_type "$MODEL_TYPE" \
        --output_dir "$OUTPUT_DIR" \
        --api_key_name "$API_KEY_NAME"

echo "[4/4] post-hoc P/R/F1 report ..."
docker run --rm \
    -v "$REPO_ROOT":/workspace/project \
    -w /workspace/project \
    "$IMAGE" \
    python -m src.evals.product_matching.report \
        "$OUTPUT_DIR" \
        --out-summary "$OUTPUT_DIR/summary.json"

echo "done — results at $OUTPUT_DIR"
