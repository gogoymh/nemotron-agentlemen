#!/usr/bin/env bash
# Start a detached container based on the NeMo image for inference with an
# upgraded vLLM. The upgraded vLLM is installed into a separate venv at
# /tmp/vllm-new inside the container so the base install stays intact.
#
# Usage:
#   bash experiment/start_infer_ctr.sh                     # default name: nemotron-infer
#   NAME=my-infer bash experiment/start_infer_ctr.sh
#
# After startup:
#   docker exec -it nemotron-infer bash
#   # inside:
#   source /tmp/vllm-new/bin/activate
#   python -m src.evals.eval_on_test --model artifacts/sft/nemotron-30b/hf-iter_0001875 \
#     --backend vllm --gpu 0 --out-dir results/eval/iter1875
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

NAME="${NAME:-nemotron-infer}"
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.11.nemotron_3_nano}"
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.12.0}"
VENV_DIR="${VENV_DIR:-/tmp/vllm-new}"

ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

# Kill stale container with same name (idempotent).
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[infer] removing existing container $NAME"
    docker rm -f "$NAME" >/dev/null
fi

echo "[infer] starting detached container '$NAME' from $IMAGE"
docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --gpus all --ipc=host --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO_ROOT":/workdir \
    -e HF_HOME=/workdir/.cache/hf \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    -p 8000:8000 \
    --entrypoint bash \
    "$IMAGE" \
    -c "sleep infinity"

echo "[infer] container up. Installing '$VLLM_SPEC' into $VENV_DIR (this takes a few minutes)..."
docker exec "$NAME" bash -c "
    set -e
    python -m venv --system-site-packages $VENV_DIR
    $VENV_DIR/bin/pip install --upgrade pip
    $VENV_DIR/bin/pip install --upgrade '$VLLM_SPEC'
    $VENV_DIR/bin/python -c 'import vllm; print(\"[infer] vllm upgraded to\", vllm.__version__)'
"

echo ""
echo "done. To enter the container:"
echo "    docker exec -it $NAME bash"
echo ""
echo "Inside the container, activate the upgraded vLLM venv with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "Then run your eval / vllm serve commands."
