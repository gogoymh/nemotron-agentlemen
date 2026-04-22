#!/usr/bin/env bash
# Export a merged Megatron dist_checkpoint to HuggingFace safetensors so it can
# be loaded by vLLM / transformers (e.g. for src/evals/eval_on_test.py).
#
# Mirrors copy.md L23-26:
#   python /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py export \
#     --hf-model <hf id>                 # provides architecture/tokenizer config
#     --megatron-path <megatron dir>     # merged dist_checkpoint from merge_lora.sh
#     --hf-path <out dir>                # resulting HF safetensors
#
# Usage:
#   ITER=1875 bash experiment/export_to_hf.sh
#   MEGATRON_DIR=... HF_OUT=... bash experiment/export_to_hf.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.11.nemotron_3_nano}"
HF_MODEL="${HF_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-/opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py}"

ITER="${ITER:-1875}"
ITER_PAD="$(printf 'iter_%07d' "$ITER")"
MEGATRON_DIR="${MEGATRON_DIR:-$REPO_ROOT/artifacts/sft/nemotron-30b/merged-$ITER_PAD}"
HF_OUT="${HF_OUT:-$REPO_ROOT/artifacts/sft/nemotron-30b/hf-$ITER_PAD}"

if [[ ! -d "$MEGATRON_DIR" ]]; then
    echo "ERROR: megatron dir not found: $MEGATRON_DIR" >&2
    exit 1
fi

mkdir -p "$HF_OUT"

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

    echo "[export] megatron = $MEGATRON_DIR"
    echo "[export] hf-model = $HF_MODEL  (architecture + tokenizer)"
    echo "[export] out      = $HF_OUT"
    torchrun --nproc_per_node=1 "$EXPORT_SCRIPT" export \
        --hf-model "$HF_MODEL" \
        --megatron-path "$MEGATRON_DIR" \
        --hf-path "$HF_OUT"
    # vLLM's NemotronH loader reads config.rms_norm_eps but the HF config only
    # exports layer_norm_epsilon. Add the alias so vLLM can load it.
    python3 -c "
import json, pathlib
p = pathlib.Path('$HF_OUT/config.json')
cfg = json.loads(p.read_text())
if 'rms_norm_eps' not in cfg:
    cfg['rms_norm_eps'] = cfg.get('layer_norm_epsilon', 1e-5)
    p.write_text(json.dumps(cfg, indent=2))
    print('[export] patched config.json with rms_norm_eps =', cfg['rms_norm_eps'])
"
    exit 0
fi

# Host mode.
ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. docker pull $IMAGE" >&2
    exit 1
fi

echo "[export] megatron = $MEGATRON_DIR"
echo "[export] hf-model = $HF_MODEL"
echo "[export] out      = $HF_OUT"

# Translate host paths under $REPO_ROOT to the container's /workdir mount.
_to_container() { echo "${1/#$REPO_ROOT/\/workdir}"; }
MEGATRON_DIR_C="$(_to_container "$MEGATRON_DIR")"
HF_OUT_C="$(_to_container "$HF_OUT")"

docker run --rm --gpus all --ipc=host --shm-size=16g \
    -v "$REPO_ROOT":/workdir \
    -e HF_HOME=/workdir/.cache/hf \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    "$IMAGE" \
    bash -c "
        torchrun --nproc_per_node=1 '$EXPORT_SCRIPT' export \
            --hf-model '$HF_MODEL' \
            --megatron-path '$MEGATRON_DIR_C' \
            --hf-path '$HF_OUT_C' && \
        python3 -c \"
import json, pathlib
p = pathlib.Path('$HF_OUT_C/config.json')
cfg = json.loads(p.read_text())
if 'rms_norm_eps' not in cfg:
    cfg['rms_norm_eps'] = cfg.get('layer_norm_epsilon', 1e-5)
    p.write_text(json.dumps(cfg, indent=2))
    print('[export] patched config.json with rms_norm_eps =', cfg['rms_norm_eps'])
\"
    "

echo "done — HF safetensors at $HF_OUT"
