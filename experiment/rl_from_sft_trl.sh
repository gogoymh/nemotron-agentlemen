#!/usr/bin/env bash
# TRL-based GRPO launcher with native LoRA.
#
# Why this exists:
#   NeMo-RL v0.4.0 has no working LoRA path on the vLLM-colocated GRPO entry.
#   We use HuggingFace TRL's GRPOTrainer, which accepts `peft_config=LoraConfig(...)`.
#
# Usage:
#   sudo bash experiment/rl_from_sft_trl.sh
#
# Strategy:
#   - Base model : artifacts/sft/nemotron-30b/hf-iter_0001875
#   - LoRA       : r=32 α=64 dropout=0.05 on q/k/v_proj
#   - Sharding   : single process + HF device_map="auto" splits 30B across
#                  GPUs 0-3 (~15 GB/GPU bf16). LoRA adapters are the only
#                  trainables, so no DDP/ZeRO.
#   - Rollouts   : vLLM server on GPUs 4-7 (TP=4), started as a background
#                  subprocess by this script. Trainer talks to it via HTTP
#                  and pushes LoRA deltas between optimizer steps.
#   - Judge      : Azure OpenAI (gpt-5.4 deployment via APIM). Creds in .env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano}"
ACCEL_CFG="${ACCEL_CFG:-$REPO_ROOT/experiment/accelerate_zero3.yaml}"

# Base SFT-merged HF checkpoint.
ITER="${ITER:-1875}"
ITER_PAD="$(printf 'iter_%07d' "$ITER")"
SFT_CKPT="${SFT_CKPT:-$REPO_ROOT/artifacts/sft/nemotron-30b/hf-$ITER_PAD}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/rl/nemotron-30b-trl}"
RL_DATA_DIR="${RL_DATA_DIR:-$REPO_ROOT/data/rl}"
HF_HOME_DEFAULT="$REPO_ROOT/.cache/hf"

WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-rl}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-rl-from-sft-trl-$(date +%Y%m%d-%H%M%S)}"

# GRPO hyperparameters.
MAX_STEPS="${MAX_STEPS:-500}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-1}"
NUM_GENS_PER_PROMPT="${NUM_GENS_PER_PROMPT:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-$SEQ_LEN}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-512}"
LR="${LR:-5.0e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SAVE_PERIOD="${SAVE_PERIOD:-100}"
KL_PENALTY="${KL_PENALTY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"

# LoRA.
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj}"

# vLLM rollout server (out-of-process, on a separate GPU pool).
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8001}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.85}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((SEQ_LEN + MAX_COMPLETION_LEN))}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-1800}"  # model load can take a while

# The NeMo-RL image ships vLLM inside a per-Ray-worker venv instead of the
# main one. We install trl into this venv so `trl vllm-serve` can find vllm.
VLLM_VENV="${VLLM_VENV:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker}"
VLLM_VENV_PIP="${VLLM_VENV}/bin/pip"
VLLM_VENV_TRL="${VLLM_VENV}/bin/trl"

# Azure OpenAI judge. Credentials must live in .env.
AZURE_ENDPOINT="${AZURE_ENDPOINT:-}"
AZURE_API_KEY="${AZURE_API_KEY:-}"
AZURE_MODEL="${AZURE_MODEL:-}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2024-12-01-preview}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-32}"

# 8-GPU split: 0-3 for training (device_map=auto), 4-7 for vLLM rollout server.
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
VLLM_GPUS="${VLLM_GPUS:-4,5,6,7}"
ALL_GPUS="${ALL_GPUS:-0,1,2,3,4,5,6,7}"

# TRL/peft versions are pinned to a range the container's torch/vLLM support.
# TRL >= 0.15 is required for vllm_mode="colocate".
TRL_VERSION="${TRL_VERSION:-0.19.1}"
PEFT_VERSION="${PEFT_VERSION:-0.15.2}"
ACCEL_VERSION="${ACCEL_VERSION:-1.7.0}"
DS_VERSION="${DS_VERSION:-0.15.4}"

if [[ ! -d "$SFT_CKPT" || ! -f "$SFT_CKPT/config.json" ]]; then
    echo "ERROR: HF-format SFT checkpoint not found: $SFT_CKPT" >&2
    echo "       run the merge+export pipeline first:" >&2
    echo "         sudo ITER=$ITER bash experiment/merge_lora.sh" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$RL_DATA_DIR" "$OUT_DIR/generations" "$OUT_DIR/logs" "$OUT_DIR/checkpoints"

if [[ -n "${IN_CONTAINER:-}" ]]; then
    IN_CTR="$IN_CONTAINER"
elif [[ -f /.dockerenv ]] || ! command -v docker >/dev/null 2>&1; then
    IN_CTR=1
else
    IN_CTR=0
fi

check_judge() {
    if [[ -z "$AZURE_ENDPOINT" || -z "$AZURE_API_KEY" || -z "$AZURE_MODEL" ]]; then
        echo "ERROR: Azure judge config missing." >&2
        echo "       set AZURE_ENDPOINT / AZURE_API_KEY / AZURE_MODEL in .env" >&2
        exit 1
    fi
    # Real chat-completions probe. gpt-5.x requires `max_completion_tokens` (not
    # `max_tokens`) and needs ≥ ~8 tokens to emit a stop; setting it to 1 returns
    # HTTP 400 with "model output limit was reached". Using 16 matches runtime.
    local probe_url="${AZURE_ENDPOINT%/}/deployments/${AZURE_MODEL}/chat/completions?api-version=${AZURE_API_VERSION}"
    echo "[trl-rl] probing Azure judge at ${AZURE_ENDPOINT}"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
        -H "api-key: $AZURE_API_KEY" -X POST \
        -H "content-type: application/json" \
        -d '{"messages":[{"role":"user","content":"ping"}],"max_completion_tokens":16}' \
        "$probe_url" || echo 000)"
    if [[ "$code" =~ ^2 ]]; then
        echo "[trl-rl] Azure judge OK (HTTP $code)"
        return 0
    fi
    echo "ERROR: Azure judge probe failed (HTTP $code)" >&2
    exit 1
}

start_vllm_server() {
    local log_file="$OUT_DIR/logs/vllm_server.log"
    local pid_file="$OUT_DIR/logs/vllm_server.pid"
    : > "$log_file"

    echo "[trl-rl] starting TRL vLLM server on GPUs $VLLM_GPUS (TP=$VLLM_TP)"
    echo "[trl-rl]   host=$VLLM_SERVER_HOST  port=$VLLM_SERVER_PORT  gpu_mem=$VLLM_GPU_MEM  max_len=$VLLM_MAX_MODEL_LEN"
    echo "[trl-rl]   log : $log_file"

    # Separate CUDA_VISIBLE_DEVICES so vLLM sees only its 4 GPUs even though
    # the outer script is about to export 0-3 for training.
    # `trl vllm-serve`'s HfArgumentParser treats bool args as "flag expects
    # value" (argparse 'store' not 'store_true'). Pass True/False explicitly.
    # Use the Ray-venv trl binary so `import vllm` succeeds.
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
        nohup "$VLLM_VENV_TRL" vllm-serve \
            --model "$SFT_CKPT" \
            --tensor_parallel_size "$VLLM_TP" \
            --host "$VLLM_SERVER_HOST" \
            --port "$VLLM_SERVER_PORT" \
            --gpu_memory_utilization "$VLLM_GPU_MEM" \
            --max_model_len "$VLLM_MAX_MODEL_LEN" \
            --dtype bfloat16 \
            --enable_prefix_caching True \
            --trust_remote_code True \
            >"$log_file" 2>&1 &
    local pid=$!
    echo "$pid" >"$pid_file"

    VLLM_SERVER_PID="$pid"
    trap 'stop_vllm_server' EXIT INT TERM

    echo "[trl-rl] waiting for vLLM server (pid=$pid) to become ready (timeout=${VLLM_READY_TIMEOUT_S}s)"
    local deadline=$(( SECONDS + VLLM_READY_TIMEOUT_S ))
    local health_url="http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/health/"
    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: vLLM server died before becoming ready — see $log_file" >&2
            tail -n 80 "$log_file" >&2 || true
            exit 1
        fi
        if curl -s --max-time 3 "$health_url" >/dev/null 2>&1; then
            echo "[trl-rl] vLLM server ready"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM server didn't come up in ${VLLM_READY_TIMEOUT_S}s — see $log_file" >&2
    tail -n 80 "$log_file" >&2 || true
    kill "$pid" 2>/dev/null || true
    exit 1
}

stop_vllm_server() {
    if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
        echo "[trl-rl] stopping vLLM server (pid=$VLLM_SERVER_PID)"
        kill "$VLLM_SERVER_PID" 2>/dev/null || true
        wait "$VLLM_SERVER_PID" 2>/dev/null || true
        VLLM_SERVER_PID=""
    fi
}

install_trl_stack() {
    # The NeMo-RL container already has torch + flash-attn + transformers in
    # the main venv (`/opt/nemo_rl_venv`), but vLLM is installed only inside
    # per-Ray-worker venvs. We need trl in BOTH: the main venv for the trainer
    # to import `trl.GRPOTrainer`, and the Ray vLLM venv so `trl vllm-serve`
    # can `import vllm` from the same interpreter.
    echo "[trl-rl] installing trl=$TRL_VERSION peft=$PEFT_VERSION accelerate=$ACCEL_VERSION deepspeed=$DS_VERSION (main venv)"
    pip install --no-deps --quiet \
        "trl==$TRL_VERSION" \
        "peft==$PEFT_VERSION"
    pip install --quiet \
        "accelerate==$ACCEL_VERSION" \
        "deepspeed==$DS_VERSION" \
        "httpx>=0.27" \
        "bitsandbytes>=0.45.0"

    if [[ -x "$VLLM_VENV_PIP" ]]; then
        echo "[trl-rl] installing trl=$TRL_VERSION in vLLM Ray venv so vllm-serve can import vllm"
        "$VLLM_VENV_PIP" install --no-deps --quiet "trl==$TRL_VERSION" "peft==$PEFT_VERSION"
    else
        echo "ERROR: vLLM Ray venv not found at $VLLM_VENV_PIP — cannot run trl vllm-serve" >&2
        exit 1
    fi

    # Nemotron-H hybrid needs Mamba kernels. Install prebuilt wheels if
    # available; fall back to source build if not. `--no-build-isolation`
    # reuses the container's torch so the CUDA ext links to it.
    echo "[trl-rl] installing mamba-ssm + causal-conv1d (may build from source first time)"
    # Build only for H100 (sm_90) to cut ~3–4x off first-run compile time.
    # MAX_JOBS caps parallel nvcc — 8 keeps CPU mem sane on the H100 box.
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    pip install --no-build-isolation \
        "causal-conv1d>=1.4.0" \
        "mamba-ssm>=2.2.0"
}

run_prep_and_train() {
    echo "[trl-rl] preparing RL JSONL"
    python -m src.data_prep.csv_to_rl_jsonl --out-dir "$RL_DATA_DIR"

    check_judge
    install_trl_stack

    # Bring up vLLM rollout server on its 4 GPUs BEFORE restricting the training
    # process to the other 4. start_vllm_server sets its own CUDA_VISIBLE_DEVICES.
    start_vllm_server

    echo "[trl-rl] launching GRPO (TRL + LoRA, single-process + device_map=auto + vLLM server)"
    echo "[trl-rl]   base model   : $SFT_CKPT"
    echo "[trl-rl]   train GPUs   : $TRAIN_GPUS  (HF pipeline-parallel across 4 GPUs via device_map=auto)"
    echo "[trl-rl]   rollout GPUs : $VLLM_GPUS   (vLLM server TP=$VLLM_TP at $VLLM_SERVER_HOST:$VLLM_SERVER_PORT)"
    echo "[trl-rl]   judge        : Azure OpenAI $AZURE_MODEL @ $AZURE_ENDPOINT"
    echo "[trl-rl]   steps=$MAX_STEPS  prompts/step=$NUM_PROMPTS_PER_STEP  num_gens=$NUM_GENS_PER_PROMPT"
    echo "[trl-rl]   per-dev bs=$PER_DEVICE_BS  lr=$LR  clip=$GRAD_CLIP  kl=$KL_PENALTY  save=$SAVE_PERIOD"
    echo "[trl-rl]   LoRA         : r=$LORA_R α=$LORA_ALPHA dropout=$LORA_DROPOUT targets=$LORA_TARGETS"
    echo "[trl-rl]   per-step dump: $OUT_DIR/generations/generations.jsonl"

    export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
    export AZURE_ENDPOINT AZURE_API_KEY AZURE_MODEL AZURE_API_VERSION
    export VLLM_SERVER_HOST VLLM_SERVER_PORT
    export WANDB_PROJECT WANDB_EXP_NAME

    # Single-process: HF `device_map="auto"` pipeline-splits the 30B across
    # the 4 visible GPUs inside one Python process. Launching 4 ranks here
    # would replicate the model per rank (OOM) or force ZeRO-3 (generate()
    # gather OOM). See experiment/accelerate_zero3.yaml for the full reasoning.
    # Plain invocation (no exec) so the EXIT trap can reap the vLLM server.
    accelerate launch \
        --config_file "$ACCEL_CFG" \
        --num_processes 1 \
        "$REPO_ROOT/experiment/rl_from_sft_trl.py" \
            --model "$SFT_CKPT" \
            --train-jsonl "$RL_DATA_DIR/training.jsonl" \
            --val-jsonl "$RL_DATA_DIR/validation.jsonl" \
            --out-dir "$OUT_DIR" \
            --num-prompts-per-step "$NUM_PROMPTS_PER_STEP" \
            --num-generations "$NUM_GENS_PER_PROMPT" \
            --per-device-bs "$PER_DEVICE_BS" \
            --max-prompt-length "$MAX_PROMPT_LEN" \
            --max-completion-length "$MAX_COMPLETION_LEN" \
            --max-steps "$MAX_STEPS" \
            --lr "$LR" \
            --grad-clip "$GRAD_CLIP" \
            --kl "$KL_PENALTY" \
            --save-period "$SAVE_PERIOD" \
            --warmup-steps "$WARMUP_STEPS" \
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA" \
            --lora-dropout "$LORA_DROPOUT" \
            --lora-target-modules "$LORA_TARGETS" \
            --azure-endpoint "$AZURE_ENDPOINT" \
            --azure-api-key "$AZURE_API_KEY" \
            --azure-deployment "$AZURE_MODEL" \
            --azure-api-version "$AZURE_API_VERSION" \
            --judge-max-workers "$JUDGE_MAX_WORKERS" \
            --vllm-server-host "$VLLM_SERVER_HOST" \
            --vllm-server-port "$VLLM_SERVER_PORT" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-run-name "$WANDB_EXP_NAME" \
            "$@"
    local rc=$?
    stop_vllm_server
    exit "$rc"
}

if [[ "$IN_CTR" == "1" ]]; then
    echo "[trl-rl] detected in-container execution (skipping docker run)"
    if [[ -f "$REPO_ROOT/.env" ]]; then
        set -a; . "$REPO_ROOT/.env"; set +a
    fi
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"
    mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
    export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
    # vLLM's CuMemAllocator asserts against expandable_segments — don't set it.

    run_prep_and_train "$@"
fi

# Host mode: re-enter via docker.
ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Run: docker pull $IMAGE" >&2
    exit 1
fi

_to_container() { echo "${1/#$REPO_ROOT/\/workdir}"; }
ACCEL_CFG_C="$(_to_container "$ACCEL_CFG")"
SFT_CKPT_C="$(_to_container "$SFT_CKPT")"
OUT_DIR_C="$(_to_container "$OUT_DIR")"
RL_DATA_DIR_C="$(_to_container "$RL_DATA_DIR")"

echo "[trl-rl] launching via docker image $IMAGE"
# All 8 GPUs visible inside the container: training takes 0-3, vLLM server
# takes 4-7. Each subprocess sets its own CUDA_VISIBLE_DEVICES.
exec docker run --rm --gpus "\"device=$ALL_GPUS\"" --ipc=host --privileged \
    --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    -v "$REPO_ROOT":/workdir \
    -e "PYTHONPATH=/workdir" \
    -e HF_HOME=/workdir/.cache/hf \
    -e PIP_CACHE_DIR=/workdir/.cache/pip \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_PXN_DISABLE=1 \
    -e NCCL_COLLNET_ENABLE=0 \
    -e "IN_CONTAINER=1" \
    -e "ACCEL_CFG=$ACCEL_CFG_C" \
    -e "SFT_CKPT=$SFT_CKPT_C" \
    -e "OUT_DIR=$OUT_DIR_C" \
    -e "RL_DATA_DIR=$RL_DATA_DIR_C" \
    -e "WANDB_PROJECT=$WANDB_PROJECT" \
    -e "WANDB_EXP_NAME=$WANDB_EXP_NAME" \
    -e "MAX_STEPS=$MAX_STEPS" \
    -e "NUM_PROMPTS_PER_STEP=$NUM_PROMPTS_PER_STEP" \
    -e "NUM_GENS_PER_PROMPT=$NUM_GENS_PER_PROMPT" \
    -e "PER_DEVICE_BS=$PER_DEVICE_BS" \
    -e "SEQ_LEN=$SEQ_LEN" \
    -e "MAX_PROMPT_LEN=$MAX_PROMPT_LEN" \
    -e "MAX_COMPLETION_LEN=$MAX_COMPLETION_LEN" \
    -e "LR=$LR" \
    -e "GRAD_CLIP=$GRAD_CLIP" \
    -e "SAVE_PERIOD=$SAVE_PERIOD" \
    -e "KL_PENALTY=$KL_PENALTY" \
    -e "WARMUP_STEPS=$WARMUP_STEPS" \
    -e "LORA_R=$LORA_R" \
    -e "LORA_ALPHA=$LORA_ALPHA" \
    -e "LORA_DROPOUT=$LORA_DROPOUT" \
    -e "LORA_TARGETS=$LORA_TARGETS" \
    -e "VLLM_SERVER_HOST=$VLLM_SERVER_HOST" \
    -e "VLLM_SERVER_PORT=$VLLM_SERVER_PORT" \
    -e "VLLM_GPU_MEM=$VLLM_GPU_MEM" \
    -e "VLLM_TP=$VLLM_TP" \
    -e "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN" \
    -e "VLLM_READY_TIMEOUT_S=$VLLM_READY_TIMEOUT_S" \
    -e "TRAIN_GPUS=$TRAIN_GPUS" \
    -e "VLLM_GPUS=$VLLM_GPUS" \
    -e "ALL_GPUS=$ALL_GPUS" \
    -e "AZURE_ENDPOINT=$AZURE_ENDPOINT" \
    -e "AZURE_API_KEY=$AZURE_API_KEY" \
    -e "AZURE_MODEL=$AZURE_MODEL" \
    -e "AZURE_API_VERSION=$AZURE_API_VERSION" \
    -e "TRL_VERSION=$TRL_VERSION" \
    -e "PEFT_VERSION=$PEFT_VERSION" \
    -e "ACCEL_VERSION=$ACCEL_VERSION" \
    -e "DS_VERSION=$DS_VERSION" \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    "$IMAGE" \
    bash experiment/rl_from_sft_trl.sh "$@"
