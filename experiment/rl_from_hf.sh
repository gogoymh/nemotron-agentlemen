#!/usr/bin/env bash
# HF-based GRPO launcher. Skips SFT continuation — RL starts from the base
# HF Nemotron-3-Nano checkpoint. See experiment/grpo_rl_hf.yaml for why.
#
# Usage:
#   sudo bash experiment/rl_from_hf.sh
#
# Prerequisite:
#   bash experiment/judge_server.sh   (the 120B judge on GPUs 4-7)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano}"
CONFIG="${CONFIG:-$REPO_ROOT/experiment/grpo_rl_hf.yaml}"

HF_MODEL="${HF_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/rl/nemotron-30b-hf}"
RL_DATA_DIR="${RL_DATA_DIR:-$REPO_ROOT/data/rl}"
HF_HOME_DEFAULT="$REPO_ROOT/.cache/hf"

WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-rl}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-rl-from-hf-$(date +%Y%m%d-%H%M%S)}"

# GRPO hyperparameters.
MAX_STEPS="${MAX_STEPS:-500}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
NUM_GENS_PER_PROMPT="${NUM_GENS_PER_PROMPT:-4}"
SEQ_LEN="${SEQ_LEN:-1024}"
LR="${LR:-5.0e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SAVE_PERIOD="${SAVE_PERIOD:-100}"
KL_PENALTY="${KL_PENALTY:-0.01}"
TP="${TP:-4}"

JUDGE_URL="${JUDGE_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-nemotron-3-super-120b}"
REQUIRE_JUDGE="${REQUIRE_JUDGE:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"

mkdir -p "$OUT_DIR" "$RL_DATA_DIR" "$OUT_DIR/generations" "$OUT_DIR/logs"

if [[ -n "${IN_CONTAINER:-}" ]]; then
    IN_CTR="$IN_CONTAINER"
elif [[ -f /.dockerenv ]] || ! command -v docker >/dev/null 2>&1; then
    IN_CTR=1
else
    IN_CTR=0
fi

check_judge() {
    if [[ "$REQUIRE_JUDGE" != "1" ]]; then
        echo "[rl-hf] REQUIRE_JUDGE=$REQUIRE_JUDGE — skipping judge health check"
        return 0
    fi
    local url="${JUDGE_URL%/}/models"
    echo "[rl-hf] probing judge at $url"
    if command -v curl >/dev/null 2>&1 && \
       curl -s --max-time 5 "$url" >/dev/null; then
        echo "[rl-hf] judge OK"
        return 0
    fi
    echo "ERROR: judge server unreachable at $JUDGE_URL" >&2
    echo "       start it first: bash experiment/judge_server.sh" >&2
    exit 1
}

run_prep_and_train() {
    echo "[rl-hf] preparing RL JSONL"
    python -m src.data_prep.csv_to_rl_jsonl --out-dir "$RL_DATA_DIR"

    check_judge

    echo "[rl-hf] launching GRPO (HF + DTensor)"
    echo "[rl-hf]   base model   : $HF_MODEL"
    echo "[rl-hf]   train GPUs   : $TRAIN_GPUS  (TP=$TP)"
    echo "[rl-hf]   judge        : $JUDGE_MODEL @ $JUDGE_URL"
    echo "[rl-hf]   steps=$MAX_STEPS  gbs=$GBS  mbs=$MBS  num_gens=$NUM_GENS_PER_PROMPT"
    echo "[rl-hf]   lr=$LR  clip=$GRAD_CLIP  kl=$KL_PENALTY  save_period=$SAVE_PERIOD"
    echo "[rl-hf]   per-step dump: $OUT_DIR/generations/generations.jsonl"

    export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
    export JUDGE_URL JUDGE_MODEL

    exec python "$REPO_ROOT/experiment/rl_from_sft.py" \
        --config "$CONFIG" \
        policy.model_name="$HF_MODEL" \
        policy.tokenizer.name="$HF_MODEL" \
        policy.train_global_batch_size="$GBS" \
        policy.train_micro_batch_size="$MBS" \
        policy.max_total_sequence_length="$SEQ_LEN" \
        policy.max_grad_norm="$GRAD_CLIP" \
        policy.dtensor_cfg.tensor_parallel_size="$TP" \
        policy.generation.vllm_cfg.tensor_parallel_size="$TP" \
        policy.optimizer.kwargs.lr="$LR" \
        grpo.max_num_steps="$MAX_STEPS" \
        grpo.num_prompts_per_step="$NUM_PROMPTS_PER_STEP" \
        grpo.num_generations_per_prompt="$NUM_GENS_PER_PROMPT" \
        loss_fn.reference_policy_kl_penalty="$KL_PENALTY" \
        data.train_data_path="$RL_DATA_DIR/training.jsonl" \
        data.val_data_path="$RL_DATA_DIR/validation.jsonl" \
        checkpointing.checkpoint_dir="$OUT_DIR/checkpoints" \
        checkpointing.save_period="$SAVE_PERIOD" \
        env.product_match.judge_url="$JUDGE_URL" \
        env.product_match.judge_model="$JUDGE_MODEL" \
        env.product_match.log_dir="$OUT_DIR/generations" \
        logger.log_dir="$OUT_DIR/logs" \
        logger.wandb.project="$WANDB_PROJECT" \
        logger.wandb.name="$WANDB_EXP_NAME" \
        "$@"
}

if [[ "$IN_CTR" == "1" ]]; then
    echo "[rl-hf] detected in-container execution (skipping docker run)"
    if [[ -f "$REPO_ROOT/.env" ]]; then
        set -a; . "$REPO_ROOT/.env"; set +a
    fi
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
    mkdir -p "$HF_HOME"

    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
    export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"

    run_prep_and_train "$@"
fi

# Host mode.
ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Run: docker pull $IMAGE" >&2
    exit 1
fi

_to_container() { echo "${1/#$REPO_ROOT/\/workdir}"; }
CONFIG_C="$(_to_container "$CONFIG")"
OUT_DIR_C="$(_to_container "$OUT_DIR")"
RL_DATA_DIR_C="$(_to_container "$RL_DATA_DIR")"

echo "[rl-hf] launching via docker image $IMAGE"
exec docker run --rm --gpus "\"device=$TRAIN_GPUS\"" --ipc=host --privileged \
    --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    -v "$REPO_ROOT":/workdir \
    -e "PYTHONPATH=/workdir" \
    -e HF_HOME=/workdir/.cache/hf \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_PXN_DISABLE=1 \
    -e NCCL_COLLNET_ENABLE=0 \
    -e "IN_CONTAINER=1" \
    -e "CONFIG=$CONFIG_C" \
    -e "HF_MODEL=$HF_MODEL" \
    -e "OUT_DIR=$OUT_DIR_C" \
    -e "RL_DATA_DIR=$RL_DATA_DIR_C" \
    -e "WANDB_PROJECT=$WANDB_PROJECT" \
    -e "WANDB_EXP_NAME=$WANDB_EXP_NAME" \
    -e "MAX_STEPS=$MAX_STEPS" \
    -e "GBS=$GBS" \
    -e "MBS=$MBS" \
    -e "NUM_PROMPTS_PER_STEP=$NUM_PROMPTS_PER_STEP" \
    -e "NUM_GENS_PER_PROMPT=$NUM_GENS_PER_PROMPT" \
    -e "SEQ_LEN=$SEQ_LEN" \
    -e "LR=$LR" \
    -e "GRAD_CLIP=$GRAD_CLIP" \
    -e "SAVE_PERIOD=$SAVE_PERIOD" \
    -e "KL_PENALTY=$KL_PENALTY" \
    -e "TP=$TP" \
    -e "TRAIN_GPUS=$TRAIN_GPUS" \
    -e "JUDGE_URL=$JUDGE_URL" \
    -e "JUDGE_MODEL=$JUDGE_MODEL" \
    -e "REQUIRE_JUDGE=$REQUIRE_JUDGE" \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    "$IMAGE" \
    bash experiment/rl_from_hf.sh "$@"
