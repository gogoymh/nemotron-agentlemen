#!/usr/bin/env bash
# Launcher for experiment/rl_from_sft.py — copy_lr.md GRPO flow (Megatron).
#
# ── PREREQUISITE ────────────────────────────────────────────────────────────
# NeMo-RL requires an HF-format dir for `policy.model_name` (vLLM reads
# config.json + safetensors) even in Megatron mode. The SFT run produces a
# Megatron dist_checkpoint, so merge+export it to HF first:
#
#     sudo ITER=1875 bash experiment/merge_lora.sh
#     # → artifacts/sft/nemotron-30b/hf-iter_0001875/   (HF safetensors)
#
# Then start the judge server and this launcher:
#     bash experiment/judge_server.sh
#     sudo bash experiment/rl_from_sft.sh
# ─────────────────────────────────────────────────────────────────────────────
#
# Training lives on GPUs 0–3 (CUDA_VISIBLE_DEVICES pinned below). A vLLM
# OpenAI server hosting the 120B judge must be up on GPUs 4–7 before this
# script is run — start it with:
#
#     bash experiment/judge_server.sh
#
# Host vs in-container is auto-detected (same pattern as sft_from_copy.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano}"
CONFIG="${CONFIG:-$REPO_ROOT/experiment/grpo_rl.yaml}"

# ── Paths ────────────────────────────────────────────────────────────────────
ITER="${ITER:-1875}"
ITER_PAD="$(printf 'iter_%07d' "$ITER")"
# HF-format SFT checkpoint (LoRA merged + exported via merge_lora.sh). vLLM
# inside NeMo-RL needs this to be a real HF dir, not a Megatron dist_ckpt.
SFT_CKPT="${SFT_CKPT:-$REPO_ROOT/artifacts/sft/nemotron-30b/hf-$ITER_PAD}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/rl/nemotron-30b}"
RL_DATA_DIR="${RL_DATA_DIR:-$REPO_ROOT/data/rl}"
HF_HOME_DEFAULT="$REPO_ROOT/.cache/hf"

WANDB_PROJECT="${WANDB_PROJECT:-nemotron-agentlemen-rl}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-rl-from-sft-$(date +%Y%m%d-%H%M%S)}"

# ── GRPO hyperparameters (user-spec defaults) ────────────────────────────────
MAX_STEPS="${MAX_STEPS:-500}"      # 4000 train rows / 8 prompts = 1 epoch
GBS="${GBS:-32}"                    # train_global_batch_size (8 prompts × 4 gens)
MBS="${MBS:-4}"                     # train_micro_batch_size
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"    # GBS / num_generations
NUM_GENS_PER_PROMPT="${NUM_GENS_PER_PROMPT:-4}"
SEQ_LEN="${SEQ_LEN:-1024}"
LR="${LR:-5.0e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SAVE_PERIOD="${SAVE_PERIOD:-100}"
KL_PENALTY="${KL_PENALTY:-0.01}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-linear_qkv}"

# ── Judge server ────────────────────────────────────────────────────────────
JUDGE_URL="${JUDGE_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-nemotron-3-super-120b}"   # vLLM --served-model-name alias
REQUIRE_JUDGE="${REQUIRE_JUDGE:-1}"

# GPUs 0-3 for training; 4-7 hold the judge.
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"

if [[ ! -d "$SFT_CKPT" || ! -f "$SFT_CKPT/config.json" ]]; then
    echo "ERROR: HF-format SFT checkpoint not found: $SFT_CKPT" >&2
    echo "       run the merge+export pipeline first:" >&2
    echo "         sudo ITER=$ITER bash experiment/merge_lora.sh" >&2
    echo "       (or set SFT_CKPT=/path/to/hf-iter_XXXXXXX explicitly.)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$RL_DATA_DIR" "$OUT_DIR/generations" "$OUT_DIR/logs"

# ── Detect: are we already inside a container? ───────────────────────────────
if [[ -n "${IN_CONTAINER:-}" ]]; then
    IN_CTR="$IN_CONTAINER"
elif [[ -f /.dockerenv ]] || ! command -v docker >/dev/null 2>&1; then
    IN_CTR=1
else
    IN_CTR=0
fi

check_judge() {
    # Quick reachability probe so we don't burn a GPU startup cost just to
    # discover the judge isn't up.
    if [[ "$REQUIRE_JUDGE" != "1" ]]; then
        echo "[rl] REQUIRE_JUDGE=$REQUIRE_JUDGE — skipping judge health check"
        return 0
    fi
    local url="${JUDGE_URL%/}/models"
    echo "[rl] probing judge at $url"
    if command -v curl >/dev/null 2>&1 && \
       curl -s --max-time 5 "$url" >/dev/null; then
        echo "[rl] judge OK"
        return 0
    fi
    echo "ERROR: judge server unreachable at $JUDGE_URL" >&2
    echo "       start it first: bash experiment/judge_server.sh" >&2
    echo "       (or set REQUIRE_JUDGE=0 to bypass — training will compute" >&2
    echo "        judge rewards as 0 until the server comes up)" >&2
    exit 1
}

run_prep_and_train() {
    echo "[rl] preparing RL JSONL from data/multi-platform/rl.csv"
    python -m src.data_prep.csv_to_rl_jsonl --out-dir "$RL_DATA_DIR"

    check_judge

    echo "[rl] launching GRPO"
    echo "[rl]   SFT ckpt     : $SFT_CKPT"
    echo "[rl]   train GPUs   : $TRAIN_GPUS"
    echo "[rl]   judge        : $JUDGE_MODEL @ $JUDGE_URL"
    echo "[rl]   steps=$MAX_STEPS  gbs=$GBS  mbs=$MBS  num_gens=$NUM_GENS_PER_PROMPT"
    echo "[rl]   lr=$LR  clip=$GRAD_CLIP  kl=$KL_PENALTY  save_period=$SAVE_PERIOD"
    echo "[rl]   LoRA r=$LORA_R α=$LORA_ALPHA dropout=$LORA_DROPOUT target=$LORA_TARGETS"
    echo "[rl]   per-step dump: $OUT_DIR/generations/generations.jsonl"

    export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
    export JUDGE_URL JUDGE_MODEL

    # Hydra overrides — keys match the real NeMo-RL v0.4.0 schema (see
    # experiment/grpo_rl.yaml header).
    # Use the container's system python (ray/NeMo-RL preinstalled in the
    # nemo-rl image) — `uv run` would create a fresh venv without ray.
    exec python "$REPO_ROOT/experiment/rl_from_sft.py" \
        --config "$CONFIG" \
        policy.model_name="$SFT_CKPT" \
        policy.train_global_batch_size="$GBS" \
        policy.train_micro_batch_size="$MBS" \
        policy.max_total_sequence_length="$SEQ_LEN" \
        policy.max_grad_norm="$GRAD_CLIP" \
        policy.megatron_cfg.optimizer.lr="$LR" \
        grpo.max_num_steps="$MAX_STEPS" \
        grpo.num_prompts_per_step="$NUM_PROMPTS_PER_STEP" \
        grpo.num_generations_per_prompt="$NUM_GENS_PER_PROMPT" \
        loss_fn.reference_policy_kl_penalty="$KL_PENALTY" \
        peft.dim="$LORA_R" \
        peft.alpha="$LORA_ALPHA" \
        peft.dropout="$LORA_DROPOUT" \
        "peft.target_modules=[$LORA_TARGETS]" \
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

# ── Inside-container mode ───────────────────────────────────────────────────
if [[ "$IN_CTR" == "1" ]]; then
    echo "[rl] detected in-container execution (skipping docker run)"

    if [[ -f "$REPO_ROOT/.env" ]]; then
        set -a; . "$REPO_ROOT/.env"; set +a
    fi

    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
    mkdir -p "$HF_HOME"

    # NCCL guards for H100 PCIe (see sft_from_copy.sh).
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
    export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
    # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True —
    # vLLM's CuMemAllocator has a hard assertion against it.

    run_prep_and_train "$@"
fi

# ── Host mode ───────────────────────────────────────────────────────────────
ENV_FILE_ARG=()
[[ -f "$REPO_ROOT/.env" ]] && ENV_FILE_ARG=(--env-file "$REPO_ROOT/.env")

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' not found. Run: docker pull $IMAGE" >&2
    exit 1
fi

# Translate host paths under $REPO_ROOT to the container's /workdir mount so
# inside-container code sees valid paths for SFT_CKPT / CONFIG / OUT_DIR / etc.
_to_container() { echo "${1/#$REPO_ROOT/\/workdir}"; }
CONFIG_C="$(_to_container "$CONFIG")"
SFT_CKPT_C="$(_to_container "$SFT_CKPT")"
OUT_DIR_C="$(_to_container "$OUT_DIR")"
RL_DATA_DIR_C="$(_to_container "$RL_DATA_DIR")"

echo "[rl] launching via docker image $IMAGE"
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
    -e "SFT_CKPT=$SFT_CKPT_C" \
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
    -e "LORA_R=$LORA_R" \
    -e "LORA_ALPHA=$LORA_ALPHA" \
    -e "LORA_DROPOUT=$LORA_DROPOUT" \
    -e "LORA_TARGETS=$LORA_TARGETS" \
    -e "TRAIN_GPUS=$TRAIN_GPUS" \
    -e "JUDGE_URL=$JUDGE_URL" \
    -e "JUDGE_MODEL=$JUDGE_MODEL" \
    -e "REQUIRE_JUDGE=$REQUIRE_JUDGE" \
    "${ENV_FILE_ARG[@]}" \
    -w /workdir \
    "$IMAGE" \
    bash experiment/rl_from_sft.sh "$@"
