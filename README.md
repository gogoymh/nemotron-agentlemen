# nemotron-agentlemen

> Product matching with **Nemotron-3-Nano 30B-A3B** — SFT + GRPO RL pipeline, Nemotron-Super-120B as teacher/judge · Nemotron-Super-120B 를 teacher/judge 로 쓰는 SFT + GRPO RL 제품 매칭 파이프라인

*Team Agentlemen*

<p align="center">
  <b>
    <a href="https://drive.google.com/file/d/14uFfJUvc_jbveBqRN3CL9ngNOfrbFtzv/view?usp=sharing">Paper</a> &nbsp;·&nbsp;
    <a href="https://youtu.be/59HSHSYKiGs">Demo Video</a>
  </b>
</p>

<p align="center">
  <b>
    <a href="#english">English</a> &nbsp;·&nbsp;
    <a href="#한국어">한국어</a>
  </b>
</p>

---

<a id="english"></a>
<details open>
<summary><b>English</b></summary>

## Overview

Fine-tune **Nemotron-3-Nano 30B-A3B** (Mamba/attention hybrid MoE) into a reliable **product-matching** model for Korean e-commerce — given two product titles from different platforms, decide whether they refer to the same SKU and emit structured reasoning.

```
<reason>...evidence...</reason><label>0|1</label>
```

The pipeline covers **data generation**, **supervised fine-tuning**, **GRPO reinforcement learning with LLM-judge reward**, **offline evaluation**, and a **live demo** that drives real e-commerce search UIs.

## Architecture

```
                 ┌────────────────────── data-gen/ ──────────────────────┐
 10M raw KR  →   │  Stage A (Curator)        Stage C (DataDesigner)      │  →  dataset/v2/
 listings        │  fuzzy dedup              LLM evidence regen          │      (labeled + synth pairs)
                 │  fastText quality         pseudo-label via 120B       │
                 │  text modifier            hard-pair synthesis         │
                 │  decontaminate            judge validator             │
                 └───────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          ┌─────── experiment/ ───────┐
                          │  src/train_sft.py         │
                          │  (Megatron-Bridge LoRA)   │   SFT checkpoint
                          └───────────────────────────┘        │
                                        │                      │
                                        ▼                      │
                          ┌─── experiment/rl_from_sft_trl ───┐ │
                          │  TRL GRPO · LoRA r=32 α=64       │ │
                          │  vLLM rollout server (TP=4)      │◀┘
                          │  Judge: Nemotron-Super-120B      │
                          │         or Azure gpt-5.x         │
                          └──────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌── src/evals/product_matching ──┐    ┌──────── demo/ ────────┐
        │  BYOB benchmark                │    │  live Playwright      │
        │  precision / recall / F1       │    │  across 13 KR sites   │
        └────────────────────────────────┘    │  demo/env/ RL env     │
                                              └───────────────────────┘
```

## Layout

```
nemotron-agentlemen/
├── data-gen/              # Stage A (Curator) + Stage C (DataDesigner)
├── src/                   # train_sft.py, evals/, data_prep/, config.py
├── experiment/            # SFT + RL shell recipes, judge server, docker
├── prompt/                # reasoning + difficulty + reward-agent prompts
├── demo/                  # live multi-site product matching demo
│   └── env/               # NeMo-Gym-style RL environment (FastAPI)
├── Nemotron/              # submodule — NVIDIA-NeMo/Nemotron
├── DataDesigner/          # submodule — NVIDIA-NeMo/DataDesigner
├── Curator/               # submodule — NVIDIA-NeMo/Curator
├── Megatron-Bridge/       # submodule — NVIDIA-NeMo/Megatron-Bridge (nano-v3)
├── RL/                    # submodule — NVIDIA-NeMo/RL
└── nemo-gym/              # submodule — NVIDIA-NeMo/Gym
```

## Models

| Role | HF ID | Notes |
|---|---|---|
| Policy | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | MoE 30B / 3B active · hybrid Mamba2+attention |
| Teacher / judge | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | MoE 120B / 12B active · served via vLLM for DataDesigner + RL judge |

## Environment setup

### Hardware

- **SFT / RL**: 8× H100 80GB on one host (GPUs 0–3 for training, 4–7 for the vLLM rollout server).
- **120B teacher / judge**: a separate 8× H100 host is recommended (used by `data-gen/` synthesis and as the RL judge). GPUs can be shared with the trainer if batch size allows.
- **Storage**: ≥500 GB fast local on `/ephemeral` for checkpoints. Adjust `src/config.py:EPHEMERAL_ROOT` if your mount differs.

### Docker images

All training and serving happen inside NVIDIA NeMo containers:

```bash
docker pull nvcr.io/nvidia/nemo:25.11.nemotron_3_nano       # SFT image (Megatron-Bridge nano-v3)
docker pull nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano   # RL image (TRL GRPO)

# optional: a local image with project-pinned wheels layered on top
bash experiment/build_docker.sh
```

### Python (host) — for `data-gen/`, evals, and the demo

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Nemotron-H hybrid (Mamba2) fast-path kernels — prebuilt wheels for torch==2.7.0 + cu12:
pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# demo + RL-env extras (Playwright / FastAPI)
pip install -r demo/env/requirements.txt
playwright install chromium
```

### `.env` (repo root)

Loaded by every training / serving script via `--env-file`. Create at the repo root:

```bash
HF_TOKEN=hf_...                            # required — HuggingFace model download
WANDB_API_KEY=...                          # optional — training run logging

# Optional — Azure OpenAI judge path (RL falls back to the local 120B if unset)
AZURE_ENDPOINT=https://<resource>.openai.azure.com
AZURE_API_KEY=...
AZURE_MODEL=<deployment-name>
AZURE_API_VERSION=2024-...
```

### Expected data layout

Built by `src/data_prep/` from the `data-gen/` outputs — all paths under `data/` are gitignored:

```
data/
├── sft/
│   ├── training.jsonl         # HF-chat rows: {messages: [{role, content}, ...]}
│   └── validation.jsonl
├── rl/
│   ├── train.jsonl            # {input: "...", output: "..."} — RL prompts
│   └── validation.jsonl
└── multi-platform/
    └── test.csv               # held-out eval split (BYOB benchmark reads this)
```

### Verify

```bash
nvidia-smi                                              # 8 GPUs visible
docker run --rm --gpus all nvcr.io/nvidia/nemo:25.11.nemotron_3_nano nvidia-smi
df -h /ephemeral                                        # mount + free space
```

## Quickstart

### 0. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:gogoymh/nemotron-agentlemen.git
cd nemotron-agentlemen
```

### 1. Generate the training dataset

```bash
# Stage A: clean 10M raw listings
python data-gen/stage_a_curator/run_curator.py \
    --config data-gen/configs/curator.yaml

# Serve the 120B teacher (separate terminal)
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --served-model-name nemotron-super --port 8000

# Stage C: synthesize evidence, pseudo-labels, hard pairs
python data-gen/stage_c_datadesigner/run_datadesigner.py \
    --config data-gen/configs/datadesigner.yaml
```

See [`data-gen/README.md`](data-gen/README.md).

### 2. Supervised fine-tuning (LoRA)

```bash
# Inside NVIDIA NeMo 25.11 container, 8 GPUs
sudo bash experiment/sft.sh
# → artifacts/sft/nemotron-30b/hf-iter_0001875
```

Common overrides (env vars on the `sft.sh` command line):

| Env var | Default | Meaning |
|---|---|---|
| `NPROC` | 8 | GPUs per node (torchrun) |
| `GBS` / `MBS` | 8 / 1 | global / micro batch size |
| `NUM_EPOCHS` | 5 | passes over `data/sft/` |
| `LR` | 1e-5 | learning rate |
| `OUT_DIR` | `/ephemeral/.../sft/nemotron-30b` | checkpoint dir |
| `HF_CKPT` | `/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | base HF snapshot |
| `WANDB_PROJECT` | `nemotron-agentlemen-sft` | wandb project |

Example: `GBS=16 NUM_EPOCHS=3 LR=5e-6 sudo bash experiment/sft.sh`.

### 3. GRPO reinforcement learning

```bash
# Launches vLLM rollout server + TRL GRPOTrainer with LoRA
sudo bash experiment/rl_from_sft_trl.sh
# → artifacts/rl/nemotron-30b-trl/checkpoints/checkpoint-$STEP
```

Rollouts are scored by the frozen 120B judge (or Azure gpt-5.x if `.env` provides Azure creds). Common overrides:

| Env var | Default | Meaning |
|---|---|---|
| `ITER` | 1875 | SFT iter to resume from (maps to `artifacts/sft/.../hf-iter_0001875`) |
| `MAX_STEPS` | 500 | GRPO training steps |
| `NUM_PROMPTS_PER_STEP` | 1 | prompts per optimizer step |
| `NUM_GENS_PER_PROMPT` | 4 | rollouts per prompt (GRPO group size) |
| `LR` | 5e-5 | LoRA learning rate |
| `KL_PENALTY` | 0.01 | KL to reference policy |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | 32 / 64 / 0.05 | LoRA hyperparams |
| `LORA_TARGETS` | `q_proj,k_proj,v_proj` | adapter target modules |
| `VLLM_TP` | 4 | tensor-parallel for rollout server |
| `VLLM_GPU_MEM` | 0.85 | vLLM GPU memory utilization |
| `SAVE_PERIOD` | 100 | checkpoint every N steps |

### 4. Merge LoRA and serve

```bash
STEP=200 bash experiment/merge_rl_lora.sh      # adapter + base → bf16 HF
STEP=200 bash experiment/serve_rl.sh --detach  # vLLM on :8000
```

### 5. Evaluate

```bash
python -m src.evals.product_matching.prepare_data
nemo-evaluator run_eval \
    --eval_type byob_product_matching.product_matching \
    --model_url http://localhost:8000 --model_id nemotron-30b-rl \
    --model_type chat --output_dir results/RL-step$STEP
python -m src.evals.product_matching.report results/RL-step$STEP
```

### 6. Run the live demo

```bash
python fanout_demo.py --preset tshirt
```

See [`demo/README.md`](demo/README.md) and [`demo/env/README.md`](demo/env/README.md).

## What's not in this repo

Runtime artifacts are gitignored and must be regenerated locally: `data/`, `results/`, `artifacts/`, `megatron_checkpoints/`, `nemo_experiments/`, `wandb/`, `.cache/`, `demo/cache/`, internal `docs/`, and ad-hoc `exp/` notebooks. Models are pulled from HuggingFace on first use. Large assets (30 GB+ checkpoints) live on `/ephemeral` per machine; see `src/config.py`.

</details>

---

<a id="한국어"></a>
<details>
<summary><b>한국어</b></summary>

## 개요

**Nemotron-3-Nano 30B-A3B** (Mamba/attention 하이브리드 MoE) 를 한국 커머스 **상품 매칭** 전용으로 미세조정. 서로 다른 플랫폼의 두 상품명을 받아 같은 SKU 인지 판정하고, 구조화된 근거를 같이 출력:

```
<reason>...근거...</reason><label>0|1</label>
```

파이프라인은 **데이터 생성 → SFT → GRPO RL (LLM-judge 리워드) → 오프라인 평가 → 실시간 데모** 전체를 커버.

## 아키텍처

다이어그램은 위 English 블록과 동일 — 공유 컴포넌트:

- **`data-gen/`**: 10M 한국어 상품명을 Curator (Stage A: 퍼지 중복제거 · 품질필터 · 텍스트정제 · 탈오염) 와 DataDesigner (Stage C: 근거 재생성 · pseudo-label · hard-pair 합성 · judge validator) 로 처리
- **`experiment/` + `src/train_sft.py`**: Megatron-Bridge 기반 LoRA SFT
- **`experiment/rl_from_sft_trl.*`**: TRL GRPO + LoRA (r=32, α=64), vLLM 롤아웃 서버 (TP=4), Nemotron-Super-120B 또는 Azure gpt-5.x 를 judge 로 사용
- **`src/evals/product_matching/`**: BYOB (nemo-evaluator) 벤치마크로 P/R/F1 산출
- **`demo/` + `demo/env/`**: 실제 커머스 검색 UI 를 Playwright 로 돌리는 데모 + NeMo-Gym 스타일 RL 환경

## 디렉토리

위 English 블록의 레이아웃과 동일.

## 모델

| 역할 | HF ID | 비고 |
|---|---|---|
| Policy | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | MoE 30B / 활성 3B · Mamba2+attention 하이브리드 |
| Teacher / judge | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | MoE 120B / 활성 12B · DataDesigner 합성 + RL judge 모두 vLLM 으로 서빙 |

## 환경 설정

### 하드웨어

- **SFT / RL**: 한 호스트에 H100 80GB × 8 (GPU 0–3 학습, 4–7 vLLM 롤아웃 서버).
- **120B teacher / judge**: 별도 H100 × 8 호스트 권장 (`data-gen/` 합성 + RL judge 용). 배치 크기가 허락하면 학습 호스트와 공유 가능.
- **스토리지**: 체크포인트용 `/ephemeral` 에 500 GB 이상 여유. 마운트가 다르면 `src/config.py:EPHEMERAL_ROOT` 수정.

### Docker 이미지

모든 학습/서빙은 NVIDIA NeMo 컨테이너 안에서 돌아갑니다:

```bash
docker pull nvcr.io/nvidia/nemo:25.11.nemotron_3_nano       # SFT (Megatron-Bridge nano-v3)
docker pull nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano   # RL (TRL GRPO)

# 선택: 프로젝트 고정 휠을 덧씌운 로컬 이미지
bash experiment/build_docker.sh
```

### Python (호스트) — `data-gen/`, 평가, 데모용

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Nemotron-H 하이브리드 (Mamba2) fast-path 커널 — torch==2.7.0 + cu12 prebuilt wheel:
pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# 데모 + RL-env 추가 (Playwright / FastAPI)
pip install -r demo/env/requirements.txt
playwright install chromium
```

### `.env` (레포 루트)

모든 학습/서빙 스크립트가 `--env-file` 로 읽습니다. 레포 루트에 생성:

```bash
HF_TOKEN=hf_...                            # 필수 — HF 모델 다운로드
WANDB_API_KEY=...                          # 선택 — 학습 로깅

# 선택 — Azure OpenAI judge 경로 (미설정 시 로컬 120B 로 폴백)
AZURE_ENDPOINT=https://<resource>.openai.azure.com
AZURE_API_KEY=...
AZURE_MODEL=<deployment-name>
AZURE_API_VERSION=2024-...
```

### 예상 데이터 레이아웃

`src/data_prep/` 가 `data-gen/` 출력으로부터 빌드 — `data/` 아래는 전부 gitignore:

```
data/
├── sft/
│   ├── training.jsonl         # HF-chat 포맷: {messages: [{role, content}, ...]}
│   └── validation.jsonl
├── rl/
│   ├── train.jsonl            # {input: "...", output: "..."} — RL 프롬프트
│   └── validation.jsonl
└── multi-platform/
    └── test.csv               # held-out 평가 분할 (BYOB 벤치마크가 읽음)
```

### 확인

```bash
nvidia-smi                                              # GPU 8장 확인
docker run --rm --gpus all nvcr.io/nvidia/nemo:25.11.nemotron_3_nano nvidia-smi
df -h /ephemeral                                        # 마운트 + 여유 공간
```

## 빠른 시작

### 0. 서브모듈까지 클론

```bash
git clone --recurse-submodules git@github.com:gogoymh/nemotron-agentlemen.git
cd nemotron-agentlemen
```

### 1. 학습 데이터 생성

```bash
# Stage A: 10M raw 정제
python data-gen/stage_a_curator/run_curator.py \
    --config data-gen/configs/curator.yaml

# 120B teacher 서빙 (다른 터미널)
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --served-model-name nemotron-super --port 8000

# Stage C: 근거 재생성 · pseudo-label · hard-pair 합성
python data-gen/stage_c_datadesigner/run_datadesigner.py \
    --config data-gen/configs/datadesigner.yaml
```

자세한 내용은 [`data-gen/README.md`](data-gen/README.md).

### 2. SFT (LoRA)

```bash
# NVIDIA NeMo 25.11 컨테이너, GPU 8장
sudo bash experiment/sft.sh
# → artifacts/sft/nemotron-30b/hf-iter_0001875
```

주요 오버라이드 (env 변수로 `sft.sh` 앞에 붙이면 됨):

| Env var | 기본값 | 의미 |
|---|---|---|
| `NPROC` | 8 | 노드당 GPU 수 (torchrun) |
| `GBS` / `MBS` | 8 / 1 | global / micro 배치 크기 |
| `NUM_EPOCHS` | 5 | `data/sft/` pass 횟수 |
| `LR` | 1e-5 | 학습률 |
| `OUT_DIR` | `/ephemeral/.../sft/nemotron-30b` | 체크포인트 디렉토리 |
| `HF_CKPT` | `/ephemeral/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 베이스 HF 스냅샷 |
| `WANDB_PROJECT` | `nemotron-agentlemen-sft` | wandb 프로젝트 |

예시: `GBS=16 NUM_EPOCHS=3 LR=5e-6 sudo bash experiment/sft.sh`.

### 3. GRPO 강화학습

```bash
# vLLM 롤아웃 서버 + TRL GRPOTrainer + LoRA 를 한 번에 띄움
sudo bash experiment/rl_from_sft_trl.sh
# → artifacts/rl/nemotron-30b-trl/checkpoints/checkpoint-$STEP
```

롤아웃은 120B judge (또는 `.env` 에 Azure 키가 있으면 Azure gpt-5.x) 가 채점. 주요 오버라이드:

| Env var | 기본값 | 의미 |
|---|---|---|
| `ITER` | 1875 | 재개할 SFT iter (`artifacts/sft/.../hf-iter_0001875` 로 매핑) |
| `MAX_STEPS` | 500 | GRPO 학습 스텝 수 |
| `NUM_PROMPTS_PER_STEP` | 1 | 옵티마이저 스텝당 프롬프트 수 |
| `NUM_GENS_PER_PROMPT` | 4 | 프롬프트당 롤아웃 (GRPO group size) |
| `LR` | 5e-5 | LoRA 학습률 |
| `KL_PENALTY` | 0.01 | reference policy KL |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | 32 / 64 / 0.05 | LoRA 하이퍼 |
| `LORA_TARGETS` | `q_proj,k_proj,v_proj` | adapter 대상 모듈 |
| `VLLM_TP` | 4 | 롤아웃 서버 tensor-parallel |
| `VLLM_GPU_MEM` | 0.85 | vLLM GPU 메모리 점유율 |
| `SAVE_PERIOD` | 100 | N 스텝마다 체크포인트 저장 |

### 4. LoRA 병합 후 서빙

```bash
STEP=200 bash experiment/merge_rl_lora.sh      # adapter + base → bf16 HF
STEP=200 bash experiment/serve_rl.sh --detach  # vLLM on :8000
```

### 5. 평가

```bash
python -m src.evals.product_matching.prepare_data
nemo-evaluator run_eval \
    --eval_type byob_product_matching.product_matching \
    --model_url http://localhost:8000 --model_id nemotron-30b-rl \
    --model_type chat --output_dir results/RL-step$STEP
python -m src.evals.product_matching.report results/RL-step$STEP
```

### 6. 실시간 데모

```bash
python fanout_demo.py --preset tshirt
```

상세는 [`demo/README.md`](demo/README.md) 와 [`demo/env/README.md`](demo/env/README.md).

## 레포에 들어있지 않은 것

런타임 산출물은 `.gitignore` 로 제외 — 로컬에서 재생성 필요: `data/`, `results/`, `artifacts/`, `megatron_checkpoints/`, `nemo_experiments/`, `wandb/`, `.cache/`, `demo/cache/`, 내부 `docs/`, 실험용 `exp/`. 모델은 최초 실행 시 HuggingFace 에서 내려받음. 30GB 이상 체크포인트는 머신별 `/ephemeral` 에 위치 (`src/config.py` 참조).

</details>
