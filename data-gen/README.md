# data-gen — Curator (Stage A) + DataDesigner (Stage C)

> Training-data pipeline for the product-matching model · 상품 매칭 모델용 학습 데이터 파이프라인

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

Implements the Stage A + Stage C steps of the data plan on **~10M unlabeled Korean product listings** (`dataset/get_data/raw.jsonl`).

- **Stage A — Curator**: fuzzy dedup, fastText quality filter, title normalization, decontamination against held-out splits
- **Stage B — labeled split**: already built in `dataset/v2/` (human-labeled pairs, not part of this directory)
- **Stage C — DataDesigner**: LLM-driven evidence regeneration, pseudo-labeling with confidence gating, hard-pair synthesis, judge validator
- **Stage D — SFT/RL builders**: lives downstream of Stage C output

All LLM calls hit **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`** served locally via vLLM (OpenAI-compatible API on port 8000).

## Layout

```
data-gen/
├── configs/
│   ├── curator.yaml            # paths, thresholds, executor
│   └── datadesigner.yaml       # model, blends, judge settings
├── common/
│   ├── schema.py               # canonical column names
│   ├── io.py                   # jsonl + yaml helpers
│   └── nemotron_client.py      # OpenAI-SDK wrapper for the vLLM endpoint
├── stage_a_curator/
│   ├── a1_fuzzy_dedup.py       # MinHash LSH (Jaccard ~0.8)
│   ├── a2_quality_filter.py    # fastText + ScoreFilter (Pareto α=3)
│   ├── a3_text_modifier.py     # product_name_clean (promo/emoji strip, raw preserved)
│   ├── a4_decontaminate.py     # exact-match drop vs v2 val/test union
│   └── run_curator.py          # A.1 → A.2 → A.3 → A.4 orchestrator
└── stage_c_datadesigner/
    ├── _model.py               # shared ModelConfig/ProviderConfig
    ├── c1_evidence_regen.py    # LLMTextColumnConfig (label-conditioned)
    ├── c2_pseudo_label.py      # LLMJudgeColumnConfig + confidence gating
    ├── c3_hard_pairs.py        # 5-sample variant synthesis per seed
    ├── c4_judge_validator.py   # structural + N-sample consistency check
    └── run_datadesigner.py     # C.1 → C.2 → C.3 → C.4 orchestrator
```

## End-to-end run

Scripts are invoked by path, not `-m`: the containing directory is `data-gen/` (hyphenated) so it isn't importable as a Python package. Each entrypoint injects `data-gen/` onto `sys.path`.

```bash
# ---------- Stage A: Curator on 10M raw listings ---------------------
python data-gen/stage_a_curator/run_curator.py \
    --config data-gen/configs/curator.yaml
# → dataset/v2/raw_curated/full.jsonl  (~1.1M rows after A.1/A.2/A.4)

# ---------- Serve Nemotron-3 Super for Stage C -----------------------
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --served-model-name nemotron-super --port 8000 &
export NEMOTRON_BASE_URL=http://localhost:8000/v1
export NEMOTRON_API_KEY=EMPTY

# ---------- Stage C: DataDesigner synthesis --------------------------
python data-gen/stage_c_datadesigner/run_datadesigner.py \
    --config data-gen/configs/datadesigner.yaml
# → dataset/v2/synth/{c1_regenerated,c2_pseudo,c3_hard_pairs}.jsonl
# → dataset/v2/synth/c4_validated.jsonl  (Stage D input)
```

Each stage also runs standalone (e.g. `python data-gen/stage_a_curator/a2_quality_filter.py --input-dir ...`); the orchestrators accept `--skip a2 a3` for partial reruns.

## Source tags fed to Stage D

| Source                | Origin                              | Blend weight |
|-----------------------|-------------------------------------|--------------|
| `labeled`             | dataset/labeled (human)             | 1.0          |
| `regenerated`         | C.1 (label-conditioned regen)       | 1.0          |
| `pseudo`              | C.2 (High-confidence only)          | 0.5 – 0.7    |
| `synthetic_hard_pos`  | C.3 (2 variants/seed)               | 0.7          |
| `synthetic_hard_neg`  | C.3 (3 variants/seed)               | 0.7          |

## Dependencies

- `nemo_curator` (+ `datasketch`, `ray`/`xenna` for the real GPU path)
- `data_designer` (NVIDIA-NeMo/DataDesigner monorepo, `packages/data-designer`)
- `fasttext`, `scikit-learn` (A.2 trainer, C.2 sibling sampler)
- `openai` (client for vLLM endpoint)
- `pyyaml`

On an Apple-silicon box without CUDA the stages gracefully degrade to a pilot heuristic path; this directory is the canonical GPU-box version.

</details>

---

<a id="한국어"></a>
<details>
<summary><b>한국어</b></summary>

## 개요

약 **1천만 개의 한국어 상품명** (`dataset/get_data/raw.jsonl`) 에 대해 data-plan 의 Stage A · Stage C 단계를 구현.

- **Stage A — Curator**: 퍼지 중복제거, fastText 품질필터, 상품명 정규화, held-out 테스트셋과의 탈오염
- **Stage B — 라벨 분할**: 이미 `dataset/v2/` 에 존재 (사람 라벨 페어, 이 디렉토리 범위 아님)
- **Stage C — DataDesigner**: LLM 기반 근거 재생성, 신뢰도 게이팅 pseudo-labeling, hard-pair 합성, judge validator
- **Stage D — SFT/RL 빌더**: Stage C 출력 하류에 존재

모든 LLM 호출은 vLLM 으로 로컬 서빙되는 **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`** (OpenAI 호환 API, 포트 8000) 를 사용.

## 디렉토리

위 English 블록의 구조와 동일.

## 전체 실행

스크립트는 `-m` 이 아닌 경로로 직접 호출 — 디렉토리명이 `data-gen` (하이픈) 이라 Python 패키지로 import 불가. 각 엔트리포인트가 `data-gen/` 을 `sys.path` 에 주입함.

```bash
# ---------- Stage A: 1천만 raw 에 Curator 적용 -----------------------
python data-gen/stage_a_curator/run_curator.py \
    --config data-gen/configs/curator.yaml
# → dataset/v2/raw_curated/full.jsonl  (A.1/A.2/A.4 후 약 110만 행)

# ---------- Nemotron-3 Super 서빙 (Stage C 용) -----------------------
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --served-model-name nemotron-super --port 8000 &
export NEMOTRON_BASE_URL=http://localhost:8000/v1
export NEMOTRON_API_KEY=EMPTY

# ---------- Stage C: DataDesigner 합성 -------------------------------
python data-gen/stage_c_datadesigner/run_datadesigner.py \
    --config data-gen/configs/datadesigner.yaml
# → dataset/v2/synth/{c1_regenerated,c2_pseudo,c3_hard_pairs}.jsonl
# → dataset/v2/synth/c4_validated.jsonl  (Stage D 입력)
```

각 단계는 단독 실행도 지원 (예: `python data-gen/stage_a_curator/a2_quality_filter.py --input-dir ...`). 오케스트레이터는 `--skip a2 a3` 형태로 재실행 가능.

## Stage D 로 넘어가는 source 태그

| Source                | 출처                                 | 블렌드 가중치   |
|-----------------------|--------------------------------------|-----------------|
| `labeled`             | dataset/labeled (사람 라벨)           | 1.0             |
| `regenerated`         | C.1 (라벨 조건부 재생성)              | 1.0             |
| `pseudo`              | C.2 (High-confidence only)            | 0.5 – 0.7       |
| `synthetic_hard_pos`  | C.3 (seed 당 변형 2개)                 | 0.7             |
| `synthetic_hard_neg`  | C.3 (seed 당 변형 3개)                 | 0.7             |

## 의존성

- `nemo_curator` (+ `datasketch`, GPU 경로용 `ray`/`xenna`)
- `data_designer` (NVIDIA-NeMo/DataDesigner 모노레포 내 `packages/data-designer`)
- `fasttext`, `scikit-learn` (A.2 학습, C.2 sibling sampler)
- `openai` (vLLM 엔드포인트 클라이언트)
- `pyyaml`

CUDA 가 없는 애플 실리콘 환경에서는 각 단계가 휴리스틱 경로로 degrade 됨. 이 디렉토리는 GPU 박스용 정식 버전.

</details>
