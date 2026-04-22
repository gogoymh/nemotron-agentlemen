# demo/env — Product Matching RL Environment

> NeMo-Gym-style RL env, FastAPI-based · NeMo-Gym 스타일 RL 환경 (FastAPI 기반)

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

Self-contained RL environment inside `demo/`. No dependency on the `nemo-gym/` submodule — the Gym's three-server pattern (resources / model / agent) is mirrored here as a single FastAPI app that directly reuses `demo/scraper/` (live Playwright adapters) and `demo/model_client.py` (frozen GenRM judge).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │  demo.env.server  (FastAPI)                  │
                 │                                              │
  task JSONL ─▶  │  /run ─▶ Agent loop ─▶ /v1/chat/completions  │  ─▶ policy_model (vLLM)
                 │            │                                 │
                 │            ├─▶ /tool/search  ─▶ PlatformPool │  ─▶ demo/scraper (cache + Playwright)
                 │            ├─▶ /tool/inspect                 │
                 │            └─▶ /tool/submit_match            │
                 │                                              │
                 │  /verify ─▶ Judge ─▶ /v1/chat/completions    │  ─▶ judge_model (Nemotron-Super)
                 └──────────────────────────────────────────────┘
                              │
                              ▼
                      rollouts.jsonl  (reward, verdicts, trajectory)
```

Two endpoints are active:

- **`/tool/*`** — stateless primitives the policy calls during a rollout.
- **`/verify`** — parses `submit_match` events out of a finished response log and runs the frozen GenRM judge over each submission to compute reward.

`/run` is a convenience that does `rollout → verify` in one call (mirrors `ng_collect_rollouts`'s agent/resources dance without needing the gym CLI).

## Install

From the repo root:

```bash
pip install -r demo/env/requirements.txt
# optional (only if env.live_scraping=true):
playwright install chromium
```

## Judge (Nemotron-3-Super)

Default judge is **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`** served via vLLM's OpenAI API.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --served-model-name nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --port 8001
```

Override via env vars (picked up by `configs/product_matching.yaml`):

```bash
export JUDGE_MODEL_URL=http://localhost:8001/v1
export JUDGE_MODEL_NAME=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
export JUDGE_MODEL_KEY=dummy
```

For local smoke tests without vLLM, flip `judge.mock: true` in the YAML — the `MockMatchModel` (jaccard heuristic from `demo/model_client.py`) is used instead.

## Policy

The policy is any OpenAI-compatible chat/completions endpoint (vLLM):

```bash
vllm serve output/sft-nemotron-4b/final \
    --served-model-name nemotron-match --port 8000
```

```bash
export POLICY_MODEL_URL=http://localhost:8000/v1
export POLICY_MODEL_NAME=nemotron-match
```

## Quickstart

### 1. Launch the env server

```bash
python -m demo.env.server --config demo/env/configs/product_matching.yaml
# → 127.0.0.1:18200
```

Sanity checks:

```bash
curl -s localhost:18200/health | jq
curl -s localhost:18200/tools  | jq '.tools[].name'
curl -s -XPOST localhost:18200/tool/search \
  -H 'content-type: application/json' \
  -d '{"platform":"naver","query":"데시뉴 항균 누빔 베개커버"}' | jq
```

### 2. Collect rollouts

```bash
python -m demo.env.collect_rollouts \
    --input       demo/env/data/example.jsonl \
    --output      results/env_rollouts.jsonl \
    --num-repeats 5 \
    --concurrency 4
```

Each output line:

```json
{
  "task_index": 0,
  "repeat_index": 2,
  "reward": 0.72,
  "reward_breakdown": {"tp": 3, "fp": 1, "fn": 0, "tn": 1, "n_submit": 5, "n_search": 4, "precision": 0.75, "specificity": 1.0, "budget_overflow": 0, "budget_penalty": 0.0, "match_recall": null},
  "verdicts": [{"candidate_url": "...", "decision": "matched", "judge_matched": true, "judge_rationale": "..."}],
  "tool_call_counts": {"search": 4, "submit_match": 5},
  "response": {"output": [/* full Responses-API trajectory */], "output_text": "..."},
  "verifier_metadata": {...}
}
```

### 3. Pass-rate profiling

```bash
python -m demo.env.reward_profile \
    --rollouts       results/env_rollouts.jsonl \
    --output         results/env_profiled.jsonl \
    --pass-threshold 0.5
```

Prints per-task `avg_reward` (pass@1 proxy) and `max_reward` (pass@k upper bound). Use as an SFT-only Phase-0 baseline before wiring to a GRPO/PPO trainer.

## JSONL schema

Input (one task per line):

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a product-matching agent. Use search…"},
      {"role": "user",   "content": "anchor_id=pm-pillow\nanchor_sku=\"…\"\navailable_platforms=[…]"}
    ]
  },
  "verifier_metadata": {
    "anchor_id": "pm-pillow",
    "anchor_sku": "데시뉴 에센스 항균 누빔 베개커버 40x60 50x70 11컬러 2개세트",
    "platforms": ["naver", "ohouse", …],
    "budget_search": 6,
    "budget_submit": 12,
    "min_submissions": 3,
    "gold_matches": [{"product_url_prefix": "https://smartstore.naver.com/…"}]
  }
}
```

`gold_matches` is optional. When present, `reward_breakdown.match_recall` is reported in addition to the judge-only score. Reward itself is judge-scored.

Ships 5 example tasks in `data/example.jsonl` (built from `demo/fanout_presets.json`). `train.jsonl` / `validation.jsonl` are gitignored — build them from your own SKU catalog and drop under `data/`.

## Reward shape

See `demo/env/reward.py` for the full formula. In short:

```
r = precision
  + α · specificity            (correct rejections)
  − β · FP_rate                (over-confident matches)
  − γ · FN_rate                (missed gold)
  − δ · max(0, n_search − budget_search)
  + 0.1 · 𝟙[TP ≥ 1]           (land-something bonus)
```

Clipped to [0, 1]. All weights in `configs/product_matching.yaml`.

## Live scraping

By default the env is **cache-only**: `PlatformPool` serves 13 platforms from `demo/cache/<platform>/*.json` that Phase-0 `fanout_demo.py` has been populating. Flip on live fallback when warming the cache from fresh SKUs during rollout collection:

```yaml
env:
  live_scraping: true       # Playwright falls back on cache miss
  use_system_chrome: true   # recommended for Akamai sites (Coupang etc.)
```

Live mode launches a headless Chromium per search (serialized per platform). CAPTCHA pages are marked `status: "blocked"` in the tool response and the agent branches to another platform — the adversarial dynamics we want RL to learn around.

## Testing

```bash
cd demo/env
pip install -r requirements.txt   # pytest, pytest-asyncio, httpx, fastapi
pytest -q
```

Smoke tests use `judge.mock=true` and a synthetic cache — no vLLM, no Chrome, no network needed. The `/run` test injects a fake OpenAI client so the full rollout → verify path is exercised end-to-end.

## Roadmap

Phase-1 (this directory):

- [x] `search` / `inspect` / `submit_match` tool surface
- [x] Frozen GenRM judge plumbing (Nemotron-Super default, mock fallback)
- [x] `verify()` with precision/specificity/FP/FN/budget reward
- [x] `collect_rollouts` + `reward_profile` CLIs
- [x] End-to-end smoke tests without external deps

Phase-2 candidates:

- [ ] Real `inspect()` that pulls JSON-LD + breadcrumbs from product pages
- [ ] Per-SKU dense intermediate reward (partial credit when `search` yields gold candidates before submission)
- [ ] Gold annotation pipeline — current RL signal is judge-only (self-supervised but loses recall signal)
- [ ] Wire collected rollouts into `RL/` (GRPO) or `experiment/` trainers to close the learn loop
- [ ] Port to `nemo-gym/resources_servers/product_matching/` once exposed externally (module boundaries already match NeMo-Gym conventions)

</details>

---

<a id="한국어"></a>
<details>
<summary><b>한국어</b></summary>

## 개요

`demo/` 내부에서 자립하는 RL 환경. `nemo-gym/` 서브모듈에 의존하지 않고, Gym 의 3-server 패턴 (resources / model / agent) 을 단일 FastAPI 앱으로 구현. `demo/scraper/` (실시간 Playwright 어댑터) 와 `demo/model_client.py` (frozen GenRM judge) 를 그대로 재사용.

## 아키텍처

다이어그램은 위 English 블록과 동일.

활성 엔드포인트:

- **`/tool/*`** — 정책이 롤아웃 중 호출하는 stateless primitives.
- **`/verify`** — 완료된 응답 로그에서 `submit_match` 이벤트를 파싱, 각 제출을 frozen GenRM judge 로 채점해 리워드 계산.

`/run` 은 편의용 — `rollout → verify` 를 한 번에 실행 (nemo-gym CLI 없이 `ng_collect_rollouts` 흐름을 그대로 재현).

## 설치

프로젝트 루트에서:

```bash
pip install -r demo/env/requirements.txt
# 선택 (env.live_scraping=true 일 때만):
playwright install chromium
```

## Judge (Nemotron-3-Super)

기본 judge 는 **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`**, vLLM OpenAI API 로 서빙.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --served-model-name nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --port 8001
```

환경변수로 오버라이드 (`configs/product_matching.yaml` 가 읽음):

```bash
export JUDGE_MODEL_URL=http://localhost:8001/v1
export JUDGE_MODEL_NAME=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
export JUDGE_MODEL_KEY=dummy
```

vLLM 없이 로컬 스모크 테스트는 YAML 에서 `judge.mock: true` → `MockMatchModel` (`demo/model_client.py` 의 jaccard 휴리스틱) 이 대신 사용됨.

## Policy

OpenAI 호환 chat/completions 엔드포인트면 됨 (vLLM):

```bash
vllm serve output/sft-nemotron-4b/final \
    --served-model-name nemotron-match --port 8000
```

```bash
export POLICY_MODEL_URL=http://localhost:8000/v1
export POLICY_MODEL_NAME=nemotron-match
```

## 빠른 시작

### 1. env 서버 실행

```bash
python -m demo.env.server --config demo/env/configs/product_matching.yaml
# → 127.0.0.1:18200
```

확인:

```bash
curl -s localhost:18200/health | jq
curl -s localhost:18200/tools  | jq '.tools[].name'
curl -s -XPOST localhost:18200/tool/search \
  -H 'content-type: application/json' \
  -d '{"platform":"naver","query":"데시뉴 항균 누빔 베개커버"}' | jq
```

### 2. 롤아웃 수집

```bash
python -m demo.env.collect_rollouts \
    --input       demo/env/data/example.jsonl \
    --output      results/env_rollouts.jsonl \
    --num-repeats 5 \
    --concurrency 4
```

각 출력 라인의 스키마는 위 English 블록 JSON 참조.

### 3. Pass-rate 프로파일링

```bash
python -m demo.env.reward_profile \
    --rollouts       results/env_rollouts.jsonl \
    --output         results/env_profiled.jsonl \
    --pass-threshold 0.5
```

태스크별 `avg_reward` (pass@1 proxy) 와 `max_reward` (pass@k 상한) 출력. GRPO/PPO 트레이너에 연결하기 전 SFT-only Phase-0 베이스라인으로 사용.

## JSONL 스키마

입력 (한 줄 = 한 태스크) — 위 English 블록의 JSON 예시 참조.

`gold_matches` 는 옵션. 있으면 judge-only 스코어와 함께 `reward_breakdown.match_recall` 이 같이 리포트됨. 리워드 자체는 judge 기반.

예시 태스크 5개는 `data/example.jsonl` 에 포함 (`demo/fanout_presets.json` 에서 빌드). `train.jsonl` / `validation.jsonl` 은 gitignore — 각자의 SKU 카탈로그에서 빌드해 `data/` 아래 배치.

## 리워드 형태

전체 수식은 `demo/env/reward.py` 참조. 요약:

```
r = precision
  + α · specificity            (맞춘 거절)
  − β · FP_rate                (과신 매칭)
  − γ · FN_rate                (놓친 gold)
  − δ · max(0, n_search − budget_search)
  + 0.1 · 𝟙[TP ≥ 1]           (land-something 보너스)
```

[0, 1] 클립핑. 가중치는 `configs/product_matching.yaml` 에서 조정.

## 실시간 스크래핑

기본은 **캐시 전용** — `PlatformPool` 이 13개 플랫폼을 `demo/cache/<platform>/*.json` (Phase-0 `fanout_demo.py` 가 채워둔) 에서 서빙. 롤아웃 수집 중 새 SKU 로 캐시를 워밍하고 싶을 때만 라이브 폴백 활성화:

```yaml
env:
  live_scraping: true       # 캐시 miss 시 Playwright 폴백
  use_system_chrome: true   # Akamai 사이트 (쿠팡 등) 권장
```

라이브 모드는 검색마다 headless Chromium 을 띄움 (플랫폼별 직렬화). CAPTCHA 페이지는 툴 응답에 `status: "blocked"` 로 마킹되고 agent 는 다른 플랫폼으로 분기 — RL 이 학습했으면 하는 adversarial dynamics.

## 테스트

```bash
cd demo/env
pip install -r requirements.txt   # pytest, pytest-asyncio, httpx, fastapi
pytest -q
```

스모크 테스트는 `judge.mock=true` + 합성 캐시 — vLLM/Chrome/네트워크 불필요. `/run` 테스트는 fake OpenAI 클라이언트를 주입해 rollout → verify 전체 경로를 end-to-end 로 확인.

## 로드맵

Phase-1 (현재):

- [x] `search` / `inspect` / `submit_match` 툴 인터페이스
- [x] Frozen GenRM judge 배선 (Nemotron-Super 기본, mock 폴백)
- [x] precision / specificity / FP / FN / budget 기반 `verify()`
- [x] `collect_rollouts` + `reward_profile` CLI
- [x] 외부 의존 없는 end-to-end 스모크 테스트

Phase-2 후보:

- [ ] JSON-LD + breadcrumb 을 상품 페이지에서 뽑는 실제 `inspect()`
- [ ] SKU 별 dense intermediate reward (`search` 가 gold 후보 띄웠는데 아직 submit 전이면 부분 보상)
- [ ] Gold annotation 파이프라인 — 현재 RL 시그널은 judge-only (self-supervised 이지만 recall 신호 약함)
- [ ] 수집된 롤아웃을 `RL/` (GRPO) 또는 `experiment/` 트레이너에 연결해 학습 루프 완성
- [ ] 외부 공개 단계에서 `nemo-gym/resources_servers/product_matching/` 로 포팅 (모듈 경계가 이미 NeMo-Gym 규약에 맞춰져 있음)

</details>
