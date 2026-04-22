# demo — Product Matching Live Demo

> Drive the trained matching model against real Korean e-commerce search UIs · 학습된 매칭 모델을 실제 한국 커머스 검색 UI 위에서 구동

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

## Layout

```
demo/
  demo.py              ← main (argparse CLI, cycles through curated queries)
  fanout_demo.py       ← video/demo mode: parallel search across N sites
  queries.json         ← curated SKU queries
  fanout_presets.json  ← SKU → platform set presets (pillow / tshirt / sofa / …)
  model_client.py      ← vLLM OpenAI-compatible client (+ MockMatchModel dry-run)
  scraper/
    base.py            ← SearchAdapter base + navigation + manual CAPTCHA hook
    coupang.py         ← Coupang (home → search box + /vp/products anchor parse)
    naver.py           ← Naver Shopping (home → search box + __NEXT_DATA__ fallback)
    musinsa.py         ← MUSINSA (/products · /goods anchor parse)
    generic.py         ← Tier-1/Tier-2 shops via SITE_CONFIGS
  env/                 ← NeMo-Gym-style RL env (see env/README.md)
  cache/               ← cached search results + debug DOM dumps
```

## Install

```bash
# from repo root
python3.12 -m venv .venv && source .venv/bin/activate
pip install playwright requests ipython
playwright install chromium
```

> The demo prefers **system Chrome** over bundled Chromium (bot-detection evasion). On macOS: `brew install --cask google-chrome`.

## Fanout mode (video-friendly, N sites in parallel)

Opens the same SKU on multiple e-commerce sites **in parallel** and aggregates their matching links into a single summary view.

```bash
# preset (fanout_presets.json)
python fanout_demo.py --preset pillow --mock-model    # 6 sites
python fanout_demo.py --preset tshirt --mock-model    # 4 fashion sites

# explicit
python fanout_demo.py \
  --sku "무신사 스탠다드 오버사이즈 크루넥 반팔 티셔츠 블랙" \
  --query "무신사 스탠다드 오버사이즈 티셔츠" \
  --platforms musinsa,coupang,naver,ohouse,gmarket,shinsegaemall
```

**How it works**:

1. Each platform gets an independent Chromium/Chrome process, positioned in a grid via `--window-position x,y --window-size W,H`. Processes are fully isolated — no CDP popup races.
2. Per-platform profiles are stored in `~/.cache/nemotron-fanout/<platform>/` — reruns accumulate cookies, which softens bot detection over time.
3. Terminal event log: `🚀 start → 🪟 window → 📄 loaded → 🔍 parsing → ✓ done`.
4. On completion, writes a card-grid HTML to `cache/fanout/<slug>-<ts>.html` and opens it in the default browser. Platform windows are **not closed automatically** (so a demo recording can capture them) — Cmd+Q to close.

**Browser choice**:

- Default: **system Chrome** (`channel="chrome"`) — Coupang / Akamai sites block bundled Chromium by UA / TLS fingerprint. Install first: `brew install --cask google-chrome`.
- `--bundled-chromium` forces Playwright's bundled Chromium (works for permissive sites like Naver / MUSINSA).

**Window layout** (default `--win-w 640 --win-h 720`, ≤6 sites uses a 3-column grid):

```bash
python fanout_demo.py --preset pillow --cols 3 --win-w 520 --win-h 680   # 6 → 3×2
python fanout_demo.py --preset tshirt --cols 4 --win-w 440 --win-h 700   # 4 → row
```

**Execution mode**:

- **Default (sequential)** — opens one platform at a time. On CAPTCHA, terminal blocks until `Enter` — a human solves the puzzle in the window and hits Enter to resume parsing. Good for demo recordings and warm-up runs.
- `--parallel` — opens N windows simultaneously. On CAPTCHA, the worker marks `BLOCKED` and falls back to cache. Good once all profiles are already warmed up.

**Cache policy** — live parsing results are auto-saved to `cache/<platform>/<query_slug>.json`:

- **default**: try live first, auto-fallback to cache on any failure (BLOCKED / 0 candidates / error).
- `--prefer-cache`: hit cache first; only open a browser on miss.
- `--no-cache-fallback`: pure live, no fallback.
- `--no-write-cache`: don't persist live results.

## Attach mode (recommended for Coupang / Akamai)

Coupang's Akamai setup blocks Playwright-launched Chromes on sight. Workaround: the user launches Chrome separately, and Playwright **attaches** over CDP port 9222.

### 1. Launch the dedicated Chrome

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.cache/nemotron-demo-chrome-attach \
  --disable-popup-blocking
```

Keep this window open for the duration of the demo.

### 2. Warm per-site cookies (first run only)

Manually visit each target in the attached Chrome so real-user cookies and fingerprints accumulate:

- `coupang.com`
- `shopping.naver.com/ns/home`
- `musinsa.com`

### 3. Run

```bash
cd demo
python demo.py                              # all 5 queries (coupang×2, naver×2, musinsa×1)
python demo.py --platforms naver,musinsa    # skip coupang
python demo.py --query-idx 2                # single query
python demo.py --mock-model                 # heuristic only, no LLM
python demo.py --help
```

## CLI flags (`demo.py`)

| Flag | Default | Description |
|---|---|---|
| `--platforms` | `all` | comma-list of `coupang,naver,musinsa` or `all` |
| `--query-idx` | `None` | single index (0-based) |
| `--mode` | `attach` | `attach` (port 9222) or `launch` (Playwright launches Chrome) |
| `--mock-model` | off | heuristic judge, no model server |
| `--use-cache` | off | skip live if cache exists |
| `--no-write-cache` | off | don't persist live results |
| `--max-results` | 12 | max candidates per query |

## Model serving (optional)

When not using `--mock-model`, serve vLLM in a separate terminal:

```bash
vllm serve output/sft-nemotron-4b/final \
    --served-model-name nemotron-match \
    --port 8000
```

Env overrides:

```bash
export MATCH_MODEL_URL=http://localhost:8000/v1
export MATCH_MODEL_NAME=nemotron-match
```

## CAPTCHA / block handling

On CAPTCHA or Akamai `Access Denied`, the console prompts for input:

```
🔒 [쿠팡] CAPTCHA/blocked detected.
   Solve it in the browser window, then press Enter here.
```

Solve in the browser, hit Enter **once**, parsing resumes. If the block is IP-level, even a VPN won't help — attach mode + prior cookie warm-up covers most cases.

## Supported platforms

**Custom adapters (per-site logic)**:

| Site | File | Block level | Notes |
|---|---|:---:|---|
| Coupang `coupang` | `scraper/coupang.py` | high (Akamai) | home → search box + `/vp/products/` anchor |
| Naver Shopping `naver` | `scraper/naver.py` | med | home → search box + `__NEXT_DATA__` fallback |
| MUSINSA `musinsa` | `scraper/musinsa.py` | low | direct link + `/products` · `/goods` anchor |

**Generic adapter** (`scraper/generic.py` SITE_CONFIGS):

| key | Site | home-nav | Notes |
|---|---|:---:|---|
| `ohouse` | 오늘의집 | ✓ | product_platform — core for matching |
| `eleventh_street` | 11번가 | - | |
| `lotteon` | 롯데온 | ✓ | strong bot blocker |
| `gmarket` | 지마켓 | ✓ | strong bot blocker |
| `emartinternetshopping` | 이마트몰 (SSG) | - | |
| `auction` | 옥션 | ✓ | strong bot blocker |
| `hmall` | Hyundai Hmall | ✓ | |
| `gsshop` | GS샵 | ✓ | |
| `shinsegaemall` | 신세계몰 (SSG) | ✓ | Chakra selectors |

> CJ onstyle (cjonstyle) is currently excluded — the search subdomain (`display.cjonstyle.com`) returns "invalid/removed" even for manual browsing.

## Troubleshooting

- `Access Denied` / `errors.edgesuite.net` → Akamai. Check attach mode + cookie warm-up. IP-level block needs a different exit.
- `Please use the Async API instead` → `sync_playwright` conflicts with notebook asyncio loop. Run as `python demo.py`, not in a notebook.
- 0 candidates → inspect `cache/debug/<site>-*.png|html`. DOM changed → update parser selectors.

## Demo tips

- **Pre-run**: warm the cache live before the demo; on the day, `--use-cache` falls back cleanly if the network misbehaves.
- **Curate queries**: keep only the queries that parse reliably.
- **Window size**: `1280×900` default — adjust to projector resolution.
- **Comparison demo**: run the same query through an external LLM API to contrast latency / cost.

</details>

---

<a id="한국어"></a>
<details>
<summary><b>한국어</b></summary>

## 구성

```
demo/
  demo.py              ← 메인 (argparse CLI, 큐레이션된 쿼리 순회)
  fanout_demo.py       ← 영상/시연 전용 (N개 사이트 병렬 검색)
  queries.json         ← 사전 큐레이션된 SKU 쿼리
  fanout_presets.json  ← SKU → 플랫폼 세트 프리셋 (pillow / tshirt / sofa / …)
  model_client.py      ← vLLM OpenAI-호환 클라이언트 (+ MockMatchModel 드라이런)
  scraper/
    base.py            ← SearchAdapter 베이스 + navigate / CAPTCHA 수동 해결 훅
    coupang.py         ← 쿠팡 (홈 → 검색창 + `/vp/products/` 앵커 파싱)
    naver.py           ← 네이버쇼핑 (홈 → 검색창 + `__NEXT_DATA__` fallback)
    musinsa.py         ← 무신사 (`/products` · `/goods` 앵커 파싱)
    generic.py         ← Tier-1/2 사이트 어댑터 (SITE_CONFIGS)
  env/                 ← NeMo-Gym 스타일 RL 환경 (env/README.md 참조)
  cache/               ← 검색 결과 캐시 + 디버그 DOM 덤프
```

## 설치

```bash
# 프로젝트 루트에서
python3.12 -m venv .venv && source .venv/bin/activate
pip install playwright requests ipython
playwright install chromium
```

> 번들 Chromium 이 아닌 **시스템 Chrome** 을 사용 (봇 탐지 우회). macOS: `brew install --cask google-chrome`.

## Fanout 모드 (영상용, N개 사이트 동시 검색)

하나의 SKU 를 여러 커머스에서 **병렬로** 열어 검색하고, 각 사이트의 매칭 링크를 한 화면에 모아 보여주는 시연 모드.

```bash
# 프리셋 (fanout_presets.json)
python fanout_demo.py --preset pillow --mock-model   # 6개 사이트
python fanout_demo.py --preset tshirt --mock-model   # 패션 4곳

# 직접 지정
python fanout_demo.py \
  --sku "무신사 스탠다드 오버사이즈 크루넥 반팔 티셔츠 블랙" \
  --query "무신사 스탠다드 오버사이즈 티셔츠" \
  --platforms musinsa,coupang,naver,ohouse,gmarket,shinsegaemall
```

**동작**:

1. 플랫폼마다 독립 Chromium/Chrome 프로세스를 `--window-position x,y --window-size W,H` 로 격자 배치해 띄움. 프로세스가 완전히 분리돼 CDP popup race 없음.
2. 프로파일은 `~/.cache/nemotron-fanout/<platform>/` 에 플랫폼별 저장 → 재실행 시 쿠키 누적돼 봇 탐지 완화.
3. 터미널 이벤트 로그: `🚀 start → 🪟 window → 📄 loaded → 🔍 parsing → ✓ done`.
4. 끝나면 `cache/fanout/<slug>-<ts>.html` 카드 그리드 요약 생성 + 기본 브라우저 자동 오픈. 각 플랫폼 창은 **자동 종료 안 함** (영상 촬영용) — Cmd+Q 로 직접 닫기.

**브라우저 선택**:

- 기본 **시스템 Chrome** (`channel="chrome"`) — 쿠팡/Akamai 는 번들 Chromium 을 UA/TLS 핑거프린트로 차단. 사전 설치: `brew install --cask google-chrome`.
- `--bundled-chromium` 플래그로 번들 Chromium 강제 가능 (네이버·무신사 등 관대한 사이트 전용).

**창 배치** (기본 `--win-w 640 --win-h 720`, ≤6 개면 3열 그리드):

```bash
python fanout_demo.py --preset pillow --cols 3 --win-w 520 --win-h 680   # 6 → 3×2
python fanout_demo.py --preset tshirt --cols 4 --win-w 440 --win-h 700   # 4 → 1줄
```

**실행 모드**:

- **기본 (순차)** — 플랫폼을 1개씩 띄움. CAPTCHA 감지 시 터미널이 `Enter` 대기 → 사람이 창에서 퍼즐 풀고 Enter 치면 파싱 진행. 영상 촬영 · warm-up 용.
- `--parallel` — N 개 창 동시 실행. CAPTCHA 감지 시 즉시 `BLOCKED` 마킹 후 캐시 폴백. 전 사이트 프로파일이 워밍된 뒤 fanout 스냅샷 뽑을 때 쓰는 모드.

**캐시 정책** — `cache/<platform>/<query_slug>.json` 자동 저장:

- **기본**: 라이브 시도 → 실패 (BLOCKED / 0 cands / error) 시 캐시 자동 폴백.
- `--prefer-cache`: 캐시 hit 이면 라이브 스킵.
- `--no-cache-fallback`: 폴백 비활성화.
- `--no-write-cache`: 라이브 성공해도 캐시 저장 안 함.

## Attach 모드 (쿠팡/Akamai 권장)

쿠팡의 Akamai 는 Playwright 가 직접 띄운 Chrome 을 즉시 차단. 해결: 사용자가 별도로 Chrome 을 띄우고 Playwright 가 CDP 9222 포트로 **attach**.

### 1. 전용 Chrome 창 띄우기

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.cache/nemotron-demo-chrome-attach \
  --disable-popup-blocking
```

이 창은 시연이 끝날 때까지 계속 열어둠.

### 2. 사이트별 세션 쿠키 심기 (최초 1회)

붙어 있는 Chrome 에서 **손으로** 직접 방문 (사람 트래픽 쿠키 · 핑거프린트 축적):

- `coupang.com`
- `shopping.naver.com/ns/home`
- `musinsa.com`

### 3. 실행

```bash
cd demo
python demo.py                              # 전체 5개 쿼리 (coupang×2, naver×2, musinsa×1)
python demo.py --platforms naver,musinsa    # 쿠팡 스킵
python demo.py --query-idx 2                # 특정 쿼리 하나만
python demo.py --mock-model                 # 모델 서버 없이 휴리스틱
python demo.py --help
```

## CLI 플래그 (`demo.py`)

| 플래그 | 기본 | 설명 |
|---|---|---|
| `--platforms` | `all` | `coupang,naver,musinsa` 쉼표 목록 또는 `all` |
| `--query-idx` | `None` | 특정 인덱스만 실행 (0-based) |
| `--mode` | `attach` | `attach` (9222) 또는 `launch` (Playwright 직접) |
| `--mock-model` | off | 모델 서버 없이 휴리스틱 판정 |
| `--use-cache` | off | 캐시 있으면 라이브 스킵 |
| `--no-write-cache` | off | 결과 캐시 저장 안 함 |
| `--max-results` | 12 | 각 쿼리당 후보 개수 상한 |

## 모델 서빙 (선택)

`--mock-model` 안 쓸 땐 별도 터미널에서 vLLM 실행:

```bash
vllm serve output/sft-nemotron-4b/final \
    --served-model-name nemotron-match \
    --port 8000
```

환경변수로 엔드포인트 변경:

```bash
export MATCH_MODEL_URL=http://localhost:8000/v1
export MATCH_MODEL_NAME=nemotron-match
```

## CAPTCHA / 차단 처리

CAPTCHA 나 Akamai `Access Denied` 감지 시 콘솔 프롬프트:

```
🔒 [쿠팡] CAPTCHA/차단 감지.
   브라우저 창에서 직접 풀고, 여기에 Enter 를 눌러주세요.
```

브라우저에서 풀고 터미널 Enter **한 번** → 파싱 이어짐. IP 단위 차단이면 VPN 도 안 통함 — attach 모드 + 사전 방문으로 대부분 해결.

## 지원 플랫폼

**커스텀 어댑터 (사이트별 특수 로직)**:

| 사이트 | 파일 | 봇 차단 | 비고 |
|---|---|:---:|---|
| 쿠팡 `coupang` | `scraper/coupang.py` | 강 (Akamai) | 홈 → 검색창 + `/vp/products/` 앵커 |
| 네이버쇼핑 `naver` | `scraper/naver.py` | 중 | 홈 → 검색창 + `__NEXT_DATA__` fallback |
| 무신사 `musinsa` | `scraper/musinsa.py` | 약 | 직링크 + `/products` · `/goods` 앵커 |

**제네릭 어댑터** (`scraper/generic.py` SITE_CONFIGS):

| key | 사이트 | 홈→검색창 | 비고 |
|---|---|:---:|---|
| `ohouse` | 오늘의집 | ✓ | product_platform — 매칭 데모 핵심 |
| `eleventh_street` | 11번가 | - | |
| `lotteon` | 롯데온 | ✓ | 봇 차단 강함 |
| `gmarket` | 지마켓 | ✓ | 봇 차단 강함 |
| `emartinternetshopping` | 이마트몰 (SSG) | - | |
| `auction` | 옥션 | ✓ | 봇 차단 강함 |
| `hmall` | 현대Hmall | ✓ | |
| `gsshop` | GS샵 | ✓ | |
| `shinsegaemall` | 신세계몰 (SSG) | ✓ | Chakra 셀렉터 |

> CJ온스타일 (cjonstyle) 은 검색결과 서브도메인 (`display.cjonstyle.com`) 이 수동 접속에서도 "잘못된 주소/삭제됨" 으로 떠서 현재 제외.

## 트러블슈팅

- `Access Denied` / `errors.edgesuite.net` → Akamai. attach 모드 · 사전 방문 확인. IP 차단이면 경로 변경 필요.
- `Please use the Async API instead` → 노트북 커널의 asyncio 루프 충돌. 노트북 대신 `python demo.py` 실행.
- 후보 0건 → `cache/debug/<site>-*.png|html` 확인. DOM 변경이면 파서 셀렉터 갱신.

## 시연 팁

- **사전 예행**: 행사 직전 라이브 한 번 돌려 `cache/` 채워두고, 당일 문제 생기면 `--use-cache` 로 즉시 fallback.
- **쿼리 선별**: `queries.json` 에 정상 파싱되는 쿼리만 남김.
- **브라우저 크기**: 1280×900 기본 — 발표 해상도에 맞춰 조정.
- **비교 데모**: 같은 쿼리를 외부 LLM API 로 돌려 지연/비용 대비.

</details>
