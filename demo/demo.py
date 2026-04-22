"""데모 메인 — Jupyter 노트북처럼 셀 단위로 실행 가능.

VSCode / Jupyter에서 `# %%` 블록이 셀로 인식된다.
Ctrl/Cmd+Enter 로 셀을 하나씩 실행하며 시연한다.

Prerequisites:
    pip install playwright requests
    playwright install chromium

    # 모델 서빙 (별도 터미널)
    # python src/inference.py --model nemotron-4b --checkpoint output/sft-nemotron-4b/final --serve
    # 또는 vLLM 직접:
    # vllm serve <checkpoint> --served-model-name nemotron-match --port 8000
"""

# %% [markdown]
# # Product Matching Demo
#
# 1. 사전에 정한 SKU 쿼리 선택
# 2. 해당 커머스 사이트를 브라우저로 열고 검색
# 3. 결과 카드 파싱 → 학습된 로컬 모델로 매칭 판정
# 4. 매칭된 후보를 하이라이트

# %%
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from IPython.display import HTML, display
from playwright.sync_api import sync_playwright

from scraper import ADAPTERS
from model_client import MatchModel, MockMatchModel

# 경로
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
CACHE_DIR = ROOT / "cache"
QUERIES = json.loads((ROOT / "queries.json").read_text(encoding="utf-8"))["queries"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="Product matching live demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--platforms", default="all",
        help="쉼표로 구분한 플랫폼 필터 (coupang,naver,musinsa). 'all' 이면 전체.",
    )
    p.add_argument(
        "--query-idx", type=int, default=None,
        help="특정 쿼리 인덱스만 실행. 지정 시 --platforms 무시.",
    )
    p.add_argument(
        "--mode", choices=["launch", "attach"], default="attach",
        help="launch: Playwright가 Chrome 띄움 / attach: 9222 Chrome에 붙음",
    )
    p.add_argument("--mock-model", action="store_true", help="모델 서버 없이 휴리스틱으로 판정")
    p.add_argument("--use-cache", action="store_true", help="캐시 있으면 라이브 스킵")
    p.add_argument("--no-write-cache", action="store_true", help="결과를 캐시에 저장하지 않음")
    p.add_argument("--max-results", type=int, default=12)
    # Jupyter/IPython 에서 실행될 때 들어오는 커널 인자 무시
    args, _ = p.parse_known_args()
    return args


ARGS = _parse_args()

USE_MOCK_MODEL = ARGS.mock_model
model = MockMatchModel() if USE_MOCK_MODEL else MatchModel()

print(f"Loaded {len(QUERIES)} queries · model={'mock' if USE_MOCK_MODEL else model.model_name}")

# %% [markdown]
# ## Step 1 — 브라우저 시작 (1회)

# %%
USE_CACHE_FIRST = ARGS.use_cache
WRITE_CACHE = not ARGS.no_write_cache
BROWSER_MODE = ARGS.mode
QUERY_IDX = ARGS.query_idx
if ARGS.platforms.lower() == "all":
    PLATFORMS = None
else:
    PLATFORMS = [p.strip() for p in ARGS.platforms.split(",") if p.strip()]

playwright = sync_playwright().start()

if BROWSER_MODE == "attach":
    # 먼저 별도 터미널에서:
    #   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    #       --remote-debugging-port=9222 \
    #       --user-data-dir=$HOME/.cache/nemotron-demo-chrome-attach
    # 이렇게 띄운 뒤, 쿠팡/네이버/무신사 홈에 각각 한번 들어갔다 나오면 세션이 잡힘.
    browser = playwright.chromium.connect_over_cdp("http://localhost:9222")
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else context.new_page()
else:
    # Persistent profile + 실제 Chrome + --enable-automation 제거 + 스텔스 패치
    PROFILE_DIR = str(Path.home() / ".cache" / "nemotron-demo-chrome")
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=PROFILE_DIR,
        channel="chrome",
        headless=False,
        viewport={"width": 1280, "height": 900},
        locale="ko-KR",
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
    )
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'languages', {get: () => ['ko-KR', 'ko', 'en-US', 'en']});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        window.chrome = { runtime: {} };
    """)
    page = context.pages[0] if context.pages else context.new_page()
    browser = None

# %% [markdown]
# ## Step 2 — 후보 카드 리치 디스플레이 함수 정의

# %%
def render_candidates(cands, verdicts=None):
    """후보 리스트를 HTML 그리드로. verdicts가 주어지면 matched 는 녹색 테두리."""
    html = ["""
    <style>
      .demo-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; font-family:Pretendard,sans-serif; }
      .demo-card { border:1px solid #e2e8f0; border-radius:12px; padding:12px; background:#fff; position:relative; }
      .demo-card.matched { border:2px solid #059669; background:#ecfdf5; }
      .demo-card.nomatch { opacity:0.55; }
      .demo-card img { width:100%; height:140px; object-fit:cover; border-radius:8px; background:#f8fafc; }
      .demo-card .title { font-size:13px; line-height:1.4; margin-top:8px; color:#0f172a; font-weight:500; max-height:54px; overflow:hidden; }
      .demo-card .brand { font-size:11px; color:#64748b; margin-top:4px; }
      .demo-card .price { font-size:15px; color:#0f172a; font-weight:700; margin-top:6px; }
      .demo-card .badge { position:absolute; top:8px; right:8px; padding:3px 8px; border-radius:10px; font-size:11px; font-weight:700; }
      .badge.matched { background:#059669; color:#fff; }
      .badge.nomatch { background:#e2e8f0; color:#64748b; }
      .badge.fail    { background:#fef2f2; color:#dc2626; }
    </style>
    <div class='demo-grid'>
    """]
    for i, c in enumerate(cands):
        cls = ""
        badge = ""
        if verdicts:
            v = verdicts[i]
            if v.status != "OK":
                badge = f"<span class='badge fail'>ERR</span>"
            elif v.matched:
                cls = "matched"
                badge = "<span class='badge matched'>MATCH</span>"
            else:
                cls = "nomatch"
                badge = "<span class='badge nomatch'>—</span>"
        img = c.image_url or ""
        html.append(f"""
        <div class='demo-card {cls}'>
          {badge}
          {('<img src="' + img + '" />') if img else '<div style="height:140px;background:#f8fafc;border-radius:8px;"></div>'}
          <div class='brand'>{c.brand or '·'}</div>
          <div class='title'>{c.title}</div>
          <div class='price'>{c.price}</div>
        </div>
        """)
    html.append("</div>")
    display(HTML("".join(html)))

# %% [markdown]
# ## Step 3 — 쿼리 순회 실행
#
# `QUERY_IDX=None` 이면 전체, 정수면 그 인덱스만.

# %%
if QUERY_IDX is not None:
    targets = [QUERIES[QUERY_IDX]]
else:
    targets = QUERIES if PLATFORMS is None else [q for q in QUERIES if q["platform"] in PLATFORMS]

print(f"\n총 {len(targets)}개 쿼리 실행 예정: " +
      ", ".join(f"{q['platform']}:{q['id']}" for q in targets))

summary = []
for qi, q in enumerate(targets):
    print(f"\n{'='*70}")
    print(f"[{qi+1}/{len(targets)}] {q['id']} · {q['platform']}")
    print(f"  Query    : {q['query']}")
    print(f"  SKU name : {q['sku_name']}")
    if q.get("notes"):
        print(f"  Notes    : {q['notes']}")
    print(f"{'-'*70}")

    adapter = ADAPTERS[q["platform"]](cache_dir=CACHE_DIR)
    try:
        candidates = adapter.search(
            page, q["query"],
            use_cache=USE_CACHE_FIRST,
            write_cache=WRITE_CACHE,
            max_results=12,
        )
    except Exception as e:
        print(f"  ✗ 검색 실패: {e}")
        summary.append((q["id"], q["platform"], 0, 0, "ERROR"))
        continue

    print(f"  ✓ {adapter.display_name}: {len(candidates)} candidates")
    if not candidates:
        summary.append((q["id"], q["platform"], 0, 0, "NO_CANDS"))
        continue

    render_candidates(candidates)

    verdicts = []
    for i, c in enumerate(candidates):
        v = model.predict(q["sku_name"], c.title)
        verdicts.append(v)
        status = "✓ MATCH" if v.matched else ("✗"
                 if v.status == "OK" else f"!{v.status}")
        print(f"    [{i:02d}] {status:8s} ({v.latency_s*1000:.0f}ms)  {c.title[:60]}")

    n_match = sum(1 for v in verdicts if v.matched)
    n_err = sum(1 for v in verdicts if v.status != "OK")
    print(f"  ▶ {len(verdicts)} 후보 중 {n_match}건 matched ({n_err}건 에러)")
    render_candidates(candidates, verdicts)
    summary.append((q["id"], q["platform"], len(candidates), n_match, "OK" if n_err == 0 else f"ERR×{n_err}"))

# 최종 요약
print(f"\n\n{'='*70}\n전체 요약\n{'='*70}")
print(f"{'ID':<20s} {'PLATFORM':<10s} {'CANDS':>6s} {'MATCH':>6s}  STATUS")
for sid, plat, ncand, nmatch, st in summary:
    print(f"{sid:<20s} {plat:<10s} {ncand:>6d} {nmatch:>6d}  {st}")

# %% [markdown]
# ## Step 4 — 정리

# %%
try:
    if BROWSER_MODE == "attach":
        # attach 모드에선 브라우저를 종료하지 않는다 (사용자가 띄운 거라서)
        pass
    else:
        context.close()
    playwright.stop()
except Exception:
    pass
print("✓ closed")
