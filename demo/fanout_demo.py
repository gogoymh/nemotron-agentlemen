"""Fanout demo — 동일 SKU를 N개 커머스에서 **동시에** 검색.

워커마다 **자기 Playwright 인스턴스**가 자기 Chrome 창을 띄움.
  - `--window-position`/`--window-size` 로 OS-레벨 창 배치 → 그리드 타일.
  - 창끼리 완전 독립 → CDP popup 레이스 없음, about:blank 버그 없음.
  - 각 창은 플랫폼별 persistent profile 로 유지 → 쿠키 누적.

사용 예:
  python fanout_demo.py --preset pillow --mock-model
  python fanout_demo.py --sku "무신사 스탠다드 오버사이즈 크루넥 반팔 티셔츠 블랙" \\
                        --query "무신사 스탠다드 오버사이즈 티셔츠" \\
                        --platforms musinsa,coupang,naver,ohouse,gmarket,shinsegaemall
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from dataclasses import asdict

from playwright.sync_api import sync_playwright

from scraper import ADAPTERS
from scraper.base import Candidate
from model_client import MatchModel, MockMatchModel, MatchVerdict

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
FANOUT_DIR = CACHE_DIR / "fanout"
PROFILE_ROOT = Path.home() / ".cache" / "nemotron-fanout"

# 프로세스 끝까지 살려둬서 영상 촬영 중 창이 닫히지 않게 함.
# run_platform 이 리턴하면 로컬 pw/context 가 GC 되며 Chrome subprocess 를 kill 하는
# 버그를 막기 위한 강제 reference holder.
_KEEPALIVE: List[tuple] = []

STEALTH_INIT_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['ko-KR', 'ko', 'en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = { runtime: {} };
"""

# ── 로그 ────────────────────────────────────────────
_log_lock = threading.Lock()
_t_start = time.time()


def _log(tag: str, platform: str, msg: str) -> None:
    with _log_lock:
        elapsed = time.time() - _t_start
        print(f"[{elapsed:6.2f}s] {tag} {platform:<18s} {msg}", flush=True)


# ── 캐시 ────────────────────────────────────────────
def _load_cache(adapter, query: str, max_results: int) -> Optional[List[Candidate]]:
    """어댑터의 `_cache_path` 규칙을 그대로 써서 캐시 로드."""
    p = adapter._cache_path(query)
    if not p or not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [Candidate(**c) for c in data][:max_results]
    except Exception:
        return None


def _save_cache(adapter, query: str, candidates: List[Candidate]) -> Optional[Path]:
    p = adapter._cache_path(query)
    if not p:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


# ── 그리드 ──────────────────────────────────────────
def _grid_position(idx: int, total: int, w: int, h: int, cols: Optional[int] = None,
                   x_offset: int = 20, y_offset: int = 40, gap: int = 8) -> Tuple[int, int]:
    if cols is None:
        if total <= 3:
            cols = total
        elif total <= 6:
            cols = 3
        else:
            cols = math.ceil(math.sqrt(total))
    row = idx // cols
    col = idx % cols
    return col * (w + gap) + x_offset, row * (h + gap) + y_offset


# ── 결과 ────────────────────────────────────────────
@dataclass
class WorkerResult:
    platform: str
    display_name: str
    status: str = "pending"
    n_candidates: int = 0
    n_matched: int = 0
    top_title: str = ""
    top_url: str = ""
    top_image: str = ""
    elapsed: float = 0.0
    error: str = ""
    candidates_preview: List[dict] = field(default_factory=list)


# ── 워커 ────────────────────────────────────────────
def _match_and_fill(res: WorkerResult, candidates: List[Candidate], sku_name: str,
                    model, source: str) -> None:
    """후보 → 모델 매칭 → WorkerResult 필드 채우기. source = 'live' or 'cache'."""
    top = None
    n_matched = 0
    for c in candidates:
        v = model.predict(sku_name, c.title)
        if v.matched:
            n_matched += 1
            if top is None:
                top = c

    res.n_candidates = len(candidates)
    res.n_matched = n_matched
    pick = top if top is not None else candidates[0]
    res.top_title = pick.title
    res.top_url = pick.product_url
    res.top_image = pick.image_url
    res.status = "done_cache" if source == "cache" else "done"
    res.candidates_preview = [
        {"title": c.title, "url": c.product_url, "image": c.image_url,
         "price": c.price, "brand": c.brand}
        for c in candidates[:6]
    ]


def run_platform(
    platform: str,
    query: str,
    sku_name: str,
    model,
    max_results: int,
    prefer_cache: bool,
    cache_fallback: bool,
    write_cache: bool,
    win_x: int,
    win_y: int,
    win_w: int,
    win_h: int,
    use_system_chrome: bool,
    block_on_captcha: bool = True,
    shared_pw=None,
) -> WorkerResult:
    """한 플랫폼당 **독립된 Chrome/Chromium 창**을 띄우고 검색 → 매칭.

    block_on_captcha:
      - True (순차 모드 기본) → CAPTCHA 감지 시 Enter 칠 때까지 블로킹 대기
      - False (병렬 모드) → 감지 즉시 BLOCKED 마킹 후 다음으로

    캐시 정책:
      - prefer_cache=True  → 캐시 있으면 라이브 스킵 (빠른 재생)
      - cache_fallback=True → 라이브 실패(BLOCKED/0 cands/error) 시 캐시로 폴백
      - write_cache=True   → 라이브 성공 시 캐시 저장
    """
    t0 = time.time()
    adapter = ADAPTERS[platform](cache_dir=CACHE_DIR)
    display = adapter.display_name
    res = WorkerResult(platform=platform, display_name=display)
    # 병렬 모드에서만 input() 블로킹 우회. 순차 모드에선 adapter 기본 블로킹 유지.
    if not block_on_captcha:
        adapter.wait_for_manual_captcha = lambda page, timeout_s=0: False  # type: ignore

    _log("🚀", display, f"reset(env)  obs={query!r}")

    # 0) prefer-cache 경로 — 라이브 브라우저 띄우지도 않음
    if prefer_cache:
        cached = _load_cache(adapter, query, max_results)
        if cached:
            _log("📼", display, f"replay buffer hit · {len(cached)} trajectories (skip live rollout)")
            _match_and_fill(res, cached, sku_name, model, source="cache")
            res.elapsed = time.time() - t0
            _log("🏆", display,
                 f"reward = +{res.n_matched} / {len(cached)} (replay) → {res.top_url}")
            return res
        else:
            _log("· ", display, "no buffered trajectory, rolling out live")

    profile = PROFILE_ROOT / platform
    profile.mkdir(parents=True, exist_ok=True)

    pw = None
    pw_owned = False  # 이 워커가 pw 를 직접 start 했는지 (병렬 모드에서 True)
    context = None
    live_status = None  # 'blocked' | 'no_cands' | 'error' | None(=success)
    try:
        if shared_pw is not None:
            pw = shared_pw
        else:
            # 병렬 모드: 각 스레드가 자기 sync_playwright 를 가짐.
            pw = sync_playwright().start()
            pw_owned = True
        launch_kwargs = dict(
            user_data_dir=str(profile),
            headless=False,
            viewport=None,
            locale="ko-KR",
            args=[
                "--disable-blink-features=AutomationControlled",
                f"--window-position={win_x},{win_y}",
                f"--window-size={win_w},{win_h}",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        if use_system_chrome:
            launch_kwargs["channel"] = "chrome"

        context = pw.chromium.launch_persistent_context(**launch_kwargs)
        context.add_init_script(STEALTH_INIT_JS)
        # 리턴 후 GC 되면 Chrome 이 같이 죽음 → 영상 종료될 때까지 살려둠.
        # shared_pw 는 main() 이 별도로 hold 하므로 context 만 추가.
        _KEEPALIVE.append((pw if pw_owned else None, context))

        page = context.pages[0] if context.pages else context.new_page()
        _log("🪟", display, f"render viewport @ ({win_x},{win_y}) {win_w}×{win_h}")

        # 1) navigate
        adapter.navigate(page, query)
        time.sleep(1.2)
        _log("📡", display, f"obs received · DOM settled in {time.time()-t0:.1f}s")

        # 2) CAPTCHA
        if adapter.detect_captcha(page):
            if block_on_captcha:
                _log("🛑", display, "adversarial gate (anti-bot WAF) — human-in-the-loop override ↵")
                try:
                    input(f"   [{display}] gate cleared? press Enter to resume rollout: ")
                except EOFError:
                    pass
                time.sleep(1.0)
                # Enter 신호를 신뢰 — 재감지 안 함 (오탐 방지).
                _log("🙂", display, "human signal received · resuming trajectory")
            else:
                live_status = "blocked"
                _log("🛑", display, "adversarial gate detected (async mode → mark terminated)")

        # 3) parse (CAPTCHA 안 걸렸거나, 풀고 나서)
        if live_status is None:
            _log("🔍", display, "extracting candidate states from rendered DOM")
            candidates = adapter.parse(page)[:max_results]

            if not candidates:
                live_status = "no_cands"
                _log("∅ ", display, "0 candidates · zero-reward terminal")
            else:
                _log("🧠", display, f"scoring {len(candidates)} candidates via GenRM reward model")
                _match_and_fill(res, candidates, sku_name, model, source="live")
                if write_cache:
                    saved = _save_cache(adapter, query, candidates)
                    if saved:
                        _log("📼", display, f"trajectory → replay buffer: {saved.relative_to(CACHE_DIR)}")
                tag = "🏆" if res.n_matched > 0 else "— "
                _log(tag, display,
                     f"reward = +{res.n_matched} / {len(candidates)} → {res.top_url}")

    except Exception as e:
        live_status = "error"
        res.error = f"{type(e).__name__}: {e}"
        _log("💥", display, f"env exception: {res.error[:120]}")

    # 4) 라이브 실패 → 캐시 폴백
    if live_status is not None:
        if cache_fallback:
            cached = _load_cache(adapter, query, max_results)
            if cached:
                _log("📼", display, f"replay buffer fallback · {len(cached)} trajectories")
                _match_and_fill(res, cached, sku_name, model, source="cache")
                _log("🏆", display,
                     f"reward = +{res.n_matched} / {len(cached)} (replay) → {res.top_url}")
            else:
                res.status = live_status
                _log("✗ ", display, f"env terminated ({live_status}) · no buffered trajectory")
        else:
            res.status = live_status

    res.elapsed = time.time() - t0
    # 영상용으로 창을 남겨둠 (context.close / pw.stop 호출 안 함).
    # 프로세스 종료 시 Playwright 가 자식 정리.
    return res


# ── 요약 HTML ──────────────────────────────────────
def _render_summary_html(sku_name: str, query: str, results: List[WorkerResult]) -> str:
    cards = []
    for r in results:
        if r.status in ("done", "done_cache") and r.n_matched > 0:
            badge_label, badge_color = ("MATCH ⚡" if r.status == "done_cache" else "MATCH"), "#059669"
        elif r.status in ("done", "done_cache"):
            badge_label, badge_color = "NO MATCH", "#64748b"
        elif r.status == "blocked":
            badge_label, badge_color = "BLOCKED", "#dc2626"
        elif r.status == "no_cands":
            badge_label, badge_color = "NO CANDS", "#64748b"
        else:
            badge_label, badge_color = "ERROR", "#ea580c"

        img_html = (f'<img src="{r.top_image}" />'
                    if r.top_image else '<div class="noimg"></div>')
        link_html = (f'<a href="{r.top_url}" target="_blank" rel="noopener">{r.top_url}</a>'
                     if r.top_url else '<span class="muted">—</span>')
        cards.append(f"""
        <div class="card" style="border-color:{badge_color}">
          <div class="badge" style="background:{badge_color}">{badge_label}</div>
          <div class="plat">{r.display_name}</div>
          {img_html}
          <div class="title">{r.top_title or '(결과 없음)'}</div>
          <div class="meta">후보 {r.n_candidates} · 매칭 {r.n_matched} · {r.elapsed:.1f}s</div>
          <div class="url">{link_html}</div>
        </div>
        """)

    done_count = sum(1 for r in results if r.status in ("done", "done_cache") and r.n_matched > 0)

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Fanout · {query}</title>
<style>
  *{{box-sizing:border-box}}
  body{{margin:0;padding:32px;background:#0f172a;color:#e2e8f0;
       font-family:-apple-system,'Pretendard','Apple SD Gothic Neo',sans-serif}}
  h1{{font-size:22px;margin:0 0 6px}}
  h2{{font-size:13px;color:#94a3b8;font-weight:400;margin:0 0 4px}}
  .summary{{margin:12px 0 28px;padding:14px 18px;background:#1e293b;
            border-left:4px solid #38bdf8;border-radius:6px;font-size:13px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
  .card{{background:#1e293b;border:1px solid #334155;border-radius:14px;
         padding:18px;position:relative;overflow:hidden}}
  .badge{{position:absolute;top:14px;right:14px;color:#fff;font-size:10px;
          font-weight:800;padding:4px 10px;border-radius:10px;letter-spacing:.06em}}
  .plat{{color:#38bdf8;text-transform:uppercase;font-size:12px;
         font-weight:700;letter-spacing:.08em;margin-bottom:12px}}
  .card img{{width:100%;height:160px;object-fit:cover;border-radius:8px;
             background:#0f172a;display:block}}
  .noimg{{height:160px;background:#0f172a;border-radius:8px}}
  .title{{font-size:13px;line-height:1.45;color:#f1f5f9;margin:12px 0 6px;
          max-height:55px;overflow:hidden;display:-webkit-box;
          -webkit-line-clamp:3;-webkit-box-orient:vertical}}
  .meta{{font-size:11px;color:#64748b;margin-bottom:8px}}
  .url a{{color:#38bdf8;text-decoration:none;font-size:11px;word-break:break-all}}
  .url a:hover{{text-decoration:underline}}
  .muted{{color:#475569}}
</style></head><body>
<h1>🔎 {sku_name}</h1>
<h2>검색 쿼리: {query}</h2>
<div class="summary">
  <b>{len(results)}개 사이트 병렬 검색 완료 · 매칭 {done_count}개</b>
  &nbsp;|&nbsp; 총 소요: {max((r.elapsed for r in results), default=0):.1f}s
</div>
<div class="grid">
{''.join(cards)}
</div>
</body></html>"""


def _load_preset(preset_id: str):
    fp = ROOT / "fanout_presets.json"
    if fp.exists():
        for p in json.loads(fp.read_text(encoding="utf-8"))["presets"]:
            if p["id"] == preset_id:
                return p
    qp = ROOT / "queries.json"
    if qp.exists():
        for q in json.loads(qp.read_text(encoding="utf-8"))["queries"]:
            if q["id"] == preset_id:
                return {"id": q["id"], "sku_name": q["sku_name"],
                        "query": q["query"], "platforms": None}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parallel fanout demo · 워커당 독립 브라우저 창으로 동시 검색",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sku", help="원본 SKU 타이틀 (모델 매칭 기준)")
    ap.add_argument("--query", help="검색창 입력어. 생략 시 --sku 재사용")
    ap.add_argument("--preset", help="fanout_presets.json / queries.json 의 id")
    ap.add_argument(
        "--platforms",
        default="naver,musinsa,ohouse,eleventh_street,shinsegaemall,lotteon",
        help="쉼표 구분. 지원: " + ",".join(sorted(ADAPTERS)),
    )
    ap.add_argument("--mock-model", action="store_true")
    ap.add_argument("--max-results", type=int, default=8)
    ap.add_argument("--prefer-cache", action="store_true",
                    help="캐시 hit 있으면 라이브 브라우저 띄우지 않고 바로 캐시 사용 (빠른 재생용)")
    ap.add_argument("--no-cache-fallback", action="store_true",
                    help="라이브 실패(차단/빈 결과) 시 캐시로 자동 폴백 비활성. 기본은 활성.")
    ap.add_argument("--no-write-cache", action="store_true",
                    help="라이브 성공 시 캐시 저장 안 함")
    ap.add_argument("--no-open-summary", action="store_true")
    ap.add_argument("--parallel", action="store_true",
                    help="기본은 순차 실행 (CAPTCHA 풀 여유 있음). "
                         "이 플래그 주면 N개 창을 동시에 띄우고 CAPTCHA 감지 시 즉시 BLOCKED 처리.")
    ap.add_argument("--bundled-chromium", action="store_true",
                    help="기본은 **시스템 Chrome** 사용 (Coupang 등 Akamai 사이트 차단 회피). "
                         "이 플래그 주면 Playwright 번들 Chromium 으로 강제 전환.")
    ap.add_argument("--win-w", type=int, default=640)
    ap.add_argument("--win-h", type=int, default=720)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--x-offset", type=int, default=20)
    ap.add_argument("--y-offset", type=int, default=40)
    ap.add_argument("--gap", type=int, default=8)
    args = ap.parse_args()

    platforms_override = None
    if args.preset:
        p = _load_preset(args.preset)
        if p is None:
            print(f"preset '{args.preset}' not found", file=sys.stderr); return 1
        sku = args.sku or p["sku_name"]
        query = args.query or p["query"]
        platforms_override = p.get("platforms")
    else:
        if not args.sku:
            print("--sku 또는 --preset 중 하나가 필요합니다.", file=sys.stderr); return 1
        sku = args.sku
        query = args.query or args.sku

    platforms_str = args.platforms if platforms_override is None else ",".join(platforms_override)
    platforms = [p.strip() for p in platforms_str.split(",") if p.strip()]
    unknown = [p for p in platforms if p not in ADAPTERS]
    if unknown:
        print(f"unknown platforms: {unknown}", file=sys.stderr)
        return 1

    model = MockMatchModel() if args.mock_model else MatchModel()
    model_label = "mock" if args.mock_model else getattr(model, "model_name", "model")

    use_system_chrome = not args.bundled_chromium
    ep_id = f"ep-{time.strftime('%Y%m%d-%H%M%S')}"
    print("━" * 78)
    print(f"🏋️  Nemo-Gym · ProductMatchingEnv-v2 · rollout collection")
    print(f"   episode_id    : {ep_id}")
    print(f"   anchor_sku    : {sku}")
    print(f"   query (state) : {query}")
    print(f"   action_space  : {len(platforms)} platforms → [{', '.join(platforms)}]")
    print(f"   reward_model  : GenRM-judge ({model_label})")
    print(f"   render_backend: {'Chrome (system)' if use_system_chrome else 'Chromium (Playwright bundled)'}")
    print(f"   rollout_mode  : {'async (K=' + str(len(platforms)) + ' parallel envs)' if args.parallel else 'on-policy sequential (human-gated anti-bot)'}")
    print(f"   replay_buffer : {PROFILE_ROOT}/<env>/")
    print("━" * 78)

    global _t_start
    _t_start = time.time()

    positions = [
        _grid_position(i, len(platforms), args.win_w, args.win_h,
                       cols=args.cols, x_offset=args.x_offset,
                       y_offset=args.y_offset, gap=args.gap)
        for i in range(len(platforms))
    ]

    results: List[WorkerResult] = []
    common_kwargs = dict(
        query=query, sku_name=sku, model=model,
        max_results=args.max_results,
        prefer_cache=args.prefer_cache,
        cache_fallback=not args.no_cache_fallback,
        write_cache=not args.no_write_cache,
        win_w=args.win_w, win_h=args.win_h,
        use_system_chrome=use_system_chrome,
    )

    if args.parallel:
        # 병렬: 스레드마다 자기 sync_playwright — shared_pw=None.
        with ThreadPoolExecutor(max_workers=len(platforms)) as ex:
            futs = {
                ex.submit(
                    run_platform,
                    platform=p,
                    win_x=positions[i][0], win_y=positions[i][1],
                    block_on_captcha=False,
                    shared_pw=None,
                    **common_kwargs,
                ): p
                for i, p in enumerate(platforms)
            }
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    _log("✗ ", futs[fut], f"worker crashed: {e}")
        order = {p: i for i, p in enumerate(platforms)}
        results.sort(key=lambda r: order.get(r.platform, 999))
    else:
        # 순차: **하나의 sync_playwright 인스턴스**를 공유.
        # (플랫폼마다 새로 start 하면 "sync API inside asyncio loop" 에러 발생.)
        shared_pw = sync_playwright().start()
        _KEEPALIVE.append((shared_pw, None))  # 프로세스 끝까지 hold
        try:
            for i, p in enumerate(platforms):
                _log("▶ ", p, f"step [{i+1}/{len(platforms)}] · env.step(action=search)")
                try:
                    r = run_platform(
                        platform=p,
                        win_x=positions[i][0], win_y=positions[i][1],
                        block_on_captcha=True,
                        shared_pw=shared_pw,
                        **common_kwargs,
                    )
                    results.append(r)
                except Exception as e:
                    _log("✗ ", p, f"worker crashed: {e}")
        finally:
            pass  # shared_pw 는 _KEEPALIVE 에 있으니 종료까지 유지

    total_elapsed = time.time() - _t_start
    ok = sum(1 for r in results if r.status in ("done", "done_cache") and r.n_matched > 0)
    total_cands = sum(r.n_candidates for r in results)
    total_reward = sum(r.n_matched for r in results)
    n_live = sum(1 for r in results if r.status == "done")
    n_replay = sum(1 for r in results if r.status == "done_cache")
    n_blocked = sum(1 for r in results if r.status == "blocked")
    print("━" * 78)
    print(f"📊 episode {ep_id} · terminated in {total_elapsed:.1f}s")
    print(f"   steps           : {len(results)}  "
          f"(live={n_live} · replay={n_replay} · blocked={n_blocked})")
    print(f"   cumulative reward: +{total_reward}  "
          f"(positive trajectories: {ok}/{len(results)} envs)")
    print(f"   candidates seen : {total_cands}")
    print("─" * 78)
    print(f"{'ENV':<14s} {'OUTCOME':<12s} {'CANDS':>6s} {'REWARD':>7s} {'TIME':>7s}  TOP CANDIDATE")
    print("─" * 78)
    for r in results:
        st = {"done": "✓ on-policy", "done_cache": "📼 replay", "blocked": "🛑 gated",
              "no_cands": "∅ empty", "error": "💥 error"}.get(r.status, r.status)
        top = (r.top_title[:40] + "…") if len(r.top_title) > 40 else r.top_title
        print(f"{r.display_name:<14s} {st:<10s} {r.n_candidates:>6d} {('+'+str(r.n_matched)):>7s} {r.elapsed:>6.1f}s  {top}")
    print("━" * 78)

    FANOUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() else "_" for c in query)[:40].strip("_") or "fanout"
    out = FANOUT_DIR / f"{slug}-{ts}.html"
    out.write_text(_render_summary_html(sku, query, results), encoding="utf-8")
    print(f"📝 episode artifact → {out}")
    print(f"   (trajectories written to replay buffer under {CACHE_DIR}/<env>/)")

    if not args.no_open_summary:
        try:
            import webbrowser
            webbrowser.open(f"file://{out.resolve()}")
            print("🖥  rendered trajectory summary opened in default browser.")
        except Exception as e:
            print(f"(summary open failed: {e})")

    print("\n(rollout envs kept alive for video capture. terminate: Cmd+Q or re-run this script)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
