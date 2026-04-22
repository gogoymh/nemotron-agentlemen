"""Platform search pool — cache-first, with optional Playwright live fallback.

Design:
  - sync scrapers from demo/scraper/ are the source of truth (already handle
    cache read/write, CAPTCHA detection, DOM parsing).
  - Live calls launch a fresh sync_playwright() per search in a worker thread,
    serialized per-platform to avoid concurrent Chrome launches.
  - cache-only mode (config.env.live_scraping=False) never touches Playwright
    → safe to run on headless training boxes without Chrome installed.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from .schema import CandidateOut

# demo/scraper is a sibling package — import requires demo/ on sys.path.
_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from scraper import ADAPTERS  # noqa: E402
from scraper.base import Candidate  # noqa: E402


class PlatformPool:
    """Thread-safe synchronous search dispatcher.

    Always tries the on-disk cache first. Falls back to Playwright live search
    only when `live_enabled=True`. One lock per platform guards live launches.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        live_enabled: bool = False,
        use_system_chrome: bool = True,
        allowed_platforms: Optional[List[str]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.live_enabled = live_enabled
        self.use_system_chrome = use_system_chrome
        self.allowed = set(allowed_platforms) if allowed_platforms else set(ADAPTERS)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    # ── Public sync API (callers wrap in asyncio.to_thread) ───────────
    def search(
        self, platform: str, query: str, max_results: int = 8
    ) -> Tuple[str, str, List[CandidateOut], str]:
        """Returns (status, source, candidates, error)."""
        if platform not in ADAPTERS:
            return "error", "none", [], f"unknown platform: {platform}"
        if platform not in self.allowed:
            return "error", "none", [], f"platform not in allow-list: {platform}"

        adapter = ADAPTERS[platform](cache_dir=self.cache_dir)
        cached = self._load_cache(adapter, query, max_results)
        if cached is not None:
            return "ok", "cache", _to_out(cached), ""

        if not self.live_enabled:
            return "no_cache", "none", [], ""

        with self._locks[platform]:
            return self._live_search(adapter, query, max_results)

    # ── Cache I/O (reuses adapter's _cache_path convention) ───────────
    @staticmethod
    def _load_cache(adapter, query: str, max_results: int) -> Optional[List[Candidate]]:
        p = adapter._cache_path(query)
        if not p or not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return [Candidate(**c) for c in data][:max_results]
        except Exception:
            return None

    @staticmethod
    def _save_cache(adapter, query: str, candidates: List[Candidate]) -> None:
        p = adapter._cache_path(query)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Live (Playwright) path ────────────────────────────────────────
    def _live_search(
        self, adapter, query: str, max_results: int
    ) -> Tuple[str, str, List[CandidateOut], str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            return "error", "none", [], f"playwright not installed: {e}"

        profile = Path.home() / ".cache" / "nemotron-fanout" / adapter.name
        profile.mkdir(parents=True, exist_ok=True)
        launch_kwargs = dict(
            user_data_dir=str(profile),
            headless=True,           # server mode — no visible windows
            locale="ko-KR",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        if self.use_system_chrome:
            launch_kwargs["channel"] = "chrome"

        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(**launch_kwargs)
                try:
                    page = ctx.pages[0] if ctx.pages else ctx.new_page()
                    adapter.navigate(page, query)
                    time.sleep(1.0)
                    if adapter.detect_captcha(page):
                        return "blocked", "live", [], "captcha/WAF"
                    cands = adapter.parse(page)[:max_results]
                    if cands:
                        self._save_cache(adapter, query, cands)
                        return "ok", "live", _to_out(cands), ""
                    return "ok", "live", [], ""
                finally:
                    try:
                        ctx.close()
                    except Exception:
                        pass
        except Exception as e:
            return "error", "live", [], f"{type(e).__name__}: {e}"


def _to_out(cands: List[Candidate]) -> List[CandidateOut]:
    return [
        CandidateOut(
            title=c.title, product_url=c.product_url,
            image_url=c.image_url, price=c.price, brand=c.brand,
        )
        for c in cands
    ]


def slugify_query(query: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "_", query).strip("_")[:80] or "query"
