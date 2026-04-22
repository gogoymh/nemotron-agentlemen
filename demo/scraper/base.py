"""SearchAdapter 추상 + Candidate 데이터클래스 + CAPTCHA 감지/대기."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from playwright.sync_api import Page


@dataclass
class Candidate:
    """검색 결과 카드 1건."""
    title: str
    brand: str = ""
    price: str = ""
    image_url: str = ""
    product_url: str = ""
    options: List[str] = field(default_factory=list)


CAPTCHA_MARKERS = [
    # 공통
    "captcha", "recaptcha", "h-captcha",
    # Cloudflare / WAF
    "cf-challenge", "challenge-running", "cloudflare",
    # Akamai (쿠팡 등)
    "access denied", "errors.edgesuite.net", "reference #",
    # 한글
    "보안문자", "자동 등록 방지", "로봇이 아닙니다",
    "일시적인 접근 차단", "이용이 제한",
]


class SearchAdapter(ABC):
    """사이트별 검색 어댑터 베이스."""

    name: str = "base"
    display_name: str = "Base"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    # ── 하위 클래스에서 구현 ─────────────────────────
    @abstractmethod
    def search_url(self, query: str) -> str:
        """검색 쿼리 → URL."""

    @abstractmethod
    def parse(self, page: "Page") -> List[Candidate]:
        """현재 페이지에서 결과 카드 파싱."""

    def navigate(self, page: "Page", query: str) -> None:
        """기본은 search URL 직링크. 사람처럼 포털→검색창을 타고 싶으면 오버라이드."""
        page.goto(self.search_url(query), wait_until="domcontentloaded", timeout=30_000)

    # ── 공통 로직 ──────────────────────────────────
    def detect_captcha(self, page: "Page") -> bool:
        """CAPTCHA / 봇 차단 페이지 감지."""
        try:
            url = (page.url or "").lower()
            html = (page.content() or "").lower()
        except Exception:
            return False

        for marker in CAPTCHA_MARKERS:
            if marker in url or marker in html:
                return True
        # title 단서도 확인
        try:
            title = (page.title() or "").lower()
            for marker in ("blocked", "차단", "접근 제한", "robot"):
                if marker in title:
                    return True
        except Exception:
            pass
        return False

    def wait_for_manual_captcha(self, page: "Page", timeout_s: int = 300) -> bool:
        """CAPTCHA 풀릴 때까지 사용자 입력 대기. Enter 누르면 사용자 신호 신뢰."""
        print(f"\n🔒 [{self.display_name}] CAPTCHA/차단 감지.")
        print(f"   브라우저 창에서 직접 풀고, 이 셀의 입력창에 Enter를 눌러주세요.")
        print(f"   (최대 {timeout_s}초 대기)")
        try:
            input("   ⏸  풀렸으면 Enter: ")
        except EOFError:
            # non-interactive 환경 (배치 실행) 에서는 단순 대기
            time.sleep(min(timeout_s, 30))
        time.sleep(1.0)
        # 사용자 신호를 신뢰하고 바로 진행 — 오탐으로 루프 도는 걸 막기 위해.
        # 실제로 못 풀었으면 이후 parser가 실패해서 0 candidates 반환.
        return True

    def search(
        self,
        page: "Page",
        query: str,
        *,
        use_cache: bool = True,
        write_cache: bool = True,
        wait_after_load_s: float = 1.5,
        max_results: int = 20,
    ) -> List[Candidate]:
        """메인 진입점: 캐시 → 라이브 검색 → 파싱 → 캐시 저장."""
        cache_file = self._cache_path(query)

        # 1) cache hit
        if use_cache and cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [Candidate(**c) for c in data][:max_results]

        # 2) live
        self.navigate(page, query)
        time.sleep(wait_after_load_s)

        # CAPTCHA 감지 → 사용자가 Enter 치면 신뢰하고 진행
        if self.detect_captcha(page):
            self.wait_for_manual_captcha(page)

        candidates = self.parse(page)[:max_results]

        # 3) cache write
        if write_cache and cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return candidates

    # ── 내부 ───────────────────────────────────────
    def _cache_path(self, query: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe = re.sub(r"[^0-9A-Za-z가-힣]+", "_", query).strip("_")[:80] or "query"
        return self.cache_dir / self.name / f"{safe}.json"
