"""쿠팡 검색 어댑터."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List
from urllib.parse import quote

from .base import Candidate, SearchAdapter

if TYPE_CHECKING:
    from playwright.sync_api import Page


class CoupangAdapter(SearchAdapter):
    name = "coupang"
    display_name = "쿠팡"

    def search_url(self, query: str) -> str:
        return f"https://www.coupang.com/np/search?q={quote(query)}&channel=user"

    def navigate(self, page: "Page", query: str) -> None:
        """홈 → 검색창 입력 → Enter. /np/search 직링크는 Akamai에 자주 막힘."""
        page.goto("https://www.coupang.com/", wait_until="domcontentloaded", timeout=30_000)
        time.sleep(1.5)

        # 검색창 셀렉터가 바뀔 수 있어 여러 후보 중 먼저 뜨는 걸 잡음
        selectors = [
            "input#headerSearchKeyword",
            "input[name='q']",
            "input[placeholder*='검색']",
            "header input[type='search']",
        ]
        box = None
        for sel in selectors:
            try:
                box = page.wait_for_selector(sel, timeout=3_000, state="visible")
                if box:
                    break
            except Exception:
                continue

        if box is None:
            # 홈 자체가 차단됐을 가능성 높음 — 상위 CAPTCHA 루프가 감지하도록 그냥 리턴
            print(f"   ⚠  홈페이지에서 검색창을 찾지 못함. 현재 URL: {page.url}")
            return

        box.click()
        try:
            box.fill("")
        except Exception:
            pass
        page.keyboard.type(query, delay=40)
        time.sleep(0.3)
        page.keyboard.press("Enter")
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15_000)
        except Exception:
            pass

    def parse(self, page: "Page") -> List[Candidate]:
        """쿠팡 검색결과 파싱.

        쿠팡은 DOM 클래스가 자주 바뀌므로(해시 클래스명) /vp/products/ 링크를
        기준점(anchor)으로 삼아 상위 li/div를 역추적한다.
        """
        # 결과가 렌더될 때까지 대기 — 상품 링크 기준
        anchor_sel = "a[href*='/vp/products/']"
        try:
            page.wait_for_selector(anchor_sel, timeout=15_000)
        except Exception:
            self._dump_debug(page)
            return []

        # 레이지 로드 유도: 페이지 끝까지 스크롤 한 번
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.0)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.3)
        except Exception:
            pass

        # 링크를 기준으로 카드(가까운 li) 추출 — 중복 제거
        anchors = page.query_selector_all(anchor_sel)
        seen = set()
        results: List[Candidate] = []
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)

                # 카드 컨테이너: 가장 가까운 li, 없으면 가까운 article/div
                card = a.evaluate_handle(
                    "el => el.closest('li') || el.closest('article') || el.closest('div[class*=card]') || el.parentElement"
                ).as_element()
                if card is None:
                    card = a

                # 타이틀 후보들
                title = ""
                for tsel in [
                    "[class*='productName']",
                    "[class*='ProductName']",
                    "[class*='name']",
                    ".name",
                    "img[alt]",
                ]:
                    el = card.query_selector(tsel)
                    if el is None:
                        continue
                    if tsel == "img[alt]":
                        title = (el.get_attribute("alt") or "").strip()
                    else:
                        title = (el.inner_text() or "").strip().replace("\n", " ")
                    if title:
                        break
                if not title:
                    # 최후: 카드 전체 텍스트 앞부분
                    try:
                        title = (card.inner_text() or "").strip().splitlines()[0][:120]
                    except Exception:
                        pass
                if not title:
                    continue

                # 가격
                price = ""
                for psel in [
                    "[class*='price-value']",
                    "[class*='priceValue']",
                    "[class*='Price']",
                    "strong",
                ]:
                    el = card.query_selector(psel)
                    if el:
                        txt = (el.inner_text() or "").strip()
                        if txt:
                            price = txt.splitlines()[0][:40]
                            break

                # 이미지
                img_src = ""
                img_el = card.query_selector("img")
                if img_el:
                    img_src = (
                        img_el.get_attribute("src")
                        or img_el.get_attribute("data-img-src")
                        or img_el.get_attribute("data-src")
                        or ""
                    )
                    if img_src.startswith("//"):
                        img_src = "https:" + img_src

                product_url = ("https://www.coupang.com" + href) if href.startswith("/") else href

                results.append(Candidate(
                    title=title,
                    price=price,
                    product_url=product_url,
                    image_url=img_src,
                ))
            except Exception:
                continue

        if not results:
            self._dump_debug(page)
        return results

    def _dump_debug(self, page: "Page") -> None:
        """파싱 실패 시 스크린샷+HTML을 cache/debug 에 남긴다."""
        try:
            if self.cache_dir is None:
                return
            debug_dir = self.cache_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            page.screenshot(path=str(debug_dir / f"coupang-{ts}.png"), full_page=True)
            (debug_dir / f"coupang-{ts}.html").write_text(page.content(), encoding="utf-8")
            print(f"   ⓘ 디버그 덤프 저장: {debug_dir}/coupang-{ts}.(png|html)")
        except Exception:
            pass
