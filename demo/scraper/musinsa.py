"""무신사 검색 어댑터."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List
from urllib.parse import quote

from .base import Candidate, SearchAdapter

if TYPE_CHECKING:
    from playwright.sync_api import Page


class MusinsaAdapter(SearchAdapter):
    name = "musinsa"
    display_name = "무신사"

    def search_url(self, query: str) -> str:
        return f"https://www.musinsa.com/search/goods?keyword={quote(query)}"

    # navigate는 기본(직링크) 사용 — 무신사는 봇 차단 약해서 문제 없음

    def parse(self, page: "Page") -> List[Candidate]:
        """상품 링크(`/products/` 등)를 앵커로 카드 역추적 — 클래스 해시 변동에 강건."""
        anchor_sel = (
            "a[href*='/products/'], "
            "a[href*='/goods/'], "
            "a[href*='/app/goods/']"
        )
        try:
            page.wait_for_selector(anchor_sel, timeout=15_000)
        except Exception:
            self._dump_debug(page)
            return []

        # 레이지 로드 유도
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.0)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.3)
        except Exception:
            pass

        anchors = page.query_selector_all(anchor_sel)
        seen = set()
        results: List[Candidate] = []
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                # 카테고리/태그 링크는 스킵 — 숫자 ID 포함된 것만
                if not any(seg in href for seg in ("/products/", "/goods/")):
                    continue
                seen.add(href)

                card = a.evaluate_handle(
                    "el => el.closest('li') || el.closest('article') || el.closest('div[class*=card]') || el.parentElement"
                ).as_element()
                if card is None:
                    card = a

                # 타이틀
                title = ""
                for tsel in [
                    "[class*='ProductName']",
                    "[class*='product-name']",
                    "p[class*='name']",
                    "[class*='goods_name']",
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
                    try:
                        txt = (card.inner_text() or "").strip()
                        lines = [ln for ln in txt.splitlines() if ln.strip()]
                        # 가격(숫자/원) 아닌 줄을 타이틀로
                        for ln in lines:
                            if any(ch.isalpha() for ch in ln) and "원" not in ln[-3:]:
                                title = ln[:120]
                                break
                    except Exception:
                        pass
                if not title:
                    continue

                # 브랜드
                brand = ""
                for bsel in ["[class*='Brand']", "[class*='brand']"]:
                    el = card.query_selector(bsel)
                    if el:
                        brand = (el.inner_text() or "").strip().splitlines()[0][:40]
                        if brand:
                            break

                # 가격
                price = ""
                for psel in ["[class*='Price']", "[class*='price']", "strong"]:
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
                        or img_el.get_attribute("data-src")
                        or img_el.get_attribute("data-original")
                        or ""
                    )
                    if img_src.startswith("//"):
                        img_src = "https:" + img_src

                product_url = ("https://www.musinsa.com" + href) if href.startswith("/") else href

                results.append(Candidate(
                    title=title,
                    brand=brand,
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
        try:
            if self.cache_dir is None:
                return
            debug_dir = self.cache_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            page.screenshot(path=str(debug_dir / f"musinsa-{ts}.png"), full_page=True)
            (debug_dir / f"musinsa-{ts}.html").write_text(page.content(), encoding="utf-8")
            print(f"   ⓘ 디버그 덤프 저장: {debug_dir}/musinsa-{ts}.(png|html)")
        except Exception:
            pass
