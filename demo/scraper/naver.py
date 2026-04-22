"""네이버쇼핑 검색 어댑터.

Naver는 React SPA + CSS Modules라 셀렉터가 해시되어 있어 깨지기 쉽다.
1차: `div[class*='product_item']` 같은 attribute-contains 셀렉터
2차 fallback: `<script id="__NEXT_DATA__">` JSON 파싱
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, List
from urllib.parse import quote

from .base import Candidate, SearchAdapter

if TYPE_CHECKING:
    from playwright.sync_api import Page


class NaverShoppingAdapter(SearchAdapter):
    name = "naver"
    display_name = "네이버쇼핑"

    HOME_URL = "https://shopping.naver.com/ns/home"

    def search_url(self, query: str) -> str:
        return f"https://search.shopping.naver.com/search/all?query={quote(query)}"

    def navigate(self, page: "Page", query: str) -> None:
        """홈 → 검색창 입력 → Enter. 검색창을 못 찾으면 직링크로 fallback."""
        page.goto(self.HOME_URL, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(1.5)

        selectors = [
            "input[placeholder*='검색']",
            "input[type='search']",
            "input[name='query']",
            "header input",
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
            print(f"   ⚠  네이버쇼핑 홈에서 검색창을 못 찾음 → 직링크로 fallback")
            page.goto(self.search_url(query), wait_until="domcontentloaded", timeout=30_000)
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
        try:
            page.wait_for_selector(
                "div[class*='product_item'], div[class*='basicList_item']",
                timeout=15_000,
            )
        except Exception:
            # DOM 파싱 실패 시 __NEXT_DATA__ JSON fallback
            return self._parse_next_data(page)

        selector = "div[class*='product_item'], div[class*='basicList_item']"
        cards = page.query_selector_all(selector)
        results: List[Candidate] = []
        for c in cards:
            try:
                title_el = (
                    c.query_selector("a[class*='product_link']")
                    or c.query_selector("a[class*='basicList_link']")
                    or c.query_selector("a")
                )
                price_el = c.query_selector("span[class*='price_num']") or c.query_selector(
                    "span[class*='price']"
                )
                img_el = c.query_selector("img")

                title = (title_el.inner_text().strip() if title_el else "").replace("\n", " ")
                if not title:
                    continue

                results.append(Candidate(
                    title=title,
                    price=(price_el.inner_text().strip() if price_el else ""),
                    product_url=(title_el.get_attribute("href") or "") if title_el else "",
                    image_url=(img_el.get_attribute("src") or "") if img_el else "",
                ))
            except Exception:
                continue

        if not results:
            results = self._parse_next_data(page)
        return results

    def _parse_next_data(self, page: "Page") -> List[Candidate]:
        try:
            script = page.query_selector("script#__NEXT_DATA__")
            if not script:
                return []
            data = json.loads(script.inner_text())
        except Exception:
            return []

        # Next.js 구조: pageProps → initialState → productList.list 등 (버전별 차이)
        candidates: List[Candidate] = []

        def walk(obj):
            if isinstance(obj, dict):
                # 제품 엔트리 휴리스틱: productTitle/productName + lowPrice 가 같이 있으면 후보
                title = obj.get("productTitle") or obj.get("productName") or obj.get("title")
                price = obj.get("lowPrice") or obj.get("price")
                if title and isinstance(title, str) and price:
                    candidates.append(Candidate(
                        title=re.sub(r"<[^>]+>", "", title),
                        price=str(price),
                        product_url=obj.get("crUrl") or obj.get("mallProductUrl") or "",
                        image_url=obj.get("imageUrl") or "",
                    ))
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(data)
        # dedupe by title
        seen = set()
        out = []
        for c in candidates:
            if c.title in seen:
                continue
            seen.add(c.title)
            out.append(c)
        return out
