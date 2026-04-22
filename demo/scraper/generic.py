"""Config 기반 제네릭 검색 어댑터.

사이트별 URL/셀렉터만 SiteConfig 로 선언하면 SearchAdapter 가 자동 생성된다.
파싱은 "상품 URL 패턴을 앵커로 잡고 가까운 li/article 을 카드로" 하는 동일 전략.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List
from urllib.parse import quote

from .base import Candidate, SearchAdapter

if TYPE_CHECKING:
    from playwright.sync_api import Page


@dataclass
class SiteConfig:
    name: str
    display_name: str
    base_url: str                 # href 가 상대경로일 때 앞에 붙일 origin
    home_url: str                 # 홈 URL (home-first navigate 시)
    search_url_tmpl: str          # "https://.../search?q={q}" 형태
    product_link_substrings: List[str]  # 상품 링크로 볼 URL 부분 문자열들
    use_home_navigate: bool = False     # True 면 홈→검색창→Enter
    search_box_selectors: List[str] = field(default_factory=lambda: [
        "input[placeholder*='검색']",
        "input[type='search']",
        "input[name='query']",
        "input[name='q']",
        "input[name='keyword']",
        "input[name='kwd']",
        "header input",
    ])
    # 무한스크롤 사이트(오늘의집 등)용 — 스크롤 반복 횟수
    scroll_iterations: int = 1
    # 이 수치 미만이면 디버그 덤프 강제 (파싱은 거의 실패한 것으로 간주)
    min_expected_results: int = 1
    title_selectors: List[str] = field(default_factory=lambda: [
        "[class*='productName']",
        "[class*='ProductName']",
        "[class*='product-name']",
        "[class*='goods-name']",
        "[class*='goods_name']",
        "[class*='title']",
        "[class*='Title']",
        "[class*='name']",
        "p[class*='name']",
        "em[class*='name']",
        "img[alt]",
    ])
    brand_selectors: List[str] = field(default_factory=lambda: [
        "[class*='brand']",
        "[class*='Brand']",
        "[class*='seller']",
        "[class*='mall']",
    ])
    price_selectors: List[str] = field(default_factory=lambda: [
        "[class*='price-value']",
        "[class*='priceValue']",
        "[class*='Price']",
        "[class*='price']",
        "strong",
    ])


class GenericSearchAdapter(SearchAdapter):
    """SiteConfig 하나로 동작하는 제네릭 어댑터."""

    def __init__(self, config: SiteConfig, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self.config = config
        self.name = config.name
        self.display_name = config.display_name

    def search_url(self, query: str) -> str:
        return self.config.search_url_tmpl.format(q=quote(query))

    def navigate(self, page: "Page", query: str) -> None:
        cfg = self.config
        if not cfg.use_home_navigate:
            page.goto(self.search_url(query), wait_until="domcontentloaded", timeout=30_000)
            return

        # 홈 → 검색창 → Enter
        page.goto(cfg.home_url, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(1.5)

        box = None
        for sel in cfg.search_box_selectors:
            try:
                box = page.wait_for_selector(sel, timeout=2_500, state="visible")
                if box:
                    break
            except Exception:
                continue

        if box is None:
            print(f"   ⚠  [{cfg.display_name}] 홈에서 검색창을 못 찾음 → 직링크 fallback")
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
        cfg = self.config
        # 상품 링크 앵커 셀렉터 조합
        anchor_sel = ", ".join(f"a[href*='{s}']" for s in cfg.product_link_substrings)
        try:
            page.wait_for_selector(anchor_sel, timeout=15_000)
        except Exception:
            self._dump_debug(page)
            return []

        # 레이지 로드 유도 — 무한스크롤 사이트는 여러 번 반복
        try:
            for _ in range(max(1, cfg.scroll_iterations)):
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                time.sleep(0.8)
            # 맨 위로 원복 (디스플레이용)
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
                seen.add(href)

                card = a.evaluate_handle(
                    "el => el.closest('li') || el.closest('article') || el.closest('div[class*=card]')"
                    " || el.closest('div[class*=item]') || el.parentElement"
                ).as_element()
                if card is None:
                    card = a

                title = self._extract_text(card, cfg.title_selectors, is_img_alt_ok=True)
                if not title:
                    # fallback: 카드 텍스트 첫 알파 포함 줄
                    try:
                        txt = (card.inner_text() or "").strip()
                        for ln in txt.splitlines():
                            ln = ln.strip()
                            if ln and any(ch.isalpha() for ch in ln) and "원" not in ln[-2:]:
                                title = ln[:120]
                                break
                    except Exception:
                        pass
                if not title:
                    continue

                brand = self._extract_text(card, cfg.brand_selectors)
                price = self._extract_text(card, cfg.price_selectors)

                img_src = ""
                img_el = card.query_selector("img")
                if img_el:
                    img_src = (
                        img_el.get_attribute("src")
                        or img_el.get_attribute("data-src")
                        or img_el.get_attribute("data-original")
                        or img_el.get_attribute("data-lazy")
                        or ""
                    )
                    if img_src.startswith("//"):
                        img_src = "https:" + img_src

                product_url = (cfg.base_url + href) if href.startswith("/") else href

                results.append(Candidate(
                    title=title,
                    brand=brand,
                    price=price,
                    product_url=product_url,
                    image_url=img_src,
                ))
            except Exception:
                continue

        if len(results) < cfg.min_expected_results:
            self._dump_debug(page)
        return results

    # ── helpers ──────────────────────────────────
    @staticmethod
    def _extract_text(card, selectors: List[str], is_img_alt_ok: bool = False) -> str:
        for sel in selectors:
            try:
                el = card.query_selector(sel)
            except Exception:
                el = None
            if el is None:
                continue
            if sel == "img[alt]":
                if not is_img_alt_ok:
                    continue
                val = (el.get_attribute("alt") or "").strip()
            else:
                val = (el.inner_text() or "").strip().replace("\n", " ")
            if val:
                return val.splitlines()[0][:120] if "\n" not in val else val[:120]
        return ""

    def _dump_debug(self, page: "Page") -> None:
        try:
            if self.cache_dir is None:
                return
            debug_dir = self.cache_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            page.screenshot(path=str(debug_dir / f"{self.name}-{ts}.png"), full_page=True)
            (debug_dir / f"{self.name}-{ts}.html").write_text(page.content(), encoding="utf-8")
            print(f"   ⓘ 디버그 덤프 저장: {debug_dir}/{self.name}-{ts}.(png|html)")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# 플랫폼별 설정 (Tier 1 + Tier 2 = 10개)
# URL/앵커 패턴은 공개 검색 URL 및 상품 URL 구조 기반 best-guess.
# 최초 실행 시 0 candidates 나오면 cache/debug/<name>-*.(png|html) 확인 후 갱신.
# ─────────────────────────────────────────────────────────────────────

SITE_CONFIGS = {
    # ── Tier 1 ──────────────────────────────────────
    "ohouse": SiteConfig(
        name="ohouse",
        display_name="오늘의집",
        base_url="https://ohou.se",
        home_url="https://ohou.se/",
        search_url_tmpl="https://ohou.se/commerces/search?query={q}",
        # 실제 상품 URL: store.ohou.se/goods/<id>
        product_link_substrings=["store.ohou.se/goods/", "/goods/"],
        use_home_navigate=True,
        scroll_iterations=6,
        min_expected_results=3,
    ),
    "eleventh_street": SiteConfig(
        name="eleventh_street",
        display_name="11번가",
        base_url="https://www.11st.co.kr",
        home_url="https://www.11st.co.kr/",
        search_url_tmpl="https://search.11st.co.kr/Search.tmall?kwd={q}",
        product_link_substrings=["/products/", "prd.11st.co.kr"],
        use_home_navigate=False,
        title_selectors=[
            "[class*='c-card-item__name']",  # 87개 카드 모두 여기에
            "img[alt]",
        ],
        brand_selectors=["[class*='c-card-item__brand-name']", "[class*='c-seller__name']"],
        price_selectors=["[class*='c-card-item__price-info']", "[class*='c-card-item__price']"],
    ),
    "gmarket": SiteConfig(
        name="gmarket",
        display_name="지마켓",
        base_url="https://www.gmarket.co.kr",
        home_url="https://www.gmarket.co.kr/",
        search_url_tmpl="https://browse.gmarket.co.kr/search?keyword={q}",
        product_link_substrings=["item.gmarket.co.kr", "goodscode=", "goodsCode="],
        use_home_navigate=True,
        title_selectors=[
            "[class*='text__item-title']",
            "[class*='box__item-title']",
            "img[alt]",
        ],
        brand_selectors=["[class*='box__brand']", "[class*='seller']"],
        price_selectors=["[class*='box__item-price']", "[class*='text__price']"],
    ),
    "lotteon": SiteConfig(
        name="lotteon",
        display_name="롯데온",
        base_url="https://www.lotteon.com",
        home_url="https://www.lotteon.com/",
        search_url_tmpl="https://www.lotteon.com/search/search/search.ecn?render=search&q={q}",
        product_link_substrings=["/p/product/", "/p/goods/"],
        use_home_navigate=True,
    ),

    # ── Tier 2 ──────────────────────────────────────
    "emartinternetshopping": SiteConfig(
        name="emartinternetshopping",
        display_name="이마트몰",
        base_url="https://emart.ssg.com",
        home_url="https://emart.ssg.com/",
        search_url_tmpl="https://emart.ssg.com/search.ssg?target=all&query={q}",
        product_link_substrings=["itemView.ssg", "itemId=", "/item/"],
        use_home_navigate=False,
        # Chakra UI 해시 클래스라 구조 기반 셀렉터 — em = 브랜드, del/strong = 가격
        # (auction/emart 는 scrape 시점 selector 매칭 실패로 brand/price 비는 경우 있음 — title 만으로 데모 가능)
        brand_selectors=["a[href*='itemView'] em", "em"],
        price_selectors=["del", "strong", "em[class*='price']"],
    ),
    "auction": SiteConfig(
        name="auction",
        display_name="옥션",
        base_url="https://www.auction.co.kr",
        home_url="https://www.auction.co.kr/",
        search_url_tmpl="https://browse.auction.co.kr/search?keyword={q}",
        product_link_substrings=["itemno=", "itemNo=", "/item/"],
        use_home_navigate=True,
        brand_selectors=["[class*='text__brand']", "[class*='box__brand']"],
        # text__price-seller 안에 실제 가격 숫자, box__price-sale 는 할인율만
        price_selectors=["[class*='text__price-seller']", "[class*='box__price-sale']"],
    ),
    "hmall": SiteConfig(
        name="hmall",
        display_name="현대Hmall",
        base_url="https://www.hmall.com",
        home_url="https://www.hmall.com/",
        search_url_tmpl="https://www.hmall.com/pd/pda/search.do?search_word={q}",
        product_link_substrings=["slitmCd=", "/p/", "/pd/pda/itemPtc"],
        use_home_navigate=True,
    ),
    "gsshop": SiteConfig(
        name="gsshop",
        display_name="GS샵",
        base_url="http://www.gsshop.com",
        home_url="http://www.gsshop.com/",
        search_url_tmpl="http://www.gsshop.com/search/search.gs?tq={q}",
        product_link_substrings=["prdid=", "prdId=", "/prd/"],
        use_home_navigate=True,
    ),
    "shinsegaemall": SiteConfig(
        name="shinsegaemall",
        display_name="신세계몰",
        # 신세계몰은 SSG.COM 통합 — 동일 검색엔진(target=all)
        base_url="https://www.ssg.com",
        home_url="https://www.ssg.com/",
        search_url_tmpl="https://www.ssg.com/search.ssg?target=all&query={q}",
        product_link_substrings=["itemView.ssg", "itemId=", "/item/"],
        use_home_navigate=True,
        # ssg/emart 공통: Chakra UI 해시 클래스 → 구조 기반
        brand_selectors=["a[href*='itemView'] em", "em"],
        price_selectors=["del", "strong", "em[class*='price']"],
    ),
    # cjonstyle 은 검색결과 페이지(display.cjonstyle.com) 가 수동 접속 시에도
    # "주소가 잘못되었거나 삭제되었다" 로 뜨는 상태 — 시연에서 제외.
}


def make_generic_adapter_class(key: str):
    """ADAPTERS 등록을 위해 plat_key → 인자 없는 클래스(wrapper) 생성."""
    cfg = SITE_CONFIGS[key]

    class _Bound(GenericSearchAdapter):
        def __init__(self, cache_dir=None):
            super().__init__(cfg, cache_dir=cache_dir)

    _Bound.__name__ = f"{key.title().replace('_','')}Adapter"
    return _Bound
