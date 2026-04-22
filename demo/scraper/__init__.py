"""커머스 사이트 검색 어댑터.

각 어댑터는 동일한 인터페이스(`search(page, query) -> list[Candidate]`)를 제공한다.
Playwright Page 객체를 받아 검색 URL로 이동 → 결과 파싱.

CAPTCHA 감지 시 `input()`으로 대기하여 사용자가 브라우저에서 직접 풀도록 한다.
"""

from .base import Candidate, SearchAdapter
from .coupang import CoupangAdapter
from .naver import NaverShoppingAdapter
from .musinsa import MusinsaAdapter
from .generic import SITE_CONFIGS, GenericSearchAdapter, make_generic_adapter_class

# 커스텀 어댑터 (특수 로직)
ADAPTERS = {
    "coupang": CoupangAdapter,
    "naver": NaverShoppingAdapter,
    "musinsa": MusinsaAdapter,
}

# Tier 1 + Tier 2 제네릭 어댑터 자동 등록
for _key in SITE_CONFIGS:
    ADAPTERS[_key] = make_generic_adapter_class(_key)

__all__ = [
    "Candidate", "SearchAdapter", "ADAPTERS", "SITE_CONFIGS",
    "CoupangAdapter", "NaverShoppingAdapter", "MusinsaAdapter",
    "GenericSearchAdapter",
]
