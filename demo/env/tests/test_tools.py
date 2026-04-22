"""Tool-layer tests. No network, no Playwright — uses the on-disk cache only."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from demo.env.pool import PlatformPool
from demo.env.schema import (
    InspectRequest,
    SearchRequest,
    SubmitMatchRequest,
)
from demo.env.tools import exec_inspect, exec_search, exec_submit_match


@pytest.fixture
def tmp_cache():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Seed a naver cache entry mirroring adapter.name == "naver".
        (root / "naver").mkdir()
        (root / "naver" / "test_query.json").write_text(
            json.dumps([
                {"title": "샘플 상품 A", "brand": "BrandA", "price": "9900",
                 "image_url": "http://img/a.jpg", "product_url": "http://n/a",
                 "options": []},
                {"title": "샘플 상품 B", "brand": "BrandB", "price": "12000",
                 "image_url": "", "product_url": "http://n/b", "options": []},
            ], ensure_ascii=False),
            encoding="utf-8",
        )
        yield root


@pytest.mark.asyncio
async def test_search_cache_hit(tmp_cache):
    pool = PlatformPool(cache_dir=tmp_cache, live_enabled=False,
                        allowed_platforms=["naver"])
    resp = await exec_search(pool, SearchRequest(platform="naver", query="test query"))
    assert resp.status == "ok"
    assert resp.source == "cache"
    assert len(resp.candidates) == 2
    assert resp.candidates[0].title == "샘플 상품 A"


@pytest.mark.asyncio
async def test_search_cache_miss_without_live(tmp_cache):
    pool = PlatformPool(cache_dir=tmp_cache, live_enabled=False,
                        allowed_platforms=["naver"])
    resp = await exec_search(pool, SearchRequest(platform="naver", query="nonexistent"))
    assert resp.status == "no_cache"
    assert resp.candidates == []


@pytest.mark.asyncio
async def test_search_platform_not_in_allowlist(tmp_cache):
    pool = PlatformPool(cache_dir=tmp_cache, live_enabled=False,
                        allowed_platforms=["naver"])
    resp = await exec_search(pool, SearchRequest(platform="coupang", query="x"))
    assert resp.status == "error"
    assert "allow-list" in resp.error


@pytest.mark.asyncio
async def test_search_unknown_platform(tmp_cache):
    pool = PlatformPool(cache_dir=tmp_cache, live_enabled=False)
    resp = await exec_search(pool, SearchRequest(platform="nowhere", query="x"))
    assert resp.status == "error"


@pytest.mark.asyncio
async def test_inspect_stub_echoes():
    resp = await exec_inspect(InspectRequest(url="https://example.com/p/1"))
    assert resp.status == "not_implemented"
    assert resp.url == "https://example.com/p/1"


@pytest.mark.asyncio
async def test_inspect_empty_url_errors():
    resp = await exec_inspect(InspectRequest(url=""))
    assert resp.status == "error"


@pytest.mark.asyncio
async def test_submit_match_records_ok():
    req = SubmitMatchRequest(anchor_id="pm-x", candidate_url="http://n/a",
                             decision="matched", title="t", platform="naver")
    resp = await exec_submit_match(req, anchor_id_expected="pm-x")
    assert resp.status == "recorded"


@pytest.mark.asyncio
async def test_submit_match_rejects_anchor_mismatch():
    req = SubmitMatchRequest(anchor_id="pm-other", candidate_url="u", decision="matched")
    resp = await exec_submit_match(req, anchor_id_expected="pm-x")
    assert resp.status == "rejected"
    assert "anchor_id mismatch" in resp.reason


@pytest.mark.asyncio
async def test_submit_match_rejects_bad_decision():
    req = SubmitMatchRequest(anchor_id="pm-x", candidate_url="u", decision="unsure")
    resp = await exec_submit_match(req, anchor_id_expected="pm-x")
    assert resp.status == "rejected"
    assert "matched|not_matched" in resp.reason
