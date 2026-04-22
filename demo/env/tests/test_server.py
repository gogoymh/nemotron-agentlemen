"""End-to-end server smoke test.

Uses mock judge (jaccard) and cache-only pool so nothing external is needed.
A fake policy client is injected into Agent to drive a deterministic
trajectory (search → submit_match → submit_match → stop).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from demo.env.config import (
    EnvConfig,
    EnvRootConfig,
    JudgeConfig,
    PolicyConfig,
    RewardConfig,
    EpisodeConfig,
)
from demo.env.server import build_app


@pytest.fixture
def cfg_and_cache():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        (cache / "naver").mkdir()
        (cache / "naver" / "베개커버.json").write_text(
            json.dumps([
                {"title": "데시뉴 에센스 항균 누빔 베개커버 50x70", "brand": "데시뉴",
                 "price": "9900", "image_url": "", "product_url": "http://n/a",
                 "options": []},
                {"title": "오늘의집 PICK 베개커버 40x60", "brand": "",
                 "price": "5500", "image_url": "", "product_url": "http://n/b",
                 "options": []},
            ], ensure_ascii=False),
            encoding="utf-8",
        )
        cfg = EnvRootConfig(
            env=EnvConfig(cache_dir=str(cache), live_scraping=False, platforms=["naver"]),
            judge=JudgeConfig(mock=True),
            policy=PolicyConfig(),
            reward=RewardConfig(),
            episode=EpisodeConfig(),
        )
        yield cfg, cache


def test_health_reports_mock_judge(cfg_and_cache):
    cfg, _ = cfg_and_cache
    client = TestClient(build_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["judge_label"] == "mock"
    assert body["platforms"] == ["naver"]


def test_tool_search_cache_hit(cfg_and_cache):
    cfg, _ = cfg_and_cache
    client = TestClient(build_app(cfg))
    r = client.post("/tool/search", json={"platform": "naver", "query": "베개커버"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["source"] == "cache"
    assert len(body["candidates"]) == 2


def test_verify_directly_scores_submissions(cfg_and_cache):
    cfg, _ = cfg_and_cache
    client = TestClient(build_app(cfg))
    # Build a Responses-API output log manually: one correct match, one wrong match.
    output = [
        {"type": "function_call", "name": "search",
         "arguments": json.dumps({"platform": "naver", "query": "베개커버"})},
        {"type": "function_call", "name": "submit_match",
         "arguments": json.dumps({
             "anchor_id": "pm-pillow", "candidate_url": "http://n/a",
             "decision": "matched",
             "title": "데시뉴 에센스 항균 누빔 베개커버 50x70",
             "platform": "naver"})},
        {"type": "function_call", "name": "submit_match",
         "arguments": json.dumps({
             "anchor_id": "pm-pillow", "candidate_url": "http://n/b",
             "decision": "matched",
             "title": "오늘의집 PICK 베개커버 40x60",
             "platform": "naver"})},
    ]
    body = {
        "response": {"output": output, "output_text": ""},
        "verifier_metadata": {
            "anchor_id": "pm-pillow",
            "anchor_sku": "데시뉴 에센스 항균 누빔 베개커버 50x70",
            "platforms": ["naver"],
            "budget_search": 6, "budget_submit": 12,
        },
    }
    r = client.post("/verify", json=body)
    assert r.status_code == 200
    v = r.json()
    assert v["reward_breakdown"]["n_submit"] == 2
    # Mock judge is jaccard≥0.5: candidate a should match, b should not.
    assert v["reward_breakdown"]["tp"] == 1
    assert v["reward_breakdown"]["fp"] == 1
    assert 0.0 <= v["reward"] <= 1.0


@pytest.mark.asyncio
async def test_run_with_fake_policy(cfg_and_cache):
    """Inject a fake OpenAI async client so /run works without a real vLLM."""
    cfg, _ = cfg_and_cache
    app = build_app(cfg)
    agent = app.state.agent

    # Fake two-turn response: first turn issues a search + submit_match,
    # second turn returns no tool_calls to terminate.
    first = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="", tool_calls=[
                SimpleNamespace(
                    id="c1", function=SimpleNamespace(
                        name="search",
                        arguments=json.dumps({"platform": "naver", "query": "베개커버"}))),
                SimpleNamespace(
                    id="c2", function=SimpleNamespace(
                        name="submit_match",
                        arguments=json.dumps({
                            "anchor_id": "pm-pillow", "candidate_url": "http://n/a",
                            "decision": "matched",
                            "title": "데시뉴 에센스 항균 누빔 베개커버 50x70",
                            "platform": "naver"}))),
            ],
        ))]
    )
    second = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done.", tool_calls=[]))]
    )
    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=[first, second])
            )
        )
    )
    agent._client = fake

    client = TestClient(app)
    r = client.post("/run", json={
        "responses_create_params": {"input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "anchor_id=pm-pillow"},
        ]},
        "verifier_metadata": {
            "anchor_id": "pm-pillow",
            "anchor_sku": "데시뉴 에센스 항균 누빔 베개커버 50x70",
            "platforms": ["naver"],
            "budget_search": 6, "budget_submit": 12,
        },
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verify"]["reward_breakdown"]["n_search"] == 1
    assert body["verify"]["reward_breakdown"]["n_submit"] == 1
    assert body["verify"]["reward_breakdown"]["tp"] == 1
    assert body["reward"] > 0.9
