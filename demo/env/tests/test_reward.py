"""Unit tests for reward computation and response-log parsing."""

from __future__ import annotations

import json

from demo.env.config import RewardConfig
from demo.env.reward import (
    compute_reward,
    count_tool_calls,
    parse_submissions_from_response,
)
from demo.env.schema import SubmitMatchRequest, Verdict, VerifierMetadata


def _meta(**kw):
    return VerifierMetadata(
        anchor_id=kw.pop("anchor_id", "pm-x"),
        anchor_sku=kw.pop("anchor_sku", "some sku"),
        platforms=kw.pop("platforms", ["naver"]),
        budget_search=kw.pop("budget_search", 6),
        budget_submit=kw.pop("budget_submit", 12),
        **kw,
    )


def test_reward_perfect_precision_gets_bonus():
    subs = [SubmitMatchRequest(anchor_id="pm-x", candidate_url="u1", decision="matched")]
    verds = [Verdict(candidate_url="u1", decision="matched", judge_matched=True)]
    cfg = RewardConfig()
    r, bd = compute_reward(subs, verds, _meta(), {"search": 1}, cfg)
    assert bd.tp == 1 and bd.fp == 0 and bd.fn == 0
    assert r > 0.9  # precision=1 + bonus


def test_reward_false_positive_hurts():
    subs = [SubmitMatchRequest(anchor_id="pm-x", candidate_url="u1", decision="matched")]
    verds = [Verdict(candidate_url="u1", decision="matched", judge_matched=False)]
    r, bd = compute_reward(subs, verds, _meta(), {"search": 1}, RewardConfig())
    assert bd.fp == 1 and bd.tp == 0
    assert r == 0.0  # no precision, no tn, beta penalty → clipped to 0


def test_reward_mixed_precision_half():
    subs = [
        SubmitMatchRequest(anchor_id="pm-x", candidate_url="u1", decision="matched"),
        SubmitMatchRequest(anchor_id="pm-x", candidate_url="u2", decision="matched"),
    ]
    verds = [
        Verdict(candidate_url="u1", decision="matched", judge_matched=True),
        Verdict(candidate_url="u2", decision="matched", judge_matched=False),
    ]
    r, bd = compute_reward(subs, verds, _meta(), {"search": 2}, RewardConfig())
    assert bd.tp == 1 and bd.fp == 1
    assert 0.4 <= bd.precision <= 0.6
    assert 0.0 < r < 1.0


def test_reward_budget_overflow_applies_penalty():
    subs = [SubmitMatchRequest(anchor_id="pm-x", candidate_url="u1", decision="matched")]
    verds = [Verdict(candidate_url="u1", decision="matched", judge_matched=True)]
    cfg = RewardConfig(delta=0.05)
    r_ok, _ = compute_reward(subs, verds, _meta(budget_search=6), {"search": 6}, cfg)
    r_over, bd_over = compute_reward(subs, verds, _meta(budget_search=6), {"search": 10}, cfg)
    assert bd_over.budget_overflow == 4
    assert r_over < r_ok


def test_reward_no_submit_is_zero():
    r, bd = compute_reward([], [], _meta(), {}, RewardConfig(no_submit_penalty=0.0))
    assert bd.n_submit == 0
    assert r == 0.0


def test_reward_true_negative_contributes_with_alpha():
    subs = [SubmitMatchRequest(anchor_id="pm-x", candidate_url="u1", decision="not_matched")]
    verds = [Verdict(candidate_url="u1", decision="not_matched", judge_matched=False)]
    r, bd = compute_reward(subs, verds, _meta(), {}, RewardConfig(alpha=0.4, min_positive_bonus=0.0))
    assert bd.tn == 1
    assert abs(r - 0.4) < 1e-6


def test_reward_recall_computed_when_gold_given():
    subs = [SubmitMatchRequest(anchor_id="pm-x", candidate_url="https://naver.com/p/123", decision="matched")]
    verds = [Verdict(candidate_url="https://naver.com/p/123", decision="matched", judge_matched=True)]
    meta = _meta(gold_matches=[{"product_url_prefix": "https://naver.com/p/"}])
    r, bd = compute_reward(subs, verds, meta, {"search": 1}, RewardConfig())
    assert bd.match_recall == 1.0


def test_parse_submissions_filters_noise():
    output = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "thinking"}]},
        {"type": "function_call", "name": "search", "arguments": json.dumps({"platform": "naver", "query": "x"})},
        {"type": "function_call", "name": "submit_match", "arguments": json.dumps({
            "anchor_id": "pm-x", "candidate_url": "u1", "decision": "matched", "title": "t"
        })},
        {"type": "function_call", "name": "submit_match", "arguments": "not-json"},  # malformed → dropped
        {"type": "function_call", "name": "submit_match", "arguments": json.dumps({
            "anchor_id": "pm-x", "candidate_url": "u2", "decision": "not_matched"
        })},
    ]
    subs = parse_submissions_from_response(output)
    urls = [s.candidate_url for s in subs]
    assert urls == ["u1", "u2"]


def test_count_tool_calls_picks_up_blocked_outputs():
    output = [
        {"type": "function_call", "name": "search"},
        {"type": "function_call", "name": "search"},
        {"type": "function_call_output", "output": '{"status": "blocked", "error": "captcha"}'},
        {"type": "function_call", "name": "submit_match"},
    ]
    counts = count_tool_calls(output)
    assert counts["search"] == 2
    assert counts["submit_match"] == 1
    assert counts["blocked"] == 1
