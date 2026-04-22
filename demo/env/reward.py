"""Episode reward: precision-weighted, specificity-weighted, budget-penalized.

Formula (see docs/nemo-gym-market.md §6):

    r = precision + α·specificity
        − β · (FP / n_submit)
        − γ · (FN / n_submit)
        − δ · max(0, search_calls − budget_search)
        + min_positive_bonus · 𝟙[TP ≥ 1]

Clipped to [0, 1]. When verifier_metadata.gold_matches is provided we also
compute recall over gold URLs (reported in breakdown, not added to reward).
"""

from __future__ import annotations

from typing import List, Tuple

from .config import RewardConfig
from .schema import RewardBreakdown, SubmitMatchRequest, Verdict, VerifierMetadata


def compute_reward(
    submissions: List[SubmitMatchRequest],
    verdicts: List[Verdict],
    metadata: VerifierMetadata,
    tool_counts: dict,
    config: RewardConfig,
) -> Tuple[float, RewardBreakdown]:
    assert len(submissions) == len(verdicts), "submissions/verdicts length mismatch"

    n_submit = len(submissions)
    n_search = int(tool_counts.get("search", 0))
    n_inspect = int(tool_counts.get("inspect", 0))
    n_blocked = int(tool_counts.get("blocked", 0))

    tp = fp = tn = fn = 0
    for s, v in zip(submissions, verdicts):
        agent_matched = (s.decision == "matched")
        if agent_matched and v.judge_matched:
            tp += 1
        elif agent_matched and not v.judge_matched:
            fp += 1
        elif not agent_matched and not v.judge_matched:
            tn += 1
        else:
            fn += 1

    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    specificity = tn / max(tn + fn, 1) if (tn + fn) > 0 else 0.0

    budget_overflow = max(0, n_search - metadata.budget_search)
    budget_penalty = config.delta * budget_overflow

    r = precision + config.alpha * specificity
    if n_submit > 0:
        r -= config.beta * (fp / n_submit)
        r -= config.gamma * (fn / n_submit)
    r -= budget_penalty
    if tp >= 1:
        r += config.min_positive_bonus
    if n_submit == 0:
        r -= config.no_submit_penalty

    r = max(0.0, min(1.0, r))

    recall = None
    if metadata.gold_matches:
        gold_urls = {g.get("product_url_prefix") or g.get("product_url", "") for g in metadata.gold_matches}
        gold_urls.discard("")
        submitted_match = {
            s.candidate_url for s, v in zip(submissions, verdicts)
            if s.decision == "matched" and v.judge_matched
        }
        hit = sum(1 for u in submitted_match if any(u.startswith(g) for g in gold_urls))
        recall = hit / max(len(gold_urls), 1)

    return r, RewardBreakdown(
        tp=tp, fp=fp, tn=tn, fn=fn, n_submit=n_submit,
        n_search=n_search, n_inspect=n_inspect, n_blocked=n_blocked,
        precision=round(precision, 4), specificity=round(specificity, 4),
        budget_overflow=budget_overflow, budget_penalty=round(budget_penalty, 4),
        match_recall=round(recall, 4) if recall is not None else None,
    )


def parse_submissions_from_response(
    output_items: list[dict],
) -> list[SubmitMatchRequest]:
    """Walk a Responses-API output log and pull out submit_match calls.

    Tolerant to both `chat.completions` tool_calls (already normalised by the
    agent into function_call items) and raw Responses-API function_call events.
    """
    import json
    subs: list[SubmitMatchRequest] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if item.get("name") != "submit_match":
            continue
        args_raw = item.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except Exception:
            continue
        try:
            subs.append(SubmitMatchRequest.model_validate(args))
        except Exception:
            # malformed args → skip silently (counts as trajectory noise)
            continue
    return subs


def count_tool_calls(output_items: list[dict]) -> dict:
    counts: dict[str, int] = {"search": 0, "inspect": 0, "submit_match": 0, "blocked": 0}
    import json as _json
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            name = item.get("name", "")
            counts[name] = counts.get(name, 0) + 1
        elif item.get("type") == "function_call_output":
            # Sniff for blocked status in the tool result payload.
            out = item.get("output", "")
            if isinstance(out, str) and '"status": "blocked"' in out:
                counts["blocked"] += 1
    return counts
