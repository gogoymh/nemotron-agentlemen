"""Tool definitions exposed to the policy model + their server-side executors."""

from __future__ import annotations

from typing import Any, Dict, List

from .config import EnvConfig
from .pool import PlatformPool
from .schema import (
    CandidateOut,
    InspectRequest,
    InspectResponse,
    SearchRequest,
    SearchResponse,
    SubmitMatchRequest,
    SubmitMatchResponse,
)


def build_tool_schemas(platforms: List[str]) -> List[Dict[str, Any]]:
    """OpenAI/Responses-API function tool schemas. Kept minimal & stable so the
    policy doesn't need to re-learn tool signatures between runs."""
    return [
        {
            "type": "function",
            "name": "search",
            "description": (
                "Search a Korean e-commerce platform for product candidates. "
                "Returns up to `max_results` cards with title, url, brand, price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "enum": platforms},
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                },
                "required": ["platform", "query"],
            },
        },
        {
            "type": "function",
            "name": "inspect",
            "description": (
                "Fetch additional detail for a product URL (stub in Phase 1 — "
                "returns the URL echo plus any cached metadata)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "type": "function",
            "name": "submit_match",
            "description": (
                "Record a final matched/not_matched verdict for an anchor SKU "
                "against a candidate URL. Call once per meaningful candidate; "
                "the env scores each via a frozen GenRM judge at episode end."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_id": {"type": "string"},
                    "candidate_url": {"type": "string"},
                    "decision": {"type": "string", "enum": ["matched", "not_matched"]},
                    "title": {"type": "string"},
                    "platform": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
                "required": ["anchor_id", "candidate_url", "decision"],
            },
        },
    ]


# ── Async executors (called by server + agent) ─────────────────────────
async def exec_search(pool: PlatformPool, req: SearchRequest) -> SearchResponse:
    import asyncio
    status, source, cands, err = await asyncio.to_thread(
        pool.search, req.platform, req.query, req.max_results
    )
    return SearchResponse(
        platform=req.platform, query=req.query, status=status,
        source=source, candidates=cands, error=err,
    )


async def exec_inspect(req: InspectRequest) -> InspectResponse:
    # Phase-1 stub: echo URL. Hook a Playwright DOM/JSON-LD fetch here later.
    if not req.url:
        return InspectResponse(url="", status="error", error="empty url")
    return InspectResponse(
        url=req.url, status="not_implemented",
        detail={"note": "Phase-1 stub. Implement with scraper/generic.inspect_page()."},
    )


async def exec_submit_match(
    req: SubmitMatchRequest, anchor_id_expected: str
) -> SubmitMatchResponse:
    if req.anchor_id != anchor_id_expected:
        return SubmitMatchResponse(
            status="rejected", anchor_id=req.anchor_id,
            candidate_url=req.candidate_url, decision=req.decision,
            reason=f"anchor_id mismatch (expected {anchor_id_expected})",
        )
    if req.decision not in ("matched", "not_matched"):
        return SubmitMatchResponse(
            status="rejected", anchor_id=req.anchor_id,
            candidate_url=req.candidate_url, decision=req.decision,
            reason="decision must be matched|not_matched",
        )
    return SubmitMatchResponse(
        status="recorded", anchor_id=req.anchor_id,
        candidate_url=req.candidate_url, decision=req.decision,
    )


def build_pool(cfg: EnvConfig) -> PlatformPool:
    return PlatformPool(
        cache_dir=cfg.cache_dir,
        live_enabled=cfg.live_scraping,
        use_system_chrome=cfg.use_system_chrome,
        allowed_platforms=cfg.platforms,
    )
