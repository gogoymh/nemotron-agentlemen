"""Pydantic schemas. Shape mirrors NeMo-Gym's BaseRunRequest / BaseVerifyRequest
so the env can later be ported to nemo-gym/resources_servers/product_matching/
by swapping the base class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── OpenAI Responses-API compatible event types ────────────────────────
class ResponsesCreateParams(BaseModel):
    """Subset of openai.responses.create params we actually consume."""

    input: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


class VerifierMetadata(BaseModel):
    """Task-specific verifier data. Opaque to the framework — used by verify()."""

    anchor_id: str
    anchor_sku: str
    platforms: List[str] = Field(default_factory=list)
    budget_search: int = 6
    budget_submit: int = 12
    min_submissions: int = 1
    # Optional gold annotations — when absent, judge verdict IS ground truth.
    gold_matches: Optional[List[Dict[str, str]]] = None
    neg_candidate_urls: Optional[List[str]] = None


class RunRequest(BaseModel):
    responses_create_params: ResponsesCreateParams
    verifier_metadata: VerifierMetadata


class ResponseEnvelope(BaseModel):
    """Mirror of openai.responses output — list of message/function_call/function_call_output items."""

    output: List[Dict[str, Any]] = Field(default_factory=list)
    output_text: str = ""


class VerifyRequest(BaseModel):
    response: ResponseEnvelope
    verifier_metadata: VerifierMetadata


class Verdict(BaseModel):
    candidate_url: str
    decision: str           # matched | not_matched (agent's)
    judge_matched: bool     # ground truth
    judge_confidence: float = 0.0
    judge_rationale: str = ""
    judge_status: str = "OK"
    title: str = ""
    platform: str = ""


class RewardBreakdown(BaseModel):
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    n_submit: int = 0
    n_search: int = 0
    n_inspect: int = 0
    n_blocked: int = 0
    precision: float = 0.0
    specificity: float = 0.0
    budget_overflow: int = 0
    budget_penalty: float = 0.0
    match_recall: Optional[float] = None    # only when gold_matches given


class VerifyResponse(BaseModel):
    reward: float
    reward_breakdown: RewardBreakdown
    verdicts: List[Verdict] = Field(default_factory=list)
    trajectory_len: int = 0
    tool_call_counts: Dict[str, int] = Field(default_factory=dict)


# ── Tool I/O ───────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    platform: str
    query: str
    max_results: int = 8


class CandidateOut(BaseModel):
    title: str
    product_url: str
    image_url: str = ""
    price: str = ""
    brand: str = ""


class SearchResponse(BaseModel):
    platform: str
    query: str
    status: str    # ok | blocked | error | no_cache
    source: str    # cache | live | mock | none
    candidates: List[CandidateOut] = Field(default_factory=list)
    error: str = ""


class InspectRequest(BaseModel):
    url: str


class InspectResponse(BaseModel):
    url: str
    status: str    # ok | error | not_implemented
    detail: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class SubmitMatchRequest(BaseModel):
    anchor_id: str
    candidate_url: str
    decision: str          # matched | not_matched
    title: str = ""
    platform: str = ""
    confidence: Optional[float] = None
    rationale: str = ""


class SubmitMatchResponse(BaseModel):
    status: str            # recorded | rejected
    anchor_id: str
    candidate_url: str
    decision: str
    reason: str = ""


# ── Run response (agent→env HTTP) ──────────────────────────────────────
class RunResponse(BaseModel):
    reward: float
    response: ResponseEnvelope
    verify: VerifyResponse
    verifier_metadata: VerifierMetadata
    task_index: Optional[int] = None
    error: str = ""
