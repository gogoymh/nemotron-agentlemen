"""Frozen GenRM judge — wraps demo/model_client.MatchModel.

Default points at a Nemotron-Super vLLM OpenAI-compatible endpoint. The
system prompt and <think>/<label> parser live in demo/model_client.py; we
only add async plumbing and concurrency control here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from model_client import MatchModel, MatchVerdict, MockMatchModel  # noqa: E402

from .config import JudgeConfig
from .schema import SubmitMatchRequest, Verdict


class Judge:
    """Async adapter over MatchModel.predict(). Bounded concurrency."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        if config.mock:
            self.client = MockMatchModel()
            self.label = "mock"
        else:
            self.client = MatchModel(
                base_url=config.base_url,
                model_name=config.model_name,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout_s=config.timeout_s,
            )
            self.label = config.model_name
        self._sem = asyncio.Semaphore(config.concurrency)

    async def verify_one(
        self, anchor_sku: str, candidate_title: str
    ) -> MatchVerdict:
        async with self._sem:
            return await asyncio.to_thread(
                self.client.predict, anchor_sku, candidate_title
            )

    async def verify_submissions(
        self, anchor_sku: str, submissions: List[SubmitMatchRequest]
    ) -> List[Verdict]:
        tasks = [self.verify_one(anchor_sku, s.title or s.candidate_url) for s in submissions]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[Verdict] = []
        for sub, v in zip(submissions, raw):
            if isinstance(v, BaseException):
                out.append(Verdict(
                    candidate_url=sub.candidate_url, decision=sub.decision,
                    judge_matched=False, judge_confidence=0.0,
                    judge_rationale="", judge_status=f"EXC:{type(v).__name__}",
                    title=sub.title, platform=sub.platform,
                ))
                continue
            out.append(Verdict(
                candidate_url=sub.candidate_url, decision=sub.decision,
                judge_matched=bool(v.matched), judge_confidence=float(v.confidence),
                judge_rationale=v.rationale[:512], judge_status=v.status,
                title=sub.title, platform=sub.platform,
            ))
        return out
