"""학습된 매칭 모델 클라이언트.

vLLM OpenAI-호환 서버(`/v1/chat/completions`)를 기본 가정한다.
환경변수:
  MATCH_MODEL_URL   기본 http://localhost:8000/v1
  MATCH_MODEL_NAME  기본 nemotron-match
  MATCH_MODEL_KEY   기본 dummy (vLLM은 API 키 무시)

모델 출력 포맷: <think>...</think><label>matched|not_matched</label>
(학습 시 SYSTEM_PROMPT와 동일 규약)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import requests


MATCH_SYSTEM_PROMPT = (
    "You are a product matching expert. "
    "Given two product titles — a SKU (marketplace listing) and a Product (canonical catalog item) — "
    "determine whether they refer to the same physical product variant.\n\n"
    "You see only the raw titles. Infer brand, model/series, and variant attributes "
    "(color, size, quantity, option) from the title text itself.\n\n"
    "Rules:\n"
    "- Brand, product line/model, and variant attributes must all be consistent\n"
    "- Ignore promotional text, platform prefixes, and word order\n"
    "- Conflicting variant (different color/size/quantity) → not_matched\n"
    "- A title being less specific than the other is fine if no conflicts exist\n\n"
    "Think through your reasoning inside <think>...</think>, then output your final decision as "
    "<label>matched</label> or <label>not_matched</label>."
)


@dataclass
class MatchVerdict:
    """모델 판정 결과."""
    matched: bool
    confidence: float       # 0.0 ~ 1.0 (단일 샘플이면 이진, 선택적)
    rationale: str          # <think> 내부 텍스트
    raw_output: str
    latency_s: float = 0.0
    status: str = "OK"      # OK / FORMAT_FAIL / HTTP_ERROR


class MatchModel:
    """vLLM/Nemotron OpenAI-호환 서버에 매칭 판정을 요청하는 얇은 클라이언트."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 60,
    ):
        self.base_url = (base_url or os.getenv("MATCH_MODEL_URL",
                                               "http://localhost:8000/v1")).rstrip("/")
        self.model_name = model_name or os.getenv("MATCH_MODEL_NAME", "nemotron-match")
        self.api_key = api_key or os.getenv("MATCH_MODEL_KEY", "dummy")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def predict(self, sku_title: str, candidate_title: str) -> MatchVerdict:
        import time
        user_msg = f"SKU: {sku_title}\nProduct: {candidate_title}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": MATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return MatchVerdict(
                matched=False, confidence=0.0, rationale="", raw_output=str(e),
                latency_s=time.time() - t0, status="HTTP_ERROR",
            )

        return _parse_verdict(text, latency_s=time.time() - t0)


def _parse_verdict(text: str, latency_s: float = 0.0) -> MatchVerdict:
    """<think>...</think><label>matched|not_matched</label> 추출."""
    label_match = re.search(r"<label>\s*(matched|not_matched)\s*</label>",
                            text, re.IGNORECASE | re.DOTALL)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    rationale = think_match.group(1).strip() if think_match else ""

    if label_match:
        return MatchVerdict(
            matched=label_match.group(1).lower() == "matched",
            confidence=1.0,
            rationale=rationale,
            raw_output=text,
            latency_s=latency_s,
            status="OK",
        )
    return MatchVerdict(
        matched=False, confidence=0.0,
        rationale=rationale, raw_output=text,
        latency_s=latency_s, status="FORMAT_FAIL",
    )


class MockMatchModel:
    """서빙 없이 데모 드라이런용. 단순 휴리스틱(토큰 중복률)."""

    def predict(self, sku_title: str, candidate_title: str) -> MatchVerdict:
        import time
        t0 = time.time()
        a = set(re.findall(r"[가-힣A-Za-z0-9]+", sku_title.lower()))
        b = set(re.findall(r"[가-힣A-Za-z0-9]+", candidate_title.lower()))
        if not a or not b:
            return MatchVerdict(False, 0.0, "empty", "", time.time() - t0, "OK")
        jaccard = len(a & b) / len(a | b)
        matched = jaccard >= 0.5
        return MatchVerdict(
            matched=matched,
            confidence=jaccard,
            rationale=f"jaccard={jaccard:.2f} (mock)",
            raw_output=f"<label>{'matched' if matched else 'not_matched'}</label>",
            latency_s=time.time() - t0,
            status="OK",
        )
