"""Minimal tool-loop agent.

Policy model is called via OpenAI-compatible chat/completions (vLLM default).
Each turn:
  1. send messages + tool schemas
  2. parse tool_calls; for each, execute against the env and append a
     tool-result message + mirror the call into a Responses-API output log
     (so verify() can introspect the trajectory in the NeMo-Gym shape).
  3. stop when no tool_calls, budget exhausted, or max_turns reached.

The output log (`response.output`) is the canonical artifact; verify() reads
submit_match arguments directly from it, independent of the chat history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import EnvRootConfig
from .judge import Judge
from .pool import PlatformPool
from .reward import compute_reward, count_tool_calls, parse_submissions_from_response
from .schema import (
    InspectRequest,
    ResponseEnvelope,
    RunRequest,
    RunResponse,
    SearchRequest,
    SubmitMatchRequest,
    VerifyResponse,
)
from .tools import build_tool_schemas, exec_inspect, exec_search, exec_submit_match

log = logging.getLogger("demo.env.agent")

SYSTEM_FALLBACK = (
    "You are a product-matching agent. Given an anchor SKU, use search(platform, query) "
    "to gather candidates across the listed e-commerce platforms, optionally call "
    "inspect(url) for detail, then call submit_match(anchor_id, candidate_url, decision) "
    "once per promising candidate. Prefer decision='matched' only when brand, model, "
    "and variant (color/size/quantity) all align. Do NOT exceed the search or submit budget."
)


class Agent:
    def __init__(
        self,
        pool: PlatformPool,
        judge: Judge,
        config: EnvRootConfig,
        openai_client: Any = None,
    ):
        self.pool = pool
        self.judge = judge
        self.config = config
        # Lazy-import: only needed when we actually call the policy.
        self._client = openai_client

    # ── public entry ──────────────────────────────────────────────────
    async def run(self, req: RunRequest, task_index: Optional[int] = None) -> RunResponse:
        t0 = time.time()
        envelope = ResponseEnvelope(output=[], output_text="")
        try:
            await self._rollout(req, envelope)
        except Exception as e:
            log.exception("rollout failed")
            envelope.output_text = f"[agent error] {type(e).__name__}: {e}"

        verify_resp = await self._verify(req, envelope)
        return RunResponse(
            reward=verify_resp.reward,
            response=envelope,
            verify=verify_resp,
            verifier_metadata=req.verifier_metadata,
            task_index=task_index,
        )

    # ── rollout loop ──────────────────────────────────────────────────
    async def _rollout(self, req: RunRequest, envelope: ResponseEnvelope) -> None:
        from openai import AsyncOpenAI  # local import so tests w/o network can import the module

        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.config.policy.base_url,
                api_key=self.config.policy.api_key,
                timeout=self.config.policy.timeout_s,
            )

        platforms = req.verifier_metadata.platforms or self.config.env.platforms
        tools = build_tool_schemas(platforms)
        chat_tools = [_to_chat_tool(t) for t in tools]
        messages = _input_to_messages(req.responses_create_params.input)

        # Ensure a system message exists.
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_FALLBACK})

        ep = self.config.episode
        n_search = 0
        n_submit = 0
        budget_search = req.verifier_metadata.budget_search
        budget_submit = req.verifier_metadata.budget_submit

        for turn in range(ep.max_turns):
            completion = await self._client.chat.completions.create(
                model=self.config.policy.model_name,
                messages=messages,
                tools=chat_tools,
                temperature=self.config.policy.temperature,
                max_tokens=self.config.policy.max_tokens,
            )
            choice = completion.choices[0].message
            content = choice.content or ""
            tool_calls = getattr(choice, "tool_calls", None) or []

            # Mirror assistant message into Responses-API output log.
            if content:
                envelope.output.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls] or None,
            })

            if not tool_calls:
                break

            # Execute each tool call (cap per-turn).
            for tc in tool_calls[: ep.max_tool_calls_per_turn]:
                name = tc.function.name
                args_raw = tc.function.arguments or "{}"
                envelope.output.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": name,
                    "arguments": args_raw,
                })
                tool_output = await self._dispatch(name, args_raw, req.verifier_metadata.anchor_id)
                if name == "search":
                    n_search += 1
                elif name == "submit_match":
                    n_submit += 1

                out_str = _truncate(json.dumps(tool_output, ensure_ascii=False), ep.max_output_chars)
                envelope.output.append({
                    "type": "function_call_output",
                    "call_id": tc.id,
                    "output": out_str,
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": out_str,
                })

                # Hard stops.
                if n_search >= budget_search and name == "search":
                    break
                if n_submit >= budget_submit and name == "submit_match":
                    break

            if n_search >= budget_search and n_submit >= budget_submit:
                break

        # Populate a human-readable summary.
        envelope.output_text = f"rollout complete · search={n_search} submit={n_submit} turns={turn + 1}"

    # ── tool dispatch ─────────────────────────────────────────────────
    async def _dispatch(self, name: str, args_raw: str, anchor_id: str) -> Dict[str, Any]:
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"invalid json: {e}"}

        if name == "search":
            try:
                resp = await exec_search(self.pool, SearchRequest.model_validate(args))
            except Exception as e:
                return {"status": "error", "error": f"{type(e).__name__}: {e}"}
            return resp.model_dump()
        if name == "inspect":
            resp = await exec_inspect(InspectRequest.model_validate(args))
            return resp.model_dump()
        if name == "submit_match":
            try:
                resp = await exec_submit_match(SubmitMatchRequest.model_validate(args), anchor_id)
            except Exception as e:
                return {"status": "error", "error": f"{type(e).__name__}: {e}"}
            return resp.model_dump()
        return {"status": "error", "error": f"unknown tool: {name}"}

    # ── verify ────────────────────────────────────────────────────────
    async def _verify(self, req: RunRequest, envelope: ResponseEnvelope) -> VerifyResponse:
        submissions = parse_submissions_from_response(envelope.output)
        tool_counts = count_tool_calls(envelope.output)
        verdicts = await self.judge.verify_submissions(
            req.verifier_metadata.anchor_sku, submissions
        )
        reward, breakdown = compute_reward(
            submissions, verdicts, req.verifier_metadata, tool_counts, self.config.reward,
        )
        return VerifyResponse(
            reward=reward,
            reward_breakdown=breakdown,
            verdicts=verdicts,
            trajectory_len=len(envelope.output),
            tool_call_counts={k: int(v) for k, v in tool_counts.items()},
        )


# ── helpers ────────────────────────────────────────────────────────────
def _input_to_messages(inp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Responses-API input items → chat.completions messages.

    Accepts both already-chat-shaped items ({role, content}) and Responses
    items ({type: "message", role, content: [{type: "input_text", text}]}).
    """
    msgs: List[Dict[str, Any]] = []
    for item in inp:
        if not isinstance(item, dict):
            continue
        if "role" in item and "content" in item and isinstance(item["content"], str):
            msgs.append({"role": item["role"], "content": item["content"]})
            continue
        if item.get("type") == "message":
            role = item.get("role", "user")
            parts = item.get("content", [])
            if isinstance(parts, list):
                text = "".join(
                    p.get("text", "") for p in parts
                    if isinstance(p, dict) and p.get("type", "").endswith("text")
                )
                msgs.append({"role": role, "content": text})
            elif isinstance(parts, str):
                msgs.append({"role": role, "content": parts})
    return msgs


def _to_chat_tool(responses_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Responses-API tool schema → chat.completions tool schema."""
    return {
        "type": "function",
        "function": {
            "name": responses_tool["name"],
            "description": responses_tool.get("description", ""),
            "parameters": responses_tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _tool_call_to_dict(tc) -> Dict[str, Any]:
    return {
        "id": tc.id, "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 12] + " …[truncated]"
