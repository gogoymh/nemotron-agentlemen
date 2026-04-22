"""FastAPI resources server.

Endpoint surface mirrors NeMo-Gym's SimpleResourcesServer:
  POST /tool/search          — CandidateOut[] for (platform, query)
  POST /tool/inspect         — stub, returns URL metadata
  POST /tool/submit_match    — echo/validate an agent submission
  POST /verify               — score a completed Responses-API output log
  POST /run                  — one-shot orchestration: rollout + verify
  GET  /health               — liveness + config fingerprint

Run:
    python -m demo.env.server --config demo/env/configs/product_matching.yaml
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import Agent
from .config import EnvRootConfig, load_config
from .judge import Judge
from .pool import PlatformPool
from .schema import (
    InspectRequest,
    InspectResponse,
    RunRequest,
    RunResponse,
    SearchRequest,
    SearchResponse,
    SubmitMatchRequest,
    SubmitMatchResponse,
    VerifyRequest,
    VerifyResponse,
)
from .tools import build_pool, build_tool_schemas, exec_inspect, exec_search, exec_submit_match
from .reward import compute_reward, count_tool_calls, parse_submissions_from_response


log = logging.getLogger("demo.env.server")


class Health(BaseModel):
    status: str
    judge_label: str
    live_scraping: bool
    platforms: list[str]
    config_version: str


def build_app(config: Optional[EnvRootConfig] = None) -> FastAPI:
    cfg = config or load_config()
    pool = build_pool(cfg.env)
    judge = Judge(cfg.judge)
    agent = Agent(pool=pool, judge=judge, config=cfg)

    app = FastAPI(title="demo.env · ProductMatching", version="0.1.0")
    app.state.cfg = cfg
    app.state.pool = pool
    app.state.judge = judge
    app.state.agent = agent

    @app.get("/health", response_model=Health)
    async def health() -> Health:
        return Health(
            status="ok",
            judge_label=judge.label,
            live_scraping=cfg.env.live_scraping,
            platforms=cfg.env.platforms,
            config_version="0.1.0",
        )

    @app.get("/tools")
    async def list_tools():
        return {"tools": build_tool_schemas(cfg.env.platforms)}

    @app.post("/tool/search", response_model=SearchResponse)
    async def tool_search(body: SearchRequest) -> SearchResponse:
        return await exec_search(pool, body)

    @app.post("/tool/inspect", response_model=InspectResponse)
    async def tool_inspect(body: InspectRequest) -> InspectResponse:
        return await exec_inspect(body)

    @app.post("/tool/submit_match", response_model=SubmitMatchResponse)
    async def tool_submit_match(body: SubmitMatchRequest) -> SubmitMatchResponse:
        # Server-side doesn't know the anchor context — accept as-is; verify()
        # cross-checks anchor_id against verifier_metadata.
        return await exec_submit_match(body, body.anchor_id)

    @app.post("/verify", response_model=VerifyResponse)
    async def verify(body: VerifyRequest) -> VerifyResponse:
        submissions = parse_submissions_from_response(body.response.output)
        tool_counts = count_tool_calls(body.response.output)
        verdicts = await judge.verify_submissions(body.verifier_metadata.anchor_sku, submissions)
        reward, breakdown = compute_reward(
            submissions, verdicts, body.verifier_metadata, tool_counts, cfg.reward,
        )
        return VerifyResponse(
            reward=reward, reward_breakdown=breakdown, verdicts=verdicts,
            trajectory_len=len(body.response.output),
            tool_call_counts={k: int(v) for k, v in tool_counts.items()},
        )

    @app.post("/run", response_model=RunResponse)
    async def run(body: RunRequest) -> RunResponse:
        try:
            return await agent.run(body)
        except Exception as e:
            log.exception("/run failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="demo.env FastAPI resources server")
    ap.add_argument("--config", default=None, help="Path to YAML config (defaults to configs/product_matching.yaml)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.env.host
    port = args.port or cfg.env.port

    import uvicorn
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("serving demo.env on %s:%d · judge=%s · live=%s",
             host, port, cfg.judge.model_name if not cfg.judge.mock else "mock",
             cfg.env.live_scraping)

    app = build_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
