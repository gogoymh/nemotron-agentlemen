"""Collect N rollouts per task from a JSONL input.

Mirrors ``ng_collect_rollouts`` in spirit: each JSONL line has
``responses_create_params`` + ``verifier_metadata``. We dispatch each (task,
repeat) pair through a single Agent instance (shared judge + pool), write
one JSONL row per rollout.

Usage:
    python -m demo.env.collect_rollouts \\
        --input  demo/env/data/example.jsonl \\
        --output results/rollouts.jsonl \\
        --num-repeats 5 \\
        --concurrency 4 \\
        --config demo/env/configs/product_matching.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import Agent
from .config import EnvRootConfig, load_config
from .judge import Judge
from .schema import RunRequest
from .tools import build_pool


log = logging.getLogger("demo.env.collect")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{i + 1}: bad JSON ({e})") from e
    return tasks


def _to_run_request(raw: Dict[str, Any]) -> RunRequest:
    return RunRequest.model_validate(raw)


async def _run_all(
    tasks: List[Dict[str, Any]],
    num_repeats: int,
    concurrency: int,
    config: EnvRootConfig,
    out_path: Path,
    progress_every: int,
) -> Dict[str, Any]:
    pool = build_pool(config.env)
    judge = Judge(config.judge)
    agent = Agent(pool=pool, judge=judge, config=config)

    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")
    lock = asyncio.Lock()

    async def run_one(task_idx: int, repeat_idx: int, raw: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal done
        async with sem:
            req = _to_run_request(raw)
            try:
                resp = await agent.run(req, task_index=task_idx)
                row = {
                    "task_index": task_idx,
                    "repeat_index": repeat_idx,
                    "reward": resp.reward,
                    "reward_breakdown": resp.verify.reward_breakdown.model_dump(),
                    "verdicts": [v.model_dump() for v in resp.verify.verdicts],
                    "tool_call_counts": resp.verify.tool_call_counts,
                    "trajectory_len": resp.verify.trajectory_len,
                    "verifier_metadata": resp.verifier_metadata.model_dump(),
                    "response": resp.response.model_dump(),
                }
            except Exception as e:
                log.exception("task %d repeat %d crashed", task_idx, repeat_idx)
                row = {
                    "task_index": task_idx,
                    "repeat_index": repeat_idx,
                    "reward": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                }
            async with lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1
                if progress_every and done % progress_every == 0:
                    log.info("progress %d/%d · %.1fs elapsed",
                             done, len(tasks) * num_repeats, time.time() - t0)
            return row

    coros = [
        run_one(i, r, tasks[i])
        for i in range(len(tasks))
        for r in range(num_repeats)
    ]
    results = await asyncio.gather(*coros, return_exceptions=False)
    out_f.close()

    elapsed = time.time() - t0
    rewards = [r["reward"] for r in results]
    summary = {
        "n_rollouts": len(results),
        "n_tasks": len(tasks),
        "num_repeats": num_repeats,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "positive_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0,
        "elapsed_s": round(elapsed, 2),
        "throughput_rps": round(len(results) / elapsed, 3) if elapsed > 0 else 0.0,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect RL rollouts against the product_matching env")
    ap.add_argument("--input", required=True, help="Input JSONL (responses_create_params + verifier_metadata)")
    ap.add_argument("--output", required=True, help="Output rollouts JSONL")
    ap.add_argument("--num-repeats", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--progress-every", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfg = load_config(args.config)
    tasks = _load_jsonl(Path(args.input))
    log.info("loaded %d tasks from %s", len(tasks), args.input)

    summary = asyncio.run(_run_all(
        tasks=tasks,
        num_repeats=args.num_repeats,
        concurrency=args.concurrency,
        config=cfg,
        out_path=Path(args.output),
        progress_every=args.progress_every,
    ))

    log.info("done · wrote %d rollouts → %s", summary["n_rollouts"], args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
