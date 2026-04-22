"""NeMo-Gym-style RL training environment for product matching.

Self-contained subpackage living entirely inside demo/. Wraps demo/scraper/
(Playwright adapters) as tools and demo/model_client.MatchModel as a frozen
GenRM judge, exposing a FastAPI surface that mirrors the NeMo-Gym three-
server pattern (resources + agent + policy) without importing the nemo-gym
submodule.

Entry points:
  demo.env.server          — FastAPI resources server (tools + /verify + /run)
  demo.env.agent           — tool-loop agent calling a policy vLLM endpoint
  demo.env.collect_rollouts — CLI: JSONL → N episodes → rollouts.jsonl
  demo.env.reward_profile   — pass@1 / pass@k aggregator
"""

__version__ = "0.1.0"
