"""Product-match GRPO entry — modeled on NeMo-RL's examples/run_grpo_math.py.

NeMo-RL doesn't dispatch data processors or reward envs by string name; the
official `run_grpo_math.py` explicitly constructs two in-memory dicts and
hands them to `grpo_train`:

    task_data_processors: {task_name: (TaskDataSpec, processor_fn)}
    task_to_env         : {task_name: ray-remote env actor}

So this script mirrors that structure for the `product_match` task and
registers our custom reward env in ACTOR_ENVIRONMENT_REGISTRY before Ray
spawns it (so Ray knows which py executable to bind).

Entry: python experiment/rl_from_sft.py --config experiment/grpo_rl.yaml [overrides]
"""
from __future__ import annotations

import argparse
import os
import pprint
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

from experiment.rl_reward_env import (
    ProductMatchRewardEnvironment,
    product_match_data_processor,
)

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

# Tell the Ray actor-env registry which python env our custom env uses.
# SYSTEM = the container's system python (ray, httpx, torch all pre-installed
# in nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano).
ACTOR_ENVIRONMENT_REGISTRY[
    "experiment.rl_reward_env.ProductMatchRewardEnvironment"
] = PY_EXECUTABLES.SYSTEM

TASK_NAME = "product_match"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run product-match GRPO training")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


TokenizerType = PreTrainedTokenizerBase


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data...")
    task_spec = TaskDataSpec(
        task_name=TASK_NAME,
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )

    data: Any = load_response_dataset(data_config, seed)

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (task_spec, product_match_data_processor))
    )
    task_data_processors[TASK_NAME] = (task_spec, product_match_data_processor)

    # Instantiate the reward env as a Ray actor. `py_executable` binds it to
    # the container's system python — see ACTOR_ENVIRONMENT_REGISTRY above.
    env_actor = ProductMatchRewardEnvironment.options(  # type: ignore[attr-defined]
        runtime_env={
            "py_executable": get_actor_python_env(
                "experiment.rl_reward_env.ProductMatchRewardEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get(TASK_NAME, {}))

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds.get("validation"):
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: env_actor)
    task_to_env[TASK_NAME] = env_actor
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        args.config = str(REPO_ROOT / "experiment" / "grpo_rl.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer, config["data"], config["env"], config["grpo"]["seed"]
    )

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    print("🚀 Running synchronous GRPO training")
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
