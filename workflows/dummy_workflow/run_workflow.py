#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

import os

import yaml
from radical.asyncflow import WorkflowEngine

from workflows.dummy_workflow.dummy_workflow import DummyWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load_config(config_file: str) -> dict:
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw.items()}
    return {}


async def run_ddmd(config_file):
    cfg = _load_config(config_file)
    use_dragon = cfg.get("engine", "concurrent") == "dragon"

    if use_dragon:
        try:
            from rhapsody.backends import DragonExecutionBackendV3
        except ImportError as e:
            print(f"Dragon backend requested but not available: {e}")
            use_dragon = False

    if use_dragon:
        engine = await DragonExecutionBackendV3()
    else:
        from rhapsody.backends import ConcurrentExecutionBackend

        engine = await ConcurrentExecutionBackend()

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)

    home_dir = Path(cfg.get("home_dir", Path.home() / "DDSim")).expanduser()

    # Initialize the workflow
    workflow = DummyWorkflow(
        config=cfg,
        asyncflow=asyncflow,
        home_dir=home_dir,
    )

    try:
        await workflow.start()
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
    finally:
        await workflow.close()
        await asyncflow.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="Dummy DDSim workflow"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to workflow configuration file (default: config.yaml next to this script)",
    )

    args = parser.parse_args()

    asyncio.run(run_ddmd(args.config_file))
