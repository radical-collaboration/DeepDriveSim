#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

import yaml
from radical.asyncflow import WorkflowEngine

from workflows.miniapps_workflow.miniapps_workflow import MiniAppsWorkflow


def _load_config(config_file: str) -> dict:
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


async def run_miniapps(config_file, use_dragon):
    cfg = _load_config(config_file)

    if use_dragon:
        try:
            from rhapsody.backends import DragonExecutionBackendV3
        except ImportError:
            use_dragon = False

    if use_dragon:
        engine = await DragonExecutionBackendV3()
    else:
        from rhapsody.backends import ConcurrentExecutionBackend

        engine = await ConcurrentExecutionBackend()

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)

    home_dir = Path(cfg.get("home_dir", Path.home() / "MiniApps")).expanduser()

    # Initialize the workflow
    workflow = MiniAppsWorkflow(
        config=cfg,
        asyncflow=asyncflow,
        home_dir=home_dir,
    )

    try:
        # Run the workflow
        await workflow.start()
    except Exception as e:
        print(f"An error occurred during teaching: {e}")
    finally:
        # Ensure cleanup regardless of errors
        await workflow.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="Dummy DDSim workflow"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to workflow configuration file",
    )

    parser.add_argument("--use_dragon", action="store_true", help="Use Dragon backend")

    args = parser.parse_args()

    asyncio.run(run_miniapps(args.config_file, args.use_dragon))
