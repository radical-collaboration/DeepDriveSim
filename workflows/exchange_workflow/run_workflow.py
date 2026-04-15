#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

import yaml

from radical.asyncflow import WorkflowEngine

from workflows.exchange_workflow.exchange_workflow import ExchangeWorkflow


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


async def run_ddmd(config_file):
    config = load_config(config_file)
    backend = config.get("engine", "concurrent")

    if backend == "dragon":
        try:
            from rhapsody.backends import DragonExecutionBackendV3
        except ImportError as e:
            print(f"Dragon backend requested but not available: {e}")
            backend = "concurrent"

    if backend == "dragon":
        engine = await DragonExecutionBackendV3()
    else:
        from rhapsody.backends import ConcurrentExecutionBackend

        engine = await ConcurrentExecutionBackend()

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)
    # Initialize the workflow
    workflow = ExchangeWorkflow(
                asyncflow=asyncflow,
                config=config,
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

    args = parser.parse_args()

    asyncio.run(run_ddmd(args.config_file))