#!/usr/bin/env python3
"""Entry point for running the DeepDriveMD workflow.

Initializes the execution backend (Dragon or concurrent), creates the
asyncflow workflow engine, and runs the DDMdWorkflow workflow.
"""

import argparse
import asyncio
from pathlib import Path

from radical.asyncflow import WorkflowEngine

from workflows.ddmd_workflow.ddmd_workflow import DDMdWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


async def run_ddmd(config, use_dragon=False):
    """Set up the execution backend and run the DeepDriveMD workflow."""

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

    # Initialize and run the workflow
    workflow = DDMdWorkflow(asyncflow=asyncflow, config=config)
    await workflow.start()
    await workflow.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDriveMD workflow entry point")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    parser.add_argument("--use_dragon", action="store_true", help="Use Dragon backend")

    args = parser.parse_args()
    asyncio.run(run_ddmd(args.config, args.use_dragon))
