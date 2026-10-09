#!/usr/bin/env python3
import argparse
import asyncio
import os

from radical.asyncflow import WorkflowEngine

from ddsim.util import load_config
from workflows.exchange_workflow.exchange_workflow import ExchangeWorkflow


async def run_exchange(config_file: str) -> None:
    cfg = load_config(config_file)
    backend = os.environ.get("DDSIM_BACKEND", "dragon")

    # --- Build backend and asyncflow ---
    if backend == "dragon":
        try:
            from rhapsody.backends import DragonExecutionBackend

            engine_dragon = await DragonExecutionBackend()
            asyncflow = await WorkflowEngine.create(engine_dragon)
        except ImportError:
            backend = "concurrent"

    if backend != "dragon":
        from rhapsody.backends import ConcurrentExecutionBackend

        engine_concurrent = await ConcurrentExecutionBackend()
        asyncflow = await WorkflowEngine.create(engine_concurrent)

    # --- Telemetry
    telemetry = None
    if cfg.get("telemetry", True):
        telemetry_dir = cfg.get("telemetry_dir", "telemetry-output")
        if hasattr(asyncflow, "start_telemetry"):
            telemetry = await asyncflow.start_telemetry(
                resource_poll_interval=0.5,
                checkpoint_path=telemetry_dir,
            )
            print("Started Asyncflow telemetry ...")

    workflow = ExchangeWorkflow(
        config=cfg,
        asyncflow=asyncflow,
    )

    try:
        await workflow.start()
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
    finally:
        try:
            summary = telemetry.summary()
            print(f"Tasks — {summary['tasks']}")
            if summary.get("duration"):
                d = summary["duration"]
                print(
                    f"Mean task time: {d['mean_seconds']:.2f} s  "
                    f"Max: {d['max_seconds']:.2f} s"
                )
        except Exception:
            pass

        if telemetry:
            await telemetry.stop()

        # ExchangeWorkflow.close() calls asyncflow.shutdown() internally.
        await workflow.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="Exchange workflow entry point"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to workflow config file (default: config.yaml next to this script)",
    )

    args = parser.parse_args()

    asyncio.run(run_exchange(args.config_file))
