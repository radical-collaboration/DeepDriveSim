#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

from radical.asyncflow import WorkflowEngine

from ddsim.util import find_gpus, load_config, make_policies
from workflows.exchange_workflow.exchange_workflow import ExchangeWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


async def run_exchange(config_file: str) -> None:
    cfg = load_config(config_file)
    backend = cfg.get("engine", "dragon")
    num_replicas = cfg.get("num_replicas", 1)

    # --- Build backend and asyncflow ---
    policy = None

    if backend == "dragon":
        try:
            from rhapsody.backends import DragonExecutionBackendV3

            engine_dragon = await DragonExecutionBackendV3()
            asyncflow = await WorkflowEngine.create(engine_dragon)
            policies = make_policies(find_gpus(), nprocs=num_replicas)
            policy = policies[0] if policies else None
        except ImportError:
            backend = "concurrent"

    if backend != "dragon":
        from rhapsody.backends import ConcurrentExecutionBackend

        engine_concurrent = await ConcurrentExecutionBackend()
        asyncflow = await WorkflowEngine.create(engine_concurrent)

    # --- Telemetry (Dragon mode only) ---
    telemetry = None
    collector = None
    if backend == "dragon":
        if hasattr(asyncflow, "start_telemetry"):
            telemetry = await asyncflow.start_telemetry(
                resource_poll_interval=0.5,
                checkpoint_path="telemetry-output",
            )
            print("Started Asyncflow telemetry ...")
        else:
            from rhapsody.backends import DragonTelemetryCollector

            collector_dir = "telemetry-results"
            Path(collector_dir).mkdir(parents=True, exist_ok=True)
            collector = DragonTelemetryCollector(
                collection_rate=1.0,
                checkpoint_interval=30.0,
                checkpoint_dir=collector_dir,
                checkpoint_count=150,
                enable_cpu=True,
                enable_gpu=True,
                enable_memory=False,
                metric_prefix="SPHERICAL-exchange",
            )
            collector.start()
            print("Started Dragon telemetry...")

    workflow = ExchangeWorkflow(
        config=cfg,
        asyncflow=asyncflow,
        policies=[policy] if policy is not None else [],
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
        if collector:
            collector.stop()

        # ExchangeWorkflow.close() calls asyncflow.shutdown() internally.
        await workflow.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="Exchange workflow entry point"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to workflow config file (default: config.yaml next to this script)",
    )

    args = parser.parse_args()

    asyncio.run(run_exchange(args.config_file))
