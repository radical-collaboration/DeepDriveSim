#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

from radical.asyncflow import WorkflowEngine

from ddsim.util import find_gpus, load_config, make_policies
from workflows.ddmd_workflow.ddmd_workflow import DDMdWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


async def run_ddmd(config_file: str) -> None:
    cfg = load_config(config_file)
    backend = cfg.get("engine", "dragon")
    num_replicas = cfg.get("num_replicas", 1)

    # --- Build backend and asyncflow ---
    policies = [None] * num_replicas

    if backend == "dragon":
        from rhapsody.backends import DragonExecutionBackendV3

        engine_dragon = await DragonExecutionBackendV3()
        asyncflow = await WorkflowEngine.create(engine_dragon)
        policies = make_policies(find_gpus(), nprocs=num_replicas)
    else:  # concurrent
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
                metric_prefix="SPHERICAL-inference",
            )
            collector.start()
            print("Started Dragon telemetry...")

    # --- Launch replicas ---
    replicas = [
        DDMdWorkflow(
            camp_config=config_file,
            name=f"ddmd{i + 1}",
            asyncflow=asyncflow,
            policies=[policies[i]] if policies[i] is not None else [],
        )
        for i in range(num_replicas)
    ]

    try:
        await asyncio.gather(*[r.start() for r in replicas])
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
    finally:
        await asyncio.gather(*[r.close() for r in replicas])

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
        await asyncflow.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDriveMD workflow entry point")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    asyncio.run(run_ddmd(args.config))
