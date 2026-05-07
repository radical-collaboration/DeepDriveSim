#!/usr/bin/env python3
import argparse
import asyncio

from radical.asyncflow import WorkflowEngine

from ddsim.util import find_gpus, load_config, make_policies
from workflows.ddmd_workflow.ddmd_workflow import DDMdWorkflow


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
        await asyncflow.shutdown()

        pending = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        pending = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDriveMD workflow entry point")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    asyncio.run(run_ddmd(args.config))
