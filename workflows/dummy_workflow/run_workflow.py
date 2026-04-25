#!/usr/bin/env python3
import argparse
import asyncio
import os
from pathlib import Path

from radical.asyncflow import WorkflowEngine

from ddsim.util import find_gpus, load_config, make_policies
from workflows.dummy_workflow.dummy_workflow import DummyWorkflow


async def run_dummy(config_file: str) -> None:
    cfg = load_config(config_file)
    backend = cfg.get("engine", "dragon")
    num_replicas = cfg.get("num_replicas", 1)

    # --- Build backend and asyncflow ---
    policies = [None] * num_replicas

    if backend == "dragon":
        try:
            from rhapsody.backends import DragonExecutionBackendV3

            engine_dragon = await DragonExecutionBackendV3()
            asyncflow = await WorkflowEngine.create(engine_dragon)
            policies = make_policies(find_gpus(), nprocs=num_replicas)
        except ImportError:
            backend = "concurrent"

    else:
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

    # Use campaign replica_id when available; fall back to PID for standalone runs.
    replica_id = cfg.get("replica_id", "")
    _base = replica_id.replace("_", "") if replica_id else f"dummy{os.getpid()}"
    home_dir = Path(cfg.get("home_dir", Path.home() / "DDSim")).expanduser() / _base

    def _replica_name(i: int) -> str:
        if num_replicas == 1:
            return _base
        return f"{_base}_{i + 1}"

    # --- Launch replicas ---
    replicas = [
        DummyWorkflow(
            config=cfg,
            name=_replica_name(i),
            asyncflow=asyncflow,
            home_dir=str(home_dir),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="Dummy DDSim workflow"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to workflow config file",
    )

    args = parser.parse_args()
    if not Path(args.config_file).exists():
        parser.error(f"Config file not found: {args.config_file}")

    asyncio.run(run_dummy(args.config_file))
