#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
from pathlib import Path

from radical.asyncflow import WorkflowEngine
from wfMiniAPI.api import MiniApp, require_dragon

from ddsim.util import load_config
from workflows.miniapps_workflow.miniapps_workflow import MiniAppsWorkflow


async def run_miniapps(config_file: str) -> None:
    cfg = load_config(config_file)
    workflow_class = MiniAppsWorkflow
    backend = os.environ.get("DDSIM_BACKEND", "dragon")
    num_replicas = cfg.get("num_replicas", 4)

    # --- Build backend and asyncflow (explicit; no fallback) ---
    if backend == "dragon":
        require_dragon()
        from rhapsody.backends import DragonExecutionBackend

        execution_backend = await DragonExecutionBackend()
    elif backend == "concurrent":
        from rhapsody.backends import ConcurrentExecutionBackend

        execution_backend = await ConcurrentExecutionBackend()
    else:
        raise ValueError(f"DDSIM_BACKEND={backend!r}: expected 'dragon' or 'concurrent'")
    asyncflow = await WorkflowEngine.create(execution_backend)

    # --- Telemetry (mandatory) ---
    if not cfg.get("telemetry", True):
        logging.getLogger(__name__).warning(
            "config 'telemetry: false' is ignored: telemetry is mandatory"
        )
    telemetry_dir = cfg.get("telemetry_dir", "telemetry-output")
    telemetry = await asyncflow.start_telemetry(
        resource_poll_interval=0.05,
        checkpoint_path=telemetry_dir,
    )
    print("Started Asyncflow telemetry ...")

    home_dir = Path(cfg.get("home_dir", Path.home() / "MiniApps")).expanduser()

    async with MiniApp(asyncflow, telemetry), asyncflow.workflow_scope("ddsim"):
        # --- Launch replicas ---
        replicas = [
            workflow_class(
                config=cfg,
                name=f"min{i + 1}",
                asyncflow=asyncflow,
                telemetry=telemetry,
                home_dir=str(home_dir),
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
            except Exception as e:
                print(f"An error occurred during telemetry summary: {e}")

            # telemetry.stop() is done by MiniApp, after the engine has shut down
            await asyncflow.shutdown()

    # after MiniApp: the sweep would otherwise cancel RHAPSODY's dispatch tasks
    pending = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    if pending:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    print()
    print("=" * 60)
    print("  MiniApps workflow complete — no errors.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="MiniApps multi-replica workflow"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to workflow configuration file",
    )

    args = parser.parse_args()

    asyncio.run(run_miniapps(args.config_file))
