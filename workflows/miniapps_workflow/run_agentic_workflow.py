#!/usr/bin/env python3
"""Agentic MiniApps workflow: DeepDriveSim MiniApp with LLM steering.

Same edge as run_workflow.py (wfMiniAPI MiniApp, explicit backends, mandatory
telemetry), plus an explicit LLM inference backend registered on the same AsyncFlow
engine. Run with:

    DDSIM_BACKEND=dragon dragon run_agentic_workflow.py \
        --config_file config_agentic.yaml
"""

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

from radical.asyncflow import WorkflowEngine
from wfMiniAPI.api import MiniApp, require_dragon

from ddsim.util import load_config
from workflows.miniapps_workflow.agentic_workflow import (
    AgenticMiniAppsWorkflow,
    InferenceService,
    emit,
)
from workflows.miniapps_workflow.llm_backends import make_llm_backend


async def run_miniapps(config_file: str) -> None:
    cfg = load_config(config_file)
    workflow_class = AgenticMiniAppsWorkflow
    backend = os.environ.get("DDSIM_BACKEND", "dragon")
    num_replicas = cfg.get("num_replicas", 4)

    # --- Build backends and asyncflow (explicit; no fallback) ---
    if backend == "dragon":
        require_dragon()
        import multiprocessing as mp

        from rhapsody.backends import DragonExecutionBackend

        mp.set_start_method("dragon", force=True)
        execution_backend = await DragonExecutionBackend()
    elif backend == "concurrent":
        from rhapsody.backends import ConcurrentExecutionBackend

        execution_backend = await ConcurrentExecutionBackend()
    else:
        raise ValueError(
            f"DDSIM_BACKEND={backend!r}: expected 'dragon' or 'concurrent'"
        )
    # LLM inference backend (llm.backend: vllm = RHAPSODY Dragon inference; standin =
    # CPU-only stand-in for testing). Its start-up is timed for the service metrics.
    llm_backend, llm_info = await make_llm_backend(cfg.get("llm", {}) or {}, name="llm")
    asyncflow = await WorkflowEngine.create(backend=[execution_backend, llm_backend])

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
    service = dict(
        service=llm_info["service"],
        llm_backend=llm_info["backend"],
        model=llm_info["model"],
        num_gpus=llm_info.get("num_gpus", 0),
        t_init_start=llm_info["t_init_start"],
        t_ready=llm_info["t_ready"],
        startup_ms=(llm_info["t_ready"] - llm_info["t_init_start"]) * 1000,
        emulator_profile=llm_info.get("emulator_profile", ""),
        emulator_profile_sha256=llm_info.get("emulator_profile_sha256", ""),
    )
    emit(
        telemetry,
        InferenceService,
        backend,
        event_time=llm_info["t_ready"],
        phase="ready",
        t_stopped=-1.0,
        lifetime_ms=-1.0,
        **service,
    )

    home_dir = Path(cfg.get("home_dir", Path.home() / "MiniApps")).expanduser()

    async with (
        MiniApp(asyncflow, telemetry, backends=[execution_backend, llm_backend]),
        asyncflow.workflow_scope("ddsim-agentic"),
    ):
        # --- Launch replicas ---
        replicas = [
            workflow_class(
                config=cfg,
                name=f"min{i + 1}",
                asyncflow=asyncflow,
                telemetry=telemetry,
                home_dir=str(home_dir),
                llm_backend="llm",
                backend_label=backend,
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

            # telemetry.stop() is done by MiniApp, after the engine has shut down.
            # Shutting the engine down also stops the inference service (EC2 end).
            await asyncflow.shutdown()
            t_stopped = time.time()
            emit(
                telemetry,
                InferenceService,
                backend,
                event_time=t_stopped,
                phase="stopped",
                t_stopped=t_stopped,
                lifetime_ms=(t_stopped - llm_info["t_ready"]) * 1000,
                **service,
            )

    # after MiniApp: the sweep would otherwise cancel RHAPSODY's dispatch tasks
    pending = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    if pending:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    print()
    print("=" * 60)
    print("  Agentic MiniApps workflow complete — no errors.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_agentic_workflow.py",
        description="Agentic MiniApps workflow (LLM steering)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config_agentic.yaml",
        help="Path to workflow configuration file",
    )
    args = parser.parse_args()
    asyncio.run(run_miniapps(args.config_file))
