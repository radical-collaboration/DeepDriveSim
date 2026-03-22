#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

import os

import yaml
from radical.asyncflow import WorkflowEngine

from workflows.ddmd_workflow.ddmd_workflow import DDMdWorkflow

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def find_gpus():
    from dragon.native.machine import Node, System
    all_gpus = []
    for huid in System().nodes:
        node = Node(huid)
        for gpu_id in node.gpus:
            all_gpus.append((node.hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=1):
    from dragon.infrastructure.policy import Policy
    policies = []
    i = 0
    for _ in range(nprocs):
        policies.append(
            Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=all_gpus[i][0],
                gpu_affinity=[all_gpus[i][1]],
            )
        )
        i = (i + 1) % len(all_gpus)
    return policies


def _load_config(config_file: str) -> dict:
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw.items()}
    return {}


async def run_ddmd(config_file: str) -> None:
    cfg = _load_config(config_file)
    backend = cfg.get("engine", "dragon")
    num_replicas = cfg.get("num_replicas", 1)
    md_config = cfg.get("ddsim_config", None)
    num_workers = cfg.get("num_workers", 1)

    # --- Build backend and asyncflow ---
    policies = [None] * num_replicas  # per-replica Dragon policy (None for concurrent mode)

    if backend == "dragon":
        from rhapsody.backends import DragonExecutionBackendV3
        engine_dragon = await DragonExecutionBackendV3(num_workers=num_workers)
        asyncflow = await WorkflowEngine.create(engine_dragon)
        policies = make_policies(find_gpus(), nprocs=num_replicas)
    else:  # concurrent
        from rhapsody.backends import ConcurrentExecutionBackend
        engine_dragon = None
        engine_concurrent = await ConcurrentExecutionBackend()
        asyncflow = await WorkflowEngine.create(engine_concurrent)

    # --- Telemetry (Dragon mode only) ---
    collector = None
    nvml_monitor = None
    if backend == "dragon":
        from ddsim.nvml_monitor import NvmlMonitor
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

        nvml_dir = "nvml-telemetry"
        Path(nvml_dir).mkdir(parents=True, exist_ok=True)
        nvml_monitor = NvmlMonitor(
            output_dir=nvml_dir,
            collection_rate=1.0,
            checkpoint_interval=30.0,
        )
        nvml_monitor.start()

    # --- Launch replicas ---
    replicas = [
        DDMdWorkflow(
            config=md_config,
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
        if collector:
            collector.stop()
        if nvml_monitor:
            nvml_monitor.stop()
            
        await asyncflow.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDriveMD workflow entry point")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    asyncio.run(run_ddmd(args.config))
