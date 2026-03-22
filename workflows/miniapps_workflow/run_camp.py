#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

import os

import yaml
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node, System
from radical.asyncflow import WorkflowEngine

from workflows.miniapps_workflow.miniapps_workflow import MiniAppsWorkflow
from workflows.miniapps_workflow.miniapps_workflow_asyncflow import MiniAppsWorkflowAsyncflow


def find_gpus():

    all_gpus = []
    # loop through all nodes Dragon is running on
    for huid in System().nodes:
        node = Node(huid)
        # loop through however many GPUs it may have
        for gpu_id in node.gpus:
            all_gpus.append((node.hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=32):
    """Create per-process policies with round-robin GPU assignment."""
    policies = []
    i = 0
    for _worker in range(nprocs):
        policies.append(
            Policy(
                placement=Policy.Placement.HOST_NAME,
                host_name=all_gpus[i][0],
                gpu_affinity=[all_gpus[i][1]],
            )
        )
        i += 1
        if i == len(all_gpus):
            i = 0
    return policies


def _load_config(config_file: str) -> dict:
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw.items()}
    return {}


async def run_miniapps(config_file, test_asyncflow=False):
    cfg = _load_config(config_file)
    WorkflowClass = MiniAppsWorkflowAsyncflow if test_asyncflow else MiniAppsWorkflow
    use_dragon = cfg.get("engine", "concurrent") == "dragon"

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

    home_dir = Path(cfg.get("home_dir", Path.home() / "MiniApps")).expanduser()
    num_replicas = cfg.get("num_replicas", 4)

    all_gpus = find_gpus()  # e.g. [("node-0", 0), ("node-0", 1), ("node-1", 0), ...]
    policies = make_policies(all_gpus, nprocs=num_replicas)

    from rhapsody.backends import DragonTelemetryCollector

    collector_dir = "telemetry-results"
    Path(collector_dir).mkdir(parents=True, exist_ok=True)

    collector = DragonTelemetryCollector(
        collection_rate=1.0,  # Collect every second
        checkpoint_interval=30.0,  # Checkpoint every 30 seconds
        checkpoint_dir=collector_dir,  # Save checkpoints here
        checkpoint_count=150,  # Keep last 10 checkpoints
        enable_cpu=True,
        enable_gpu=True,
        enable_memory=False,
        metric_prefix="SPHERICAL-inference",  # Prefix all metrics
    )
    collector.start()

    from ddsim.nvml_monitor import NvmlMonitor

    nvml_dir = "nvml-telemetry"
    Path(nvml_dir).mkdir(parents=True, exist_ok=True)
    nvml_monitor = NvmlMonitor(
        output_dir=nvml_dir,
        collection_rate=1.0,
        checkpoint_interval=30.0,
    )
    nvml_monitor.start()

    replicas = [
        WorkflowClass(
            config=cfg,
            name=f"min{i + 1}",
            asyncflow=asyncflow,
            home_dir=f"{home_dir}/min{i + 1}",
            policies=[policies[i]],
        )
        for i in range(num_replicas)
    ]

    try:
        await asyncio.gather(*[r.start() for r in replicas])
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
    finally:
        await asyncio.gather(*[r.close() for r in replicas])
        collector.stop()
        nvml_monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_camp.py", description="MiniApps multi-replica workflow"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to workflow configuration file",
    )
    parser.add_argument(
        "--test_asyncflow",
        action="store_true",
        default=False,
        help=(
            "Use MiniAppsWorkflowAsyncflow (@self.flow decorators) instead of "
            "the working MiniAppsWorkflow (plain asyncio subprocesses). "
            "Intended for asyncflow developer testing."
        ),
    )

    args = parser.parse_args()

    asyncio.run(run_miniapps(args.config_file, test_asyncflow=args.test_asyncflow))
