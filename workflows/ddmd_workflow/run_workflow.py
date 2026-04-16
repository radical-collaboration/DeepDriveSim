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
    """Return [(hostname, gpu_id), ...] for every GPU visible to Dragon.

    Under `dragon -s` (single-node mode) node.hostname returns 'localhost',
    which resolves to host_id=-1 and causes a ~54 s scheduling timeout per
    task.  We substitute the real hostname in that case.
    """
    import socket
    from dragon.native.machine import Node, System

    real_hostname = socket.gethostname()
    all_gpus = []
    for huid in System().nodes:
        node = Node(huid)
        hostname = node.hostname if node.hostname != "localhost" else real_hostname
        for gpu_id in (node.gpus or []):
            all_gpus.append((hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=1):
    """Create one Policy per replica, pinned to a unique node/GPU slot.

    If nprocs > len(all_gpus), caps at the number of available GPU slots
    and warns rather than silently double-pinning replicas to the same GPU.
    """
    if not all_gpus:
        print("WARNING: no GPUs found via Dragon — running without GPU affinity policies")
        return [None] * nprocs
    from dragon.infrastructure.policy import Policy
    if nprocs > len(all_gpus):
        print(
            f"WARNING: num_replicas={nprocs} exceeds available GPU slots={len(all_gpus)}. "
            f"Capping at {len(all_gpus)} replicas."
        )
        nprocs = len(all_gpus)
    return [
        Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=all_gpus[i][0],
            gpu_affinity=[all_gpus[i][1]],
        )
        for i in range(nprocs)
    ]


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

    # --- Build backend and asyncflow ---
    policies = [None] * num_replicas  # per-replica Dragon policy (None for concurrent mode)

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
    # policies length may be capped by make_policies if num_replicas > available GPUs.
    num_replicas = len(policies)
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
        except:
            pass

        if telemetry:
            await telemetry.stop()
        if collector:
            collector.stop()
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
