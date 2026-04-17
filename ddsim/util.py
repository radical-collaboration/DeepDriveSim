"""Utilities for Dragon GPU discovery, task-placement policies, and config loading."""

import os
from pathlib import Path

import yaml


def load_config(config_file: str) -> dict:
    """Load a YAML config file, expanding environment variables in string values."""
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return {
            k: os.path.expandvars(v) if isinstance(v, str) else v
            for k, v in raw.items()
        }
    return {}


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
        for gpu_id in node.gpus or []:
            all_gpus.append((hostname, gpu_id))
    return all_gpus


def make_policies(all_gpus, nprocs=1):
    """Create one Policy per replica, round-robin across available GPU slots.

    If nprocs > len(all_gpus), replicas are oversubscribed across GPUs
    (multiple replicas may share the same GPU).
    """
    if not all_gpus:
        print(
            "WARNING: no GPUs found via Dragon — running without GPU affinity policies"
        )
        return [None] * nprocs
    from dragon.infrastructure.policy import Policy

    if nprocs > len(all_gpus):
        print(
            f"WARNING: num_replicas={nprocs} exceeds available GPU"
            f" slots={len(all_gpus)}. Oversubscribing GPUs."
        )
    return [
        Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=all_gpus[i % len(all_gpus)][0],
            gpu_affinity=[all_gpus[i % len(all_gpus)][1]],
        )
        for i in range(nprocs)
    ]
