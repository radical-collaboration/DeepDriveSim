#!/usr/bin/env python3
"""
Parse a rhapsody telemetry JSONL file and plot
GPU utilization per node and per GPU.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_resource_updates(telemetry_path: str):
    """Return (per_node, per_gpu) dicts keyed by (node_id,) and (node_id, gpu_id)."""
    per_node = defaultdict(lambda: {"t": [], "gpu": [], "cpu": [], "mem": []})
    per_gpu = defaultdict(lambda: {"t": [], "gpu": []})

    t0 = None
    with open(telemetry_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("event_type") != "ResourceUpdate":
                continue
            t = e["event_time"]
            if t0 is None:
                t0 = t
            t_rel = t - t0

            scope = e.get("resource_scope")
            node = e.get("node_id") or "unknown"

            if scope == "per_node":
                rec = per_node[node]
                rec["t"].append(t_rel)
                rec["gpu"].append(e.get("gpu_percent") or 0.0)
                rec["cpu"].append(e.get("cpu_percent") or 0.0)
                rec["mem"].append(e.get("memory_percent") or 0.0)

            elif scope == "per_gpu":
                gpu_id = e.get("gpu_id", 0)
                key = (node, gpu_id)
                rec = per_gpu[key]
                rec["t"].append(t_rel)
                rec["gpu"].append(e.get("gpu_percent") or 0.0)

    return per_node, per_gpu, t0


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_per_node(per_node, output_dir: Path):
    """One figure per node showing aggregate GPU, CPU, and memory utilization."""
    for node, rec in per_node.items():
        t = np.array(rec["t"])
        gpu = np.array(rec["gpu"])
        cpu = np.array(rec["cpu"])
        mem = np.array(rec["mem"])

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Node: {node}", fontsize=12, fontweight="bold")

        for ax, data, label, color in zip(
            axes,
            [gpu, cpu, mem],
            ["GPU Utilization (%)", "CPU Utilization (%)", "Memory Utilization (%)"],
            ["tab:green", "tab:blue", "tab:orange"],
        ):
            ax.plot(t, data, alpha=0.25, color=color, linewidth=0.8)
            ax.plot(t, smooth(data), color=color, linewidth=1.5, label="smoothed")
            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (s)", fontsize=10)
        fig.tight_layout()

        safe_node = node.split(".")[0]
        out = output_dir / f"gpu_util_node_{safe_node}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_per_gpu(per_gpu, output_dir: Path):
    """One figure per node with one subplot per GPU on that node."""
    # Group by node
    by_node = defaultdict(dict)
    for (node, gpu_id), rec in per_gpu.items():
        by_node[node][gpu_id] = rec

    for node, gpus in by_node.items():
        gpu_ids = sorted(gpus.keys())
        n = len(gpu_ids)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True, squeeze=False)
        fig.suptitle(f"Per-GPU Utilization — {node}", fontsize=12, fontweight="bold")

        colors = plt.cm.tab10.colors
        for row, gpu_id in enumerate(gpu_ids):
            rec = gpus[gpu_id]
            t = np.array(rec["t"])
            gpu = np.array(rec["gpu"])
            ax = axes[row][0]
            color = colors[row % len(colors)]
            ax.plot(t, gpu, alpha=0.25, color=color, linewidth=0.8)
            ax.plot(
                t,
                smooth(gpu),
                color=color,
                linewidth=1.5,
                label=f"GPU {gpu_id} (smoothed)",
            )
            ax.set_ylabel("GPU Util (%)", fontsize=9)
            ax.set_ylim(0, 105)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"GPU {gpu_id}", fontsize=10)

        axes[-1][0].set_xlabel("Time (s)", fontsize=10)
        fig.tight_layout()

        safe_node = node.split(".")[0]
        out = output_dir / f"gpu_util_per_gpu_{safe_node}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot GPU utilization from telemetry JSONL"
    )
    parser.add_argument("telemetry_file", help="Path to .telemetry.jsonl file")
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent),
        help="Directory to save PNG plots (default: same dir as this script)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.telemetry_file} ...")
    per_node, per_gpu, t0 = load_resource_updates(args.telemetry_file)
    print(f"  Nodes: {list(per_node.keys())}")
    print(f"  GPU keys: {list(per_gpu.keys())}")

    plot_per_node(per_node, output_dir)
    plot_per_gpu(per_gpu, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
