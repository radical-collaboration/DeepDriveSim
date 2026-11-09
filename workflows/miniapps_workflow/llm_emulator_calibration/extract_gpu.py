#!/usr/bin/env python3
"""Extract the LLM service's GPU load from the exp2/llm-service campaign telemetry.

Reads RHAPSODY ``ResourceUpdate`` events (``resource_scope: per_gpu``, sampled every
~0.05 s) from ``runs/<label>/rep_NN/telemetry/*.telemetry.jsonl`` and the request
windows (``t_submit`` .. ``t_response``) from ``inputs/llm_requests.csv``. For every GPU
sample it counts how many LLM requests of the same run were in flight at that instant,
and writes one row per (run, GPU, in-flight count): number of samples and mean
``gpu_percent``.

Outputs: ``inputs/gpu_by_inflight.csv``, and ``inputs/node_usage.csv`` (mean node
``cpu_percent`` / ``memory_percent`` per run, for every label, from the ``per_node``
samples) to compare the agentic and stock cells' host load. The raw runs are ``data/runs.tar.xz`` in
radical-experiments/ai-hpc-applications, exp2-agentic-deep-drive-sim/llm-service.

Usage: python extract_gpu.py <runs_dir> [--label agentic_vllm]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from bisect import bisect_right

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--label", default="agentic_vllm")
    ap.add_argument(
        "--out", default=os.path.join(HERE, "inputs", "gpu_by_inflight.csv")
    )
    args = ap.parse_args()

    windows: dict[str, list[tuple[float, float]]] = {}
    with open(os.path.join(HERE, "inputs", "llm_requests.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["label"] == args.label:
                windows.setdefault(r["rep"], []).append(
                    (float(r["t_submit"]), float(r["t_response"]))
                )

    rows = []
    for rep in sorted(windows):
        win = windows[rep]
        starts = sorted(a for a, _ in win)
        ends = sorted(b for _, b in win)
        files = glob.glob(
            os.path.join(args.runs, args.label, rep, "telemetry", "*.telemetry.jsonl")
        )
        acc: dict[tuple[int, int], list[float]] = {}
        with open(files[0]) as fh:
            for line in fh:
                e = json.loads(line)
                if (
                    e.get("event_type") != "ResourceUpdate"
                    or e.get("resource_scope") != "per_gpu"
                ):
                    continue
                t = e["event_time"]
                n = bisect_right(starts, t) - bisect_right(
                    ends, t
                )  # started and not yet ended
                acc.setdefault((e["gpu_id"], n), []).append(float(e["gpu_percent"]))
        for (gpu, n), vals in sorted(acc.items()):
            rows.append(
                {
                    "label": args.label,
                    "rep": rep,
                    "gpu_id": gpu,
                    "inflight": n,
                    "samples": len(vals),
                    "mean_gpu_percent": round(sum(vals) / len(vals), 3),
                }
            )
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")

    node_rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs, "*", "rep_*"))):
        cpu, mem = [], []
        for f in glob.glob(os.path.join(run_dir, "telemetry", "*.telemetry.jsonl")):
            with open(f) as fh:
                for line in fh:
                    e = json.loads(line)
                    if (
                        e.get("event_type") == "ResourceUpdate"
                        and e.get("resource_scope") == "per_node"
                    ):
                        cpu.append(float(e["cpu_percent"]))
                        mem.append(float(e["memory_percent"]))
        if cpu:
            node_rows.append(
                {
                    "label": os.path.basename(os.path.dirname(run_dir)),
                    "rep": os.path.basename(run_dir),
                    "samples": len(cpu),
                    "mean_cpu_percent": round(sum(cpu) / len(cpu), 3),
                    "mean_memory_percent": round(sum(mem) / len(mem), 4),
                }
            )
    out2 = os.path.join(os.path.dirname(args.out), "node_usage.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(node_rows[0]))
        w.writeheader()
        w.writerows(node_rows)
    print(f"{len(node_rows)} rows -> {out2}")


if __name__ == "__main__":
    main()
