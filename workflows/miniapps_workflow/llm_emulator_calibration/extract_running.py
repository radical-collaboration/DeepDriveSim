#!/usr/bin/env python3
"""Running simulations at each measured LLM request (for the stop-decision calibration).

The workflow stops only simulations that are still running, and the real LLM saw each
simulation's state in its prompt. To calibrate P(stop | running), this counts, for
every measured ``ddsim.LLMRequest``, how many simulation tasks of the same replica were
running when the request was submitted.

How: every AsyncFlow task carries ``attributes["asyncflow.workflow_id"]``, one block per
replica; the LLM request's own task (``task_uid``) is in its replica's block. A
simulation task is an ``executable`` task of that block that had ``TaskStarted`` and no
``TaskCompleted`` / ``TaskFailed`` / ``TaskCanceled`` at ``t_submit``. (Training is also
an executable task, but it ends before the replica's steering step starts; the
``training_running`` column checks that.)

Output: ``inputs/requests_running.csv``. Usage: python extract_running.py <runs_dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def running_at_submit(events: list[dict], reqs: list[dict]) -> dict[str, int]:
    """Running simulation tasks of the request's replica at ``t_submit``, by task_uid."""
    block, kind, start, end = {}, {}, {}, {}
    for e in events:
        t, tid = e.get("event_type"), e.get("task_id")
        if t == "TaskCreated":
            block[tid] = e["attributes"].get("asyncflow.workflow_id")
            kind[tid] = e["attributes"].get("task_type")
        elif t == "TaskStarted":
            start[tid] = e["event_time"]
        elif t in ("TaskCompleted", "TaskFailed", "TaskCanceled"):
            end.setdefault(tid, e["event_time"])
    out = {}
    for r in reqs:
        b, t = block[r["task_uid"]], float(r["t_submit"])
        out[r["task_uid"]] = sum(
            1
            for tid in block
            if block[tid] == b
            and kind.get(tid) == "executable"
            and start.get(tid, float("inf")) <= t < end.get(tid, float("inf"))
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--label", default="agentic_vllm")
    args = ap.parse_args()
    reqs = [
        r
        for r in csv.DictReader(open(os.path.join(HERE, "inputs", "llm_requests.csv")))
        if r["label"] == args.label
    ]
    rows = []
    for rep in sorted({r["rep"] for r in reqs}):
        f = glob.glob(
            os.path.join(args.runs, args.label, rep, "telemetry", "*.telemetry.jsonl")
        )[0]
        events = [json.loads(line) for line in open(f)]
        mine = [x for x in reqs if x["rep"] == rep]
        n_running = running_at_submit(events, mine)
        for r in mine:
            rows.append(
                {
                    "rep": rep,
                    "replica": r["replica"],
                    "iteration": r["iteration"],
                    "task_uid": r["task_uid"],
                    "sims_in_prompt": r["sims_in_prompt"],
                    "running_tasks_at_submit": n_running[r["task_uid"]],
                    "stop_decisions": r["stop_decisions"],
                }
            )
    out = os.path.join(HERE, "inputs", "requests_running.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
