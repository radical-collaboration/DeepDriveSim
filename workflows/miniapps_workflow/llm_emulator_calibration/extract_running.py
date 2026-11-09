#!/usr/bin/env python3
"""Running simulations at each measured LLM request (for the stop-decision calibration).

The workflow can stop only simulations that are still running when the answer arrives
(it counts stops against its registered sims at that moment). To calibrate the stop
decision, this counts, for every measured ``ddsim.LLMRequest``, the simulation tasks of
the same replica running at ``t_submit`` and at ``t_response``. The calibration uses
``t_response``: a decision on sims that all finished during the request cannot stop
anything, so counting at ``t_submit`` dilutes the stop probability.

How: every AsyncFlow task carries ``attributes["asyncflow.workflow_id"]``, one block per
replica; the LLM request's own task (``task_uid``) is in its replica's block. A task
"runs" at ``t_submit`` if it had ``TaskStarted`` and no ``TaskCompleted`` /
``TaskFailed`` / ``TaskCanceled`` yet. Simulation and training tasks are both
``executable``; they are told apart by the first line of the task's stdout
(``task_stdio/<session>/<task_id>.stdout``): ``Temp for Darshan, sim, ...`` or
``Temp for Darshan, ml, ...``. Training can overlap the steering step, so it must not be
counted as a running simulation.

Output: ``inputs/requests_running.csv``. Usage: python extract_running.py <runs_dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def task_roles(run_dir: str) -> dict[str, str]:
    """task_id -> 'sim' | 'ml' | '' from the first line of each task's stdout."""
    roles = {}
    for f in glob.glob(os.path.join(run_dir, "task_stdio", "*", "task.*.stdout")):
        with open(f, errors="replace") as fh:
            head = fh.readline()
        parts = [x.strip() for x in head.split(",")]
        role = (
            parts[1] if len(parts) > 1 and head.startswith("Temp for Darshan") else ""
        )
        roles[os.path.basename(f)[: -len(".stdout")]] = role
    return roles


def running_at_submit(
    events: list[dict], reqs: list[dict], roles: dict[str, str], role: str = "sim"
) -> dict[str, int]:
    """Running tasks of ``role`` in the request's replica at ``t_submit``, by task_uid."""
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
            and roles.get(tid) == role
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
        roles = task_roles(os.path.join(args.runs, args.label, rep))
        n_running = running_at_submit(events, mine, roles, "sim")
        at_resp = [{**r, "t_submit": r["t_response"]} for r in mine]
        n_running_resp = running_at_submit(events, at_resp, roles, "sim")
        n_training = running_at_submit(events, mine, roles, "ml")
        unknown = [t for t in roles if roles[t] not in ("sim", "ml")]
        for r in mine:
            rows.append(
                {
                    "rep": rep,
                    "replica": r["replica"],
                    "iteration": r["iteration"],
                    "task_uid": r["task_uid"],
                    "sims_in_prompt": r["sims_in_prompt"],
                    "running_sims_at_submit": n_running[r["task_uid"]],
                    "running_sims_at_response": n_running_resp[r["task_uid"]],
                    "training_running_at_submit": n_training[r["task_uid"]],
                    "tasks_without_role": len(unknown),
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
