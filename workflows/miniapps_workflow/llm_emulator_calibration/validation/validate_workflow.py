#!/usr/bin/env python3
"""End-to-end validation: emulated vs measured agentic workflow.

Compares runs of the agentic workflow with ``llm.backend: emulated``
(``smc_emulator.sbatch``) against the measured campaign that the emulator was
calibrated on (``../inputs/``, label ``agentic_vllm``).

For every metric: the measured value, its 95 % bootstrap interval, the emulated
value, and whether the emulated value lies inside the interval. Request-level metrics
resample requests; run-level metrics resample runs (5 runs: wide intervals). Host load
is reported, not judged (the emulator does not emulate it).

Outputs (``validation/``):
  emulated/<label>_requests.csv, emulated/<label>_runs.csv, emulated/<label>_stages.csv
  emulated/<label>_gpu.csv (GPU busy % by requests in flight; gpu_mem.csv summary)
  workflow_comparison.csv

Usage: python validate_workflow.py <emulated_runs_dir> [labels ...]
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import statistics as st
import sys
from bisect import bisect_right

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
import calibrate  # noqa: E402
from extract_running import running_at_submit, task_roles  # noqa: E402

DRAWS = 2000


def read_csv(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --- emulated runs -> tables in the measured campaign's format -------------------------
def collect(runs_dir: str, label: str) -> dict[str, list[dict]]:
    requests, runs, stages, service, gpu = [], [], [], [], []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, label, "rep_*"))):
        rep = os.path.basename(run_dir)
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        files = glob.glob(os.path.join(run_dir, "telemetry", "*.telemetry.jsonl"))
        events = [json.loads(line) for f in files for line in open(f) if line.strip()]
        run_reqs: list[dict] = []
        ended = [e for e in events if e.get("event_type") == "wfminiapi.RunEnded"]
        node = [
            e
            for e in events
            if e.get("event_type") == "ResourceUpdate"
            and e.get("resource_scope") == "per_node"
        ]
        runs.append(
            {
                "label": label,
                "rep": rep,
                "host": meta.get("host"),
                "emulator_gpu": meta.get("emulator_gpu", ""),
                "exit_code": meta.get("exit_code"),
                "wall_s": meta.get("wall_s"),
                "workflow_duration_s": ended[0]["duration_s"] if ended else "",
                "mean_cpu_percent": round(st.fmean(e["cpu_percent"] for e in node), 3)
                if node
                else "",
                "mean_memory_percent": round(
                    st.fmean(e["memory_percent"] for e in node), 4
                )
                if node
                else "",
            }
        )
        win = []
        for e in events:
            t = e.get("event_type")
            if t == "ddsim.LLMRequest":
                requests.append(
                    {
                        "label": label,
                        "rep": rep,
                        "task_uid": e["task_uid"],
                        "status": e["status"],
                        "latency_ms": e["latency_ms"],
                        "prompt_tokens_est": e["prompt_tokens_est"],
                        "response_tokens_est": e["response_tokens_est"],
                        "sims_in_prompt": e["sims_in_prompt"],
                        "sims_scored": e["sims_scored"],
                        "stop_decisions": e["stop_decisions"],
                        "t_submit": e["t_submit"],
                        "t_response": e["t_response"],
                    }
                )
                win.append((e["t_submit"], e["t_response"]))
                run_reqs.append(requests[-1])
            elif t == "ddsim.MiniAppsStageTimer":
                stages.append(
                    {
                        "label": label,
                        "rep": rep,
                        "stage": e["stage"],
                        "duration_ms": e["duration_ms"],
                    }
                )
            elif t == "ddsim.InferenceService" and e.get("phase") == "stopped":
                service.append(
                    {
                        "label": label,
                        "rep": rep,
                        "startup_ms": e["startup_ms"],
                        "lifetime_ms": e["lifetime_ms"],
                        "llm_backend": e["llm_backend"],
                        "emulator_profile": e.get("emulator_profile", ""),
                        "emulator_profile_sha256": e.get("emulator_profile_sha256", ""),
                    }
                )
        # running sims of the replica at each request (as in extract_running.py)
        roles = task_roles(run_dir)
        n_running = running_at_submit(events, run_reqs, roles, "sim")
        n_running_resp = running_at_submit(
            events, [{**r, "t_submit": r["t_response"]} for r in run_reqs], roles, "sim"
        )
        n_training = running_at_submit(events, run_reqs, roles, "ml")
        for r in run_reqs:
            r["running_sims_at_submit"] = n_running[r["task_uid"]]
            r["running_sims_at_response"] = n_running_resp[r["task_uid"]]
            r["training_running_at_submit"] = n_training[r["task_uid"]]
        # GPU busy % by number of requests in flight (as in extract_gpu.py)
        starts, ends = sorted(a for a, _ in win), sorted(b for _, b in win)
        acc: dict[tuple, list[float]] = {}
        for e in events:
            if (
                e.get("event_type") == "ResourceUpdate"
                and e.get("resource_scope") == "per_gpu"
            ):
                n = bisect_right(starts, e["event_time"]) - bisect_right(
                    ends, e["event_time"]
                )
                acc.setdefault((e["gpu_id"], n), []).append(e["gpu_percent"])
        for (g, n), v in sorted(acc.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            gpu.append(
                {
                    "label": label,
                    "rep": rep,
                    "gpu_id": g,
                    "inflight": n,
                    "samples": len(v),
                    "mean_gpu_percent": round(st.fmean(v), 3),
                }
            )
        mem = os.path.join(run_dir, "gpu_mem.csv")
        if os.path.exists(mem):
            used = [float(r[2]) for r in csv.reader(open(mem)) if len(r) >= 5]
            total = [float(r[3]) for r in csv.reader(open(mem)) if len(r) >= 5]
            if used:
                runs[-1]["gpu_mem_used_max_mib"] = max(used)
                runs[-1]["gpu_mem_total_mib"] = total[0]
    return {
        "requests": requests,
        "runs": runs,
        "stages": stages,
        "service": service,
        "gpu": gpu,
    }


# --- statistics ---------------------------------------------------------------------------
def pct(v, q):
    s = sorted(v)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


STATS = {"mean": st.fmean, "p50": lambda v: pct(v, 50), "p95": lambda v: pct(v, 95)}


def boot_ci(values: list[float], stat, rng: random.Random) -> tuple[float, float]:
    draws = [stat([rng.choice(values) for _ in values]) for _ in range(DRAWS)]
    return pct(draws, 2.5), pct(draws, 97.5)


def compare(
    rows, metric, level, measured, emulated, stats=("mean",), judged=True, rng=None
):
    for s in stats:
        f = STATS[s]
        m, e = f(measured), f(emulated)
        lo, hi = boot_ci(measured, f, rng)
        rows.append(
            {
                "metric": metric,
                "stat": s,
                "level": level,
                "n_measured": len(measured),
                "n_emulated": len(emulated),
                "measured": round(m, 3),
                "measured_ci95": f"[{lo:.3f}, {hi:.3f}]",
                "emulated": round(e, 3),
                "emulated_vs_measured": f"{(e - m) / m * 100:+.1f} %" if m else "",
                "verdict": ("inside" if lo <= e <= hi else "OUTSIDE")
                if judged
                else "reported",
            }
        )


def values(rs: list[dict], key: str, div: float = 1.0) -> list[float]:
    return [float(r[key]) / div for r in rs]


def coverage(rs: list[dict]) -> list[float]:
    return [int(r["sims_scored"]) / int(r["sims_in_prompt"]) for r in rs]


def stop_all(rs: list[dict]) -> list[float]:
    """1 if a request whose sims were still running at the answer stopped all of them."""
    key = "running_sims_at_response"
    return [
        1.0 if int(r["stop_decisions"]) == int(r[key]) else 0.0
        for r in rs
        if int(r[key]) > 0
    ]


def gmean(rows_: list[dict], inflight: bool) -> float:
    """Sample-weighted GPU busy % with (or without) requests in flight."""
    sel = [r for r in rows_ if (int(r["inflight"]) > 0) == inflight]
    n = sum(int(r["samples"]) for r in sel)
    if not n:
        return float("nan")
    return sum(int(r["samples"]) * float(r["mean_gpu_percent"]) for r in sel) / n


def per_run(rs: list[dict]) -> list[float]:
    reps = sorted({r["rep"] for r in rs})
    return [float(sum(1 for r in rs if r["rep"] == rep)) for rep in reps]


def batch_sizes(reqs: list[dict]) -> list[float]:
    rows = [{k: str(v) for k, v in r.items()} for r in reqs]
    return [
        float(b["size"])
        for rs in calibrate.runs_of(rows).values()
        for b in calibrate.rebuild_batches(rs)
    ]


def main() -> None:
    runs_dir = sys.argv[1]
    labels = sys.argv[2:] or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(runs_dir, "*"))
    )
    out_dir = os.path.join(HERE, "emulated")
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(CAL, "inputs")
    m_req = [
        r
        for r in read_csv(os.path.join(inp, "llm_requests.csv"))
        if r["label"] == "agentic_vllm"
    ]
    m_runs = [
        r
        for r in read_csv(os.path.join(inp, "runs.csv"))
        if r["label"] == "agentic_vllm"
    ]
    m_stages = [
        r
        for r in read_csv(os.path.join(inp, "stages.csv"))
        if r["label"] == "agentic_vllm"
    ]
    m_service = [
        r
        for r in read_csv(os.path.join(inp, "llm_service.csv"))
        if r["label"] == "agentic_vllm"
    ]
    m_gpu = [
        r
        for r in read_csv(os.path.join(inp, "gpu_by_inflight.csv"))
        if r["gpu_id"] == "0"
    ]
    m_node = read_csv(os.path.join(inp, "node_usage.csv"))
    profile = json.load(
        open(os.path.join(CAL, "profile_qwen2.5-7b-instruct_h100_smc.json"))
    )
    warm_cut = calibrate.COLD_FACTOR * st.median(
        float(r["startup_ms"]) / 1000 for r in m_service
    )

    rows = []
    for label in labels:
        t = collect(runs_dir, label)
        for name, tab in t.items():
            write_csv(os.path.join(out_dir, f"{label}_{name}.csv"), tab)
        rng = random.Random(7)
        res: list[dict] = []
        compare(
            res,
            "EQ3 request latency (s)",
            "request",
            values(m_req, "latency_ms", 1000),
            values(t["requests"], "latency_ms", 1000),
            ("mean", "p50", "p95"),
            rng=rng,
        )
        compare(
            res,
            "answer size (est. tokens)",
            "request",
            values(m_req, "response_tokens_est"),
            values(t["requests"], "response_tokens_est"),
            ("mean", "p95"),
            rng=rng,
        )
        compare(
            res,
            "prompt size (est. tokens)",
            "request",
            values(m_req, "prompt_tokens_est"),
            values(t["requests"], "prompt_tokens_est"),
            rng=rng,
        )
        compare(
            res,
            "parse coverage",
            "request",
            coverage(m_req),
            coverage(t["requests"]),
            rng=rng,
        )
        m_run = read_csv(os.path.join(inp, "requests_running.csv"))
        compare(
            res,
            "stop-all decision, requests with running sims",
            "request",
            stop_all(m_run),
            stop_all(t["requests"]),
            rng=rng,
        )
        compare(
            res,
            "EQ4 completed",
            "request",
            [1.0 if r["status"] == "completed" else 0.0 for r in m_req],
            [1.0 if r["status"] == "completed" else 0.0 for r in t["requests"]],
            rng=rng,
        )
        compare(
            res,
            "batch size (requests)",
            "batch",
            batch_sizes(m_req),
            batch_sizes(t["requests"]),
            rng=rng,
        )
        compare(
            res,
            "EQ1 requests per run",
            "run",
            per_run(m_req),
            per_run(t["requests"]),
            rng=rng,
        )
        compare(
            res,
            "EC5 service startup, warm (s)",
            "run",
            [
                float(r["startup_ms"]) / 1000
                for r in m_service
                if float(r["startup_ms"]) / 1000 <= warm_cut
            ],
            values(t["service"], "startup_ms", 1000),
            rng=rng,
        )
        for stage in ("llm_steering", "training"):
            compare(
                res,
                f"stage {stage} (s)",
                "stage",
                [
                    float(r["duration_ms"]) / 1000
                    for r in m_stages
                    if r["stage"] == stage
                ],
                [
                    float(r["duration_ms"]) / 1000
                    for r in t["stages"]
                    if r["stage"] == stage
                ],
                rng=rng,
            )
        compare(
            res,
            "workflow duration (s)",
            "run",
            values(m_runs, "workflow_duration_s"),
            [float(r["workflow_duration_s"]) for r in t["runs"]],
            rng=rng,
        )
        # GPU busy %: only for gpu: load runs. The measured service used GPU 0 of an
        # exclusive node; on a shared node our GPU is the one whose load follows the
        # requests: per run, the GPU with the largest (in-flight - idle) busy %.
        load_reps = {r["rep"] for r in t["runs"] if r.get("emulator_gpu") == "load"}
        e_gpu = []
        for rep in sorted(load_reps):
            mine = [r for r in t["gpu"] if r["rep"] == rep]
            ids = sorted({str(r["gpu_id"]) for r in mine})
            best = max(
                ids,
                key=lambda g: (
                    gmean([r for r in mine if str(r["gpu_id"]) == g], True)
                    - gmean([r for r in mine if str(r["gpu_id"]) == g], False)
                ),
            )
            e_gpu += [r for r in mine if str(r["gpu_id"]) == best]
            for r in t["runs"]:
                if r["rep"] == rep:
                    r["service_gpu_id"] = best

        for inflight, name in (
            (True, "GPU busy % while requests in flight"),
            (False, "GPU busy % with no request in flight"),
        ):
            per_rep_m = [
                gmean([r for r in m_gpu if r["rep"] == rep], inflight)
                for rep in sorted({r["rep"] for r in m_gpu})
            ]
            per_rep_e = [
                gmean([r for r in e_gpu if r["rep"] == rep], inflight)
                for rep in sorted({r["rep"] for r in e_gpu})
            ]
            per_rep_e = [v for v in per_rep_e if not math.isnan(v)]
            if per_rep_e:
                compare(res, name, "run", per_rep_m, per_rep_e, rng=rng)
        mem = [r for r in t["runs"] if r.get("gpu_mem_used_max_mib")]
        if mem:
            cfg = profile["gpu"]["memory_fraction_config"]
            res.append(
                {
                    "metric": "GPU memory reserved, max (MiB)",
                    "stat": "mean",
                    "level": "run",
                    "n_measured": 0,
                    "n_emulated": len(mem),
                    "measured": "",
                    "measured_ci95": f"not measured; config {cfg} x total = {cfg * float(mem[0]['gpu_mem_total_mib']):.0f}",
                    "emulated": round(
                        st.fmean(float(r["gpu_mem_used_max_mib"]) for r in mem), 1
                    ),
                    "emulated_vs_measured": "",
                    "verdict": "reported",
                }
            )
        compare(
            res,
            "node CPU % (host load, not emulated)",
            "run",
            [
                float(r["mean_cpu_percent"])
                for r in m_node
                if r["label"] == "agentic_vllm"
            ],
            [float(r["mean_cpu_percent"]) for r in t["runs"]],
            judged=False,
            rng=rng,
        )
        compare(
            res,
            "node memory % (host load, not emulated)",
            "run",
            [
                float(r["mean_memory_percent"])
                for r in m_node
                if r["label"] == "agentic_vllm"
            ],
            [float(r["mean_memory_percent"]) for r in t["runs"]],
            judged=False,
            rng=rng,
        )
        hosts = sorted({r["host"] for r in t["runs"]})
        ok = sum(1 for r in t["runs"] if r["exit_code"] == 0)
        profiles = sorted(
            {
                (r["emulator_profile"], r["emulator_profile_sha256"][:12])
                for r in t["service"]
            }
        )
        print(
            f"== {label}: {ok}/{len(t['runs'])} runs exit 0 on {hosts}; profile {profiles}"
        )
        for r in res:
            r["label"] = label
            print(
                f"  {r['metric']:<42} {r['stat']:<4} measured {str(r['measured']):>9} {r['measured_ci95']:<22} "
                f"emulated {str(r['emulated']):>9} {r['emulated_vs_measured']:>9}  {r['verdict']}"
            )
        rows.extend(res)
    write_csv(
        os.path.join(HERE, "workflow_comparison.csv"),
        [{"label": r.pop("label"), **r} for r in rows],
    )


if __name__ == "__main__":
    main()
