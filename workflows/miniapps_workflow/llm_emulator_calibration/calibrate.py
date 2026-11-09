#!/usr/bin/env python3
"""Build the LLM-emulator profile from the exp2/llm-service measurements.

Inputs (``inputs/``, copied from the campaign; checksums in ``inputs/SOURCES.json``):

* ``llm_requests.csv``   one row per measured LLM request (EQ3 latency, answer size,
                         submit/response timestamps, sims in prompt / scored / stopped)
* ``llm_service.csv``    service startup (EC5) per run
* ``gpu_by_inflight.csv`` GPU busy % by number of requests in flight (``extract_gpu.py``)
* ``node_usage.csv``     node CPU / memory per run, agentic and stock (``extract_gpu.py``)
* ``requests_running.csv`` running sims of the replica at each request (``extract_running.py``)

Outputs:

* ``profile_qwen2.5-7b-instruct_h100_smc.json``  the profile the emulator loads
* ``calibration/batches.csv``       the batches rebuilt from the measurements: members,
                                    start, end, model time (end - start - batch wait)
* ``calibration/mechanism_check.csv`` replay with the *observed* batch times: checks the
                                    batching rules reproduce every measured latency
* ``calibration/models.csv``        fit quality of the batch model and two alternatives
* ``calibration/fit_requests.csv``  every request: measured vs predicted latency
* ``calibration/loro.csv``          leave-one-run-out: fit on 4 runs, predict the 5th
* ``calibration/distribution.csv``  measured vs emulated latency distribution (Monte Carlo
                                    with the calibrated noise and sampled answers)

Every number in the profile is computed by a rule written here or fitted here. Typed in
by hand, with their source: ``batch_wait_s`` and ``max_batch_size`` (RHAPSODY's
``DragonVllmInferenceBackend`` defaults, rhapsody 142f396), ``memory_fraction_config``
(the campaign's vLLM config, not a measurement), and the provenance.

Usage: python calibrate.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from llm_emulator import CHARS_PER_TOKEN, sample_answer, simulate_batches  # noqa: E402

INPUTS = os.path.join(HERE, "inputs")
OUT_DIR = os.path.join(HERE, "calibration")
PROFILE = os.path.join(HERE, "profile_qwen2.5-7b-instruct_h100_smc.json")
LABEL = "agentic_vllm"
COLD_FACTOR = (
    1.3  # a startup > 1.3 x the median startup is a cold start (weights not cached)
)
MEMORY_FRACTION_CONFIG = (
    0.85  # llm.gpu_memory_utilization of the campaign (config, not measured)
)
BATCH_WAIT_S = (
    0.5  # RHAPSODY DragonVllmInferenceBackend max_batch_wait_ms=500 (142f396)
)
MAX_BATCH_SIZE = (
    1024  # RHAPSODY DragonVllmInferenceBackend max_batch_size=1024 (142f396)
)
SAME_BATCH_S = (
    0.05  # requests answered within 50 ms of each other came back in one batch
)
MC_DRAWS = 2000


def read(name: str) -> list[dict]:
    with open(os.path.join(INPUTS, name)) as fh:
        return list(csv.DictReader(fh))


def write(path: str, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------------
# Batches and latency fit
# ---------------------------------------------------------------------------------
def runs_of(reqs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in reqs:
        out.setdefault(r["rep"], []).append(r)
    return out


def rebuild_batches(reqs: list[dict]) -> list[dict]:
    """Batches of one run's requests, from the measurements.

    Requests answered together (within SAME_BATCH_S) form a batch. The batch loop
    starts collecting at the batch's first submission, or when the previous batch
    ended if that is later (the request was queued). Model time = end - start - wait.
    """
    rs = sorted(reqs, key=lambda r: float(r["t_response"]))
    groups: list[list[dict]] = []
    for r in rs:
        if (
            groups
            and float(r["t_response"]) - float(groups[-1][-1]["t_response"])
            < SAME_BATCH_S
        ):
            groups[-1].append(r)
        else:
            groups.append([r])
    out, prev_end = [], -math.inf
    for g in groups:
        first = min(float(r["t_submit"]) for r in g)
        end = max(float(r["t_response"]) for r in g)
        start = max(first, prev_end)
        out.append(
            {
                "rep": g[0]["rep"],
                "size": len(g),
                "queued": first < prev_end,
                "max_tokens": max(int(r["response_tokens_est"]) for r in g),
                "start": start,
                "end": end,
                "model_time_s": end - start - BATCH_WAIT_S,
                "members": g,
            }
        )
        prev_end = end
    return out


def fit_batch_model(batches: list[dict]) -> dict:
    """Least squares model_time = overhead + s_per_token * max_tokens (overhead >= 0)."""
    x = [b["max_tokens"] for b in batches]
    y = [b["model_time_s"] for b in batches]
    xm, ym = st.fmean(x), st.fmean(y)
    s = sum((a - xm) * (c - ym) for a, c in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    o = ym - s * xm
    if o < 0:  # through the origin
        o, s = 0.0, sum(a * c for a, c in zip(x, y)) / sum(a * a for a in x)
    resid = [math.log(c / (o + s * a)) for a, c in zip(x, y)]
    return {
        "batch_wait_s": BATCH_WAIT_S,
        "max_batch_size": MAX_BATCH_SIZE,
        "overhead_s": round(o, 5),
        "s_per_token": round(s, 7),
        "sigma": round(st.stdev(resid), 5),
    }


def _lsq(rows: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares by normal equations (Gaussian elimination)."""
    k = len(rows[0])
    mat = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    v = [sum(r[i] * t for r, t in zip(rows, y)) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            f = mat[j][i] / mat[i][i]
            mat[j] = [a - f * b for a, b in zip(mat[j], mat[i])]
            v[j] -= f * v[i]
    x = [0.0] * k
    for i in reversed(range(k)):
        x[i] = (v[i] - sum(mat[i][j] * x[j] for j in range(i + 1, k))) / mat[i][i]
    return x


def batch_model_alternatives(batches: list[dict]) -> list[dict]:
    """Batch model time explained by other features (why the emulator uses max tokens only)."""
    y = [b["model_time_s"] for b in batches]
    ybar = st.fmean(y)
    feats = {
        "a + b x longest answer (unconstrained)": lambda b: [1, b["max_tokens"]],
        "a + b x longest answer + c x longest x (size - 1)": lambda b: [
            1,
            b["max_tokens"],
            b["max_tokens"] * (b["size"] - 1),
        ],
        "a + b x longest answer + c x size": lambda b: [1, b["max_tokens"], b["size"]],
        "a + b x sum of answers": lambda b: [
            1,
            sum(int(r["response_tokens_est"]) for r in b["members"]),
        ],
    }
    out = []
    for name, f in feats.items():
        xs = [f(b) for b in batches]
        c = _lsq(xs, y)
        pred = [sum(a * x for a, x in zip(c, row)) for row in xs]
        out.append(
            {
                "model": name,
                "coefficients": " ".join(f"{v:.5f}" for v in c),
                "n_batches": len(y),
                "r2": round(
                    1
                    - sum((a - p) ** 2 for a, p in zip(y, pred))
                    / sum((a - ybar) ** 2 for a in y),
                    3,
                ),
                "mae_s": round(st.fmean(abs(a - p) for a, p in zip(y, pred)), 3),
            }
        )
    return out


def replay(reqs: list[dict], decode: dict, rng=None, tokens=None) -> dict[int, float]:
    """Latency per request (keyed by id(row)), each run replayed through the batch loop."""
    pred = {}
    for rs in runs_of(reqs).values():
        req = [
            (
                float(r["t_submit"]),
                tokens[id(r)] if tokens else float(r["response_tokens_est"]),
            )
            for r in rs
        ]
        lat, _ = simulate_batches(req, decode, rng)
        for r, v in zip(rs, lat):
            pred[id(r)] = v
    return pred


def fit_per_request_linear(reqs: list[dict]) -> tuple[float, float]:
    """Alternative without batching: latency = a + b * own answer tokens."""
    x = [float(r["response_tokens_est"]) for r in reqs]
    y = [float(r["latency_ms"]) / 1000 for r in reqs]
    xm, ym = st.fmean(x), st.fmean(y)
    b = sum((a - xm) * (c - ym) for a, c in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    return ym - b * xm, b


def errors(reqs, pred) -> dict:
    e = [pred[id(r)] - float(r["latency_ms"]) / 1000 for r in reqs]
    y = [float(r["latency_ms"]) / 1000 for r in reqs]
    ybar = st.fmean(y)
    ss_tot = sum((v - ybar) ** 2 for v in y)
    return {
        "n": len(e),
        "mae_s": round(st.fmean(abs(x) for x in e), 3),
        "rmse_s": round(math.sqrt(st.fmean(x * x for x in e)), 3),
        "bias_s": round(st.fmean(e), 3),
        "r2": round(1 - sum(x * x for x in e) / ss_tot, 3),
    }


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def summary(v: list[float]) -> dict:
    return {
        "mean_s": round(st.fmean(v), 3),
        "p50_s": round(pct(v, 50), 3),
        "p95_s": round(pct(v, 95), 3),
        "max_s": round(max(v), 3),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    reqs = [r for r in read("llm_requests.csv") if r["label"] == LABEL]
    service = [r for r in read("llm_service.csv") if r["label"] == LABEL]
    gpu = [r for r in read("gpu_by_inflight.csv") if r["label"] == LABEL]
    node = read("node_usage.csv")

    # --- batches rebuilt from the measurements ---
    batches = [b for rs in runs_of(reqs).values() for b in rebuild_batches(rs)]
    write(
        os.path.join(OUT_DIR, "batches.csv"),
        [
            {
                "rep": b["rep"],
                "size": b["size"],
                "queued": b["queued"],
                "max_tokens": b["max_tokens"],
                "start_rel_s": round(
                    b["start"]
                    - min(float(r["t_submit"]) for r in reqs if r["rep"] == b["rep"]),
                    3,
                ),
                "model_time_s": round(b["model_time_s"], 3),
                "tasks": " ".join(r["task_uid"] for r in b["members"]),
            }
            for b in batches
        ],
    )

    # --- mechanism check: the batch loop with the observed batch times ---
    mech = []
    for rep, rs in sorted(runs_of(reqs).items()):
        bs = rebuild_batches(rs)
        lat, sim = simulate_batches(
            [(float(r["t_submit"]), float(r["response_tokens_est"])) for r in rs],
            {"batch_wait_s": BATCH_WAIT_S, "max_batch_size": MAX_BATCH_SIZE},
            batch_times=[b["model_time_s"] for b in bs],
        )
        same = sum(
            {rs[i]["task_uid"] for i in sb["members"]}
            == {r["task_uid"] for r in b["members"]}
            for sb, b in zip(sim, bs)
        )
        err = [abs(lat[i] - float(rs[i]["latency_ms"]) / 1000) for i in range(len(rs))]
        mech.append(
            {
                "rep": rep,
                "requests": len(rs),
                "batches": len(bs),
                "batches_with_same_members": same,
                "max_abs_error_s": round(max(err), 4),
                "mean_abs_error_s": round(st.fmean(err), 4),
            }
        )
    write(os.path.join(OUT_DIR, "mechanism_check.csv"), mech)

    # --- batch model and alternatives (all data, noise-free) ---
    decode = fit_batch_model(batches)
    write(os.path.join(OUT_DIR, "batch_models.csv"), batch_model_alternatives(batches))
    a_lin, b_lin = fit_per_request_linear(reqs)
    mean_lat = st.fmean(float(r["latency_ms"]) / 1000 for r in reqs)
    p_batch = replay(reqs, decode)
    p_lin = {id(r): a_lin + b_lin * float(r["response_tokens_est"]) for r in reqs}
    p_const = {id(r): mean_lat for r in reqs}
    models = [
        {
            "model": "constant (mean latency)",
            "params": f"{mean_lat:.3f} s",
            **errors(reqs, p_const),
        },
        {
            "model": "per request: a + b x own answer tokens (no batching)",
            "params": f"a={a_lin:.3f} b={b_lin:.5f}",
            **errors(reqs, p_lin),
        },
        {
            "model": "RHAPSODY batch loop + batch time (emulator, noise-free)",
            "params": f"overhead={decode['overhead_s']} s_per_token={decode['s_per_token']} sigma={decode['sigma']}",
            **errors(reqs, p_batch),
        },
    ]
    write(os.path.join(OUT_DIR, "models.csv"), models)
    write(
        os.path.join(OUT_DIR, "fit_requests.csv"),
        [
            {
                "rep": r["rep"],
                "task_uid": r["task_uid"],
                "response_tokens_est": r["response_tokens_est"],
                "measured_s": round(float(r["latency_ms"]) / 1000, 3),
                "emulator_s": round(p_batch[id(r)], 3),
                "no_batching_s": round(p_lin[id(r)], 3),
            }
            for r in reqs
        ],
    )

    # --- leave one run out ---
    loro = []
    reps = sorted(runs_of(reqs))
    for held in reps:
        train = [r for r in reqs if r["rep"] != held]
        test = [r for r in reqs if r["rep"] == held]
        dec = fit_batch_model(
            [b for rs in runs_of(train).values() for b in rebuild_batches(rs)]
        )
        al, bl = fit_per_request_linear(train)
        mc = st.fmean(float(r["latency_ms"]) / 1000 for r in train)
        loro.append(
            {
                "held_out": held,
                "n": len(test),
                "s_per_token_fit": dec["s_per_token"],
                "mae_emulator_s": errors(test, replay(test, dec))["mae_s"],
                "mae_no_batching_s": errors(
                    test,
                    {id(r): al + bl * float(r["response_tokens_est"]) for r in test},
                )["mae_s"],
                "mae_constant_s": errors(test, {id(r): mc for r in test})["mae_s"],
            }
        )
    write(os.path.join(OUT_DIR, "loro.csv"), loro)

    # --- answers ---
    tokens_per_sim = sorted(
        round(float(r["response_tokens_est"]) / int(r["sims_in_prompt"]), 3)
        for r in reqs
        if int(r["sims_in_prompt"])
    )
    sims_total = sum(int(r["sims_in_prompt"]) for r in reqs)
    scored = sum(int(r["sims_scored"]) for r in reqs)
    stops = sum(int(r["stop_decisions"]) for r in reqs)
    # stop decisions: per request, all running sims or none (checked here)
    running_rows = read("requests_running.csv")
    # sims still running when the answer arrives: the only ones a decision can stop
    key = "running_sims_at_response"
    with_running = [r for r in running_rows if int(r[key]) > 0]
    stop_all = [r for r in with_running if int(r["stop_decisions"]) == int(r[key])]
    partial = [r for r in with_running if 0 < int(r["stop_decisions"]) < int(r[key])]
    over = [r for r in running_rows if int(r["stop_decisions"]) > int(r[key])]
    if partial or over:
        raise SystemExit(
            f"stop rule does not hold: partial {len(partial)}, over {len(over)}"
        )
    by_it = {
        it: [
            sum(1 for r in stop_all if r["iteration"] == it),
            sum(1 for r in with_running if r["iteration"] == it),
        ]
        for it in sorted({r["iteration"] for r in with_running})
    }

    # --- service startup ---
    startups = [float(r["startup_ms"]) / 1000 for r in service]
    med = st.median(startups)
    warm = [round(s, 3) for s in startups if s <= COLD_FACTOR * med]
    cold = [round(s, 3) for s in startups if s > COLD_FACTOR * med]

    # --- GPU (the service's GPU is the one with load; the others are reported) ---
    def gpu_mean(rows):
        n = sum(int(r["samples"]) for r in rows)
        return (
            (sum(int(r["samples"]) * float(r["mean_gpu_percent"]) for r in rows) / n, n)
            if n
            else (0.0, 0)
        )

    by_gpu: dict[str, list[dict]] = {}
    for r in gpu:
        by_gpu.setdefault(r["gpu_id"], []).append(r)
    busy_gpu = max(
        by_gpu,
        key=lambda g: gpu_mean([r for r in by_gpu[g] if int(r["inflight"]) > 0])[0],
    )
    inflight_mean, n_in = gpu_mean(
        [r for r in by_gpu[busy_gpu] if int(r["inflight"]) > 0]
    )
    idle_mean, n_idle = gpu_mean(
        [r for r in by_gpu[busy_gpu] if int(r["inflight"]) == 0]
    )
    by_n = {}
    for k in sorted({int(r["inflight"]) for r in by_gpu[busy_gpu]}):
        m, n = gpu_mean([r for r in by_gpu[busy_gpu] if int(r["inflight"]) == k])
        by_n[str(k)] = {"mean_percent": round(m, 2), "samples": n}
    others = max(gpu_mean([r for r in by_gpu[g]])[0] for g in by_gpu if g != busy_gpu)

    def node_mean(label, key):
        return round(st.fmean(float(r[key]) for r in node if r["label"] == label), 3)

    profile = {
        "schema": "ddsim.llm_emulator.profile/1",
        "description": "LLM service emulator profile: Qwen2.5-7B-Instruct, vLLM 0.19.1 via Dragon 0.15.0 "
        "Inference (RHAPSODY DragonVllmInferenceBackend), 1 H100 80GB, SMC node smc4",
        "source": json.load(open(os.path.join(INPUTS, "SOURCES.json"))),
        "calibrated_range": {
            "requests": len(reqs),
            "runs": len(reps),
            "prompt_tokens_est": [
                min(int(r["prompt_tokens_est"]) for r in reqs),
                max(int(r["prompt_tokens_est"]) for r in reqs),
            ],
            "response_tokens_est": [
                min(int(r["response_tokens_est"]) for r in reqs),
                max(int(r["response_tokens_est"]) for r in reqs),
            ],
            "sims_in_prompt": [
                min(int(r["sims_in_prompt"]) for r in reqs),
                max(int(r["sims_in_prompt"]) for r in reqs),
            ],
            "max_tokens": 512,
            "concurrent_requests": [1, max(int(k) for k in by_n)],
            "note": "outside this range the emulator extrapolates (e.g. prefill is a constant for ~2k-token prompts)",
        },
        "service": {
            "startup_s": {"warm": warm, "cold": cold},
            "cold_rule": f"startup > {COLD_FACTOR} x median startup ({med:.3f} s)",
        },
        "decode": {
            **decode,
            "meaning": "RHAPSODY batch loop: collect batch_wait_s, then the batch takes "
            "(overhead_s + s_per_token * longest answer tokens) * exp(N(-sigma^2/2, sigma))",
            "batches_fitted": len(batches),
            "fit_noise_free": models[2],
        },
        "answer": {
            "tokens_per_sim": tokens_per_sim,
            "chars_per_token": CHARS_PER_TOKEN,
            "unscored_prob": round(1 - scored / sims_total, 5),
            "stop_all_running_prob": round(len(stop_all) / len(with_running), 5),
            "stop_score": 0.0,
            "stop_rule": "per request, the model stopped every running sim (score 0.0) or "
            "none, never some (running = the replica's sim tasks still running when the "
            "answer arrived, from telemetry). Emulated: with this probability every sim "
            "scores 0.0",
            "stop_counts": {
                "counted_at": key,
                "requests_with_running_sims": len(with_running),
                "stopped_all": len(stop_all),
                "stopped_none": len(with_running) - len(stop_all),
                "stopped_some": len(partial),
                "by_iteration_stopped_all_of": by_it,
                "requests_with_running_sims_at_submit": sum(
                    int(r["running_sims_at_submit"]) > 0 for r in running_rows
                ),
                "requests_whose_sims_all_ended_during_the_request": sum(
                    int(r["running_sims_at_submit"]) > 0 and int(r[key]) == 0
                    for r in running_rows
                ),
                "stop_decisions": stops,
            },
            "counts": {"sims_in_prompt": sims_total, "sims_scored": scored},
        },
        "gpu": {
            "gpu_id": int(busy_gpu),
            "busy_percent_inflight": round(inflight_mean, 2),
            "busy_percent_idle": round(idle_mean, 2),
            "busy_percent_by_inflight": by_n,
            "samples": {"inflight": n_in, "idle": n_idle},
            "max_mean_percent_other_gpus": round(others, 3),
            "memory_fraction_config": MEMORY_FRACTION_CONFIG,
            "memory_note": "GPU memory was not measured; vLLM reserves gpu_memory_utilization of the GPU",
        },
        "host_not_emulated": {
            "node_cpu_percent": {
                "agentic": node_mean(LABEL, "mean_cpu_percent"),
                "stock": node_mean("stock", "mean_cpu_percent"),
            },
            "node_memory_percent": {
                "agentic": node_mean(LABEL, "mean_memory_percent"),
                "stock": node_mean("stock", "mean_memory_percent"),
            },
        },
    }
    # --- distribution: measured vs emulated (noise on; measured or sampled answers) ---
    rng = random.Random(1)
    measured = [float(r["latency_ms"]) / 1000 for r in reqs]
    emu_tok, emu_full, emu_size = [], [], []
    for _ in range(MC_DRAWS):
        emu_tok.extend(replay(reqs, decode, rng).values())
        toks = {}
        for r in reqs:
            prompt = " ".join(f"sim_{k}" for k in range(int(r["sims_in_prompt"])))
            ans = sample_answer(prompt, profile, 512, rng)
            toks[id(r)] = len(ans) / CHARS_PER_TOKEN
        emu_size.extend(toks.values())
        emu_full.extend(replay(reqs, decode, rng, tokens=toks).values())
    # 95 % bootstrap intervals of the measured statistics (resampling requests)
    boot = {"mean_s": [], "p50_s": [], "p95_s": []}
    brng = random.Random(2)
    for _ in range(MC_DRAWS):
        smp = [brng.choice(measured) for _ in measured]
        for k, v in summary(smp).items():
            if k in boot:
                boot[k].append(v)
    ci = {
        f"{k}_ci95": f"[{pct(v, 2.5):.3f}, {pct(v, 97.5):.3f}]" for k, v in boot.items()
    }

    def inside(stats: dict) -> dict:
        out = {}
        for k in boot:
            lo, hi = pct(boot[k], 2.5), pct(boot[k], 97.5)
            out[f"{k}_in_measured_ci95"] = lo <= stats[k] <= hi
        return out

    s_tok, s_full = summary(emu_tok), summary(emu_full)
    write(
        os.path.join(OUT_DIR, "distribution.csv"),
        [
            {
                "series": "measured latency",
                "n": len(measured),
                **summary(measured),
                **ci,
            },
            {
                "series": "emulated latency, measured answer sizes",
                "n": len(emu_tok),
                **s_tok,
                **inside(s_tok),
            },
            {
                "series": "emulated latency, sampled answers",
                "n": len(emu_full),
                **s_full,
                **inside(s_full),
            },
            {
                "series": "measured answer tokens (as _s columns)",
                "n": len(reqs),
                **summary([float(r["response_tokens_est"]) for r in reqs]),
            },
            {
                "series": "sampled answer tokens (as _s columns)",
                "n": len(emu_size),
                **summary(emu_size),
            },
        ],
    )
    profile["decode"]["distribution_check"] = "calibration/distribution.csv"

    with open(PROFILE, "w") as fh:
        json.dump(profile, fh, indent=2)
        fh.write("\n")

    print("mechanism check (observed batch times through the batch loop):")
    for m in mech:
        print(
            f"  {m['rep']}: {m['batches_with_same_members']}/{m['batches']} batches same members, max error {m['max_abs_error_s']} s"
        )
    print(f"batch model: {decode}")
    print("latency models (all requests, noise-free):")
    for m in models:
        print(
            f"  {m['model']:<56} MAE {m['mae_s']} s  RMSE {m['rmse_s']} s  R2 {m['r2']}"
        )
    print("leave one run out (MAE, s):")
    for r in loro:
        print(
            f"  {r['held_out']}: emulator {r['mae_emulator_s']}  no-batching {r['mae_no_batching_s']}  constant {r['mae_constant_s']}"
        )
    print("distribution (s): measured 95% bootstrap CI; emulated in/OUT of it")
    for r in csv.DictReader(open(os.path.join(OUT_DIR, "distribution.csv"))):
        flags = " ".join(
            f"{k.split('_')[0]}:{'in' if r[k] == 'True' else 'OUT'}"
            for k in r
            if k.endswith("in_measured_ci95") and r[k]
        )
        cis = " ".join(
            f"{k.split('_')[0]}{r[k]}" for k in r if k.endswith("_s_ci95") and r[k]
        )
        print(
            f"  {r['series']:<40} mean {r['mean_s']:>8} p50 {r['p50_s']:>8} "
            f"p95 {r['p95_s']:>8} max {r['max_s']:>8}  {cis or flags}"
        )
    print(f"startup warm {warm} cold {cold}")
    print(
        f"answer: tokens/sim median {st.median(tokens_per_sim):.1f}, unscored {profile['answer']['unscored_prob']}, stop-all-running {profile['answer']['stop_all_running_prob']}"
    )
    print(
        f"gpu {busy_gpu}: in flight {inflight_mean:.1f}% idle {idle_mean:.1f}%  others max {others:.3f}%"
    )
    print(f"-> {PROFILE}")


if __name__ == "__main__":
    main()
