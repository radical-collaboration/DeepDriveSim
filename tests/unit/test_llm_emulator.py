"""Tests of the calibrated LLM-service emulator (miniapps_workflow/llm_emulator.py).

Besides behaviour, these tests tie the shipped profile to its data:
* the calibration inputs match the sha256 recorded in inputs/SOURCES.json;
* refitting from those inputs gives exactly the parameters in the profile;
* the batch loop reproduces the measured latencies when given the observed batch times.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import statistics as st
import sys

import pytest

from workflows.miniapps_workflow import llm_emulator as emu

CAL = os.path.join(os.path.dirname(emu.__file__), "llm_emulator_calibration")
DECODE = {
    "batch_wait_s": 0.5,
    "max_batch_size": 1024,
    "overhead_s": 0.2,
    "s_per_token": 0.01,
    "sigma": 0.0,
}


# --- batching rules (virtual time) ---
def test_single_request_latency():
    lat, batches = emu.simulate_batches([(10.0, 100)], DECODE)
    assert lat == [pytest.approx(0.5 + 0.2 + 100 * 0.01)]
    assert len(batches) == 1


def test_requests_in_one_window_return_together_at_longest_answer():
    lat, batches = emu.simulate_batches([(0.0, 50), (0.3, 300)], DECODE)
    end = 0.5 + 0.2 + 300 * 0.01  # window from the first request; longest answer
    assert lat == [pytest.approx(end), pytest.approx(end - 0.3)]
    assert len(batches) == 1


def test_request_during_a_batch_waits_for_the_next_one():
    # first batch: collect 0..0.5, model 0.2 + 1.0 -> ends 1.7; second request at 1.0
    lat, batches = emu.simulate_batches([(0.0, 100), (1.0, 100)], DECODE)
    assert len(batches) == 2
    assert lat[0] == pytest.approx(1.7)
    # the loop is free at 1.7, collects until 2.2, model 1.2 -> 3.4
    assert lat[1] == pytest.approx(3.4 - 1.0)


def test_max_batch_size():
    dec = {**DECODE, "max_batch_size": 2}
    _, batches = emu.simulate_batches([(0.0, 10)] * 5, dec)
    assert [len(b["members"]) for b in batches] == [2, 2, 1]


def test_given_batch_times_override_the_model():
    lat, _ = emu.simulate_batches([(0.0, 999)], DECODE, batch_times=[3.0])
    assert lat == [pytest.approx(3.5)]


def test_batch_time_noise_has_mean_one():
    dec = {**DECODE, "sigma": 0.355}
    rng = random.Random(0)
    base = emu.batch_time(200, DECODE)
    draws = [emu.batch_time(200, dec, rng) for _ in range(40000)]
    assert st.fmean(draws) / base == pytest.approx(1.0, abs=0.01)


# --- answers ---
def test_answer_format_and_size():
    profile = emu.load_profile()
    rng = random.Random(0)
    prompt = " ".join(f"sim_{k}: rmsd=..." for k in range(8))
    sizes, scored = [], 0
    for _ in range(300):
        ans = emu.sample_answer(prompt, profile, 512, rng)
        sizes.append(len(ans) / emu.CHARS_PER_TOKEN)
        scored += len(re.findall(r"^sim_(\d+): ([0-9.]+)$", ans, re.M))
        assert len(ans) <= 512 * emu.CHARS_PER_TOKEN
    per_sim = st.fmean(sizes) / 8
    assert per_sim == pytest.approx(
        st.fmean(profile["answer"]["tokens_per_sim"]), rel=0.1
    )
    assert scored / (300 * 8) == pytest.approx(
        1 - profile["answer"]["unscored_prob"], abs=0.02
    )


def test_stop_rule_all_running_or_none():
    profile = emu.load_profile()
    q = profile["answer"]["stop_all_running_prob"]
    rng = random.Random(1)
    prompt = (
        "sim_0: state=finished output_bytes=10 age_s=1.0\n"
        "sim_1: state=running output_bytes=0 age_s=0.0\n"
        "sim_2: state=running output_bytes=0 age_s=0.0\n"
        "sim_3: state=finished output_bytes=10 age_s=1.0\n"
    )
    stopped_all = 0
    n = 2000
    for _ in range(n):
        scores = {
            int(k): float(v)
            for k, v in re.findall(
                r"^sim_(\d+): ([0-9.]+)$",
                emu.sample_answer(prompt, profile, 512, rng),
                re.M,
            )
        }
        low = {k for k, v in scores.items() if v < 0.5}
        assert low <= {1, 2}  # finished sims are never scored low
        running_scored = {1, 2} & set(scores)
        assert low in (set(), running_scored)  # all running or none, never some
        stopped_all += bool(low)
    assert stopped_all / n == pytest.approx(q, abs=0.04)


def test_answer_without_sims():
    assert emu.sample_answer(
        "nothing here", emu.load_profile(), 512, random.Random(0)
    ) == ("no simulations")


# --- the backend (real time) ---
def test_backend_batches_in_real_time():
    profile = emu.load_profile()
    profile = {
        **profile,
        "decode": {
            **DECODE,
            "batch_wait_s": 0.05,
            "s_per_token": 1e-4,
            "overhead_s": 0.0,
        },
    }

    async def run():
        be = emu.EmulatedLLMBackend(profile=profile, max_tokens=64)
        done = {}
        be.register_callback(
            lambda task, state: (
                done.setdefault(task["uid"], time_now()) if state == "DONE" else None
            )
        )
        be._loop_task = asyncio.get_running_loop().create_task(be._batch_loop())
        t = [{"uid": f"t{i}", "prompt": "sim_0 sim_1"} for i in range(3)]
        await be.submit_tasks(t[:2])
        await asyncio.sleep(0.02)
        await be.submit_tasks(t[2:])  # same 50 ms window
        await asyncio.sleep(0.3)
        await be.shutdown()
        return t, done

    def time_now():
        import time

        return time.perf_counter()

    tasks, done = asyncio.run(run())
    assert set(done) == {"t0", "t1", "t2"}
    assert (
        max(done.values()) - min(done.values()) < 0.005
    )  # one batch, returned together
    assert all(
        re.search(r"sim_[01]", t["stdout"]) or "no clear signal" in t["stdout"]
        for t in tasks
    )


# --- provenance and reproducibility of the shipped profile ---
def test_inputs_match_recorded_checksums():
    src = json.load(open(os.path.join(CAL, "inputs", "SOURCES.json")))
    for name, sha in src["sha256"].items():
        with open(os.path.join(CAL, "inputs", name), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha, name
    assert emu.load_profile()["source"] == src


def test_refit_reproduces_the_profile_and_the_measurements():
    sys.path.insert(0, CAL)
    try:
        import calibrate
    finally:
        sys.path.remove(CAL)
    reqs = [
        r for r in calibrate.read("llm_requests.csv") if r["label"] == calibrate.LABEL
    ]
    batches = [
        b
        for rs in calibrate.runs_of(reqs).values()
        for b in calibrate.rebuild_batches(rs)
    ]
    fitted = calibrate.fit_batch_model(batches)
    profile = emu.load_profile()["decode"]
    for k in ("batch_wait_s", "max_batch_size", "overhead_s", "s_per_token", "sigma"):
        assert profile[k] == fitted[k], k
    # the batch loop, given the observed batch times, reproduces every latency
    for rs in calibrate.runs_of(reqs).values():
        bs = calibrate.rebuild_batches(rs)
        lat, sim = emu.simulate_batches(
            [(float(r["t_submit"]), float(r["response_tokens_est"])) for r in rs],
            profile,
            batch_times=[b["model_time_s"] for b in bs],
        )
        assert len(sim) == len(bs)
        for r, v in zip(rs, lat):
            assert v == pytest.approx(float(r["latency_ms"]) / 1000, abs=0.01)
