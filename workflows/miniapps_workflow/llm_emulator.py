"""Calibrated LLM-service emulator for the agentic MiniApps workflow.

An AsyncFlow backend that stands in for RHAPSODY's ``DragonVllmInferenceBackend``
with the *properties* measured for the real service, not its content. The numbers come
from a profile JSON built by ``llm_emulator_calibration/calibrate.py`` from the
exp2/llm-service campaign (Qwen2.5-7B-Instruct, vLLM through Dragon Inference, one SMC
H100). See ``llm_emulator_calibration/README.md`` for the derivation and its checks.

Emulated properties:

* **Service startup (EC5).** Sleeps a startup time drawn from the measured ones
  (``warm``: weights in page cache, or ``cold``), before the backend reports ready.
* **Request latency (EQ3): RHAPSODY's batching.** Mirrors the batch loop of
  RHAPSODY's ``DragonVllmInferenceBackend`` (rhapsody 142f396, ``backends/ai/vllm.py``):
  wait for a request, collect more for ``batch_wait_s`` (500 ms), send all pending
  requests (up to ``max_batch_size``) to the model as one batch, and return every
  answer of the batch when the last one is done; requests arriving meanwhile wait for
  the next batch. Only the batch's model time is calibrated:
  ``overhead_s + s_per_token * (longest answer in the batch, tokens)``, times a
  random factor ``exp(N(-sigma^2/2, sigma))`` (mean 1) for the measured run-to-run
  spread. :func:`simulate_batches` runs the same logic in virtual time for
  calibration and validation.
* **Answer size and format.** For every ``sim_<k>`` in the prompt: a filler sentence and
  ``sim_<k>: <score>``. Answer tokens = sims x tokens-per-sim (drawn from the measured
  answers), capped at ``max_tokens``; one token = 4 characters, the estimate the
  workflow uses for ``response_tokens_est``. A sim is left unscored with the measured
  probability. Stop decisions follow the measured pattern: per request, with
  probability ``stop_all_running_prob`` every sim the prompt lists as
  ``state=running`` gets score 0.0 (the real model scored all its stops 0.0), otherwise
  none does; finished sims (which cannot be stopped) score >= 0.5.
* **GPU load (optional, ``gpu: load``).** Reserves ``gpu_memory_utilization`` of GPU 0
  (from the vLLM config; *not* measured) and keeps the GPU busy for the measured
  fraction of time while at least one request is in flight (submitted, not answered).

Not emulated: the model's reasoning (answers are filler), host CPU and memory of the
vLLM processes (measured +2.7 points node CPU, +0.1 point node memory; documented only),
and anything outside the calibrated range recorded in the profile.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import threading
import time

from radical.asyncflow import NoopExecutionBackend

DEFAULT_PROFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "llm_emulator_calibration",
    "profile_qwen2.5-7b-instruct_h100_smc.json",
)
CHARS_PER_TOKEN = 4  # same estimate as the workflow's *_tokens_est fields


def load_profile(path: str | None = None) -> dict:
    with open(path or DEFAULT_PROFILE) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------
# Latency model, shared by the backend (real time) and calibration (virtual time)
# ----------------------------------------------------------------------------------
def batch_time(
    max_tokens_in_batch: float, decode: dict, rng: random.Random | None = None
) -> float:
    """Model time of one batch: overhead + per-token time x longest answer, x noise."""
    t = decode["overhead_s"] + decode["s_per_token"] * max_tokens_in_batch
    if rng is not None and decode.get("sigma", 0) > 0:
        sig = decode["sigma"]
        t *= math.exp(rng.gauss(-sig * sig / 2, sig))
    return max(t, 0.0)


def simulate_batches(
    requests: list[tuple[float, float]],
    decode: dict,
    rng: random.Random | None = None,
    batch_times: list[float] | None = None,
) -> tuple[list[float], list[dict]]:
    """Virtual-time replay of RHAPSODY's batch loop.

    ``requests = [(t_submit, answer_tokens), ...]``. Returns the latency of every
    request (input order) and the batches (start of collection, end, members).
    Without ``rng`` the batch times are the noise-free model; ``batch_times`` replaces
    the model with given times, in batch order (to check the batching rules alone).
    """
    given = iter(batch_times) if batch_times is not None else None
    wait = decode["batch_wait_s"]
    max_size = int(decode["max_batch_size"])
    order = sorted(range(len(requests)), key=lambda i: requests[i][0])
    lat = [0.0] * len(requests)
    batches = []
    free_at = -math.inf  # when the batch loop is back to waiting
    k = 0
    while k < len(order):
        # the loop wakes at the first pending request, or right after the last batch
        start = max(requests[order[k]][0], free_at)
        cutoff = start + wait
        members = []
        while (
            k < len(order)
            and requests[order[k]][0] <= cutoff
            and len(members) < max_size
        ):
            members.append(order[k])
            k += 1
        if given is not None:
            took = next(given)
        else:
            took = batch_time(max(requests[i][1] for i in members), decode, rng)
        end = cutoff + took
        for i in members:
            lat[i] = end - requests[i][0]
        batches.append({"start": start, "end": end, "members": members})
        free_at = end
    return lat, batches


# ----------------------------------------------------------------------------------
# Answers
# ----------------------------------------------------------------------------------
_RUNNING_RE = re.compile(r"sim_(\d+):\s*state=running")  # the workflow's prompt rows
_FILLER = (
    "The trajectory statistics suggest this simulation explores a region whose "
    "features differ from the ensemble average, so its value for further sampling "
    "is weighed against the cost of continuing it. "
)


def sample_answer(
    prompt: str, profile: dict, max_tokens: int, rng: random.Random
) -> str:
    """An answer with the measured size and format for the sims named in ``prompt``."""
    ans = profile["answer"]
    sims = sorted({int(s) for s in re.findall(r"sim_(\d+)", prompt)})
    if not sims:
        return "no simulations"
    tokens = len(sims) * rng.choice(ans["tokens_per_sim"])
    tokens = int(min(round(tokens), max_tokens))
    budget = tokens * CHARS_PER_TOKEN
    running = {int(k) for k in _RUNNING_RE.findall(prompt)}
    stop_running = bool(running) and rng.random() < ans["stop_all_running_prob"]
    lines = []
    for k in sims:
        if rng.random() < ans["unscored_prob"]:
            lines.append(f"Simulation {k}: no clear signal.")
            continue
        if stop_running and k in running:
            score = ans["stop_score"]
        else:
            score = rng.uniform(0.5, 1.0)
        lines.append(f"sim_{k}: {score:.2f}")
    # Filler before each line, sized so the whole answer is ``budget`` characters:
    # n reasons + n "\n" after them + the lines + (n - 1) "\n" between the blocks.
    # Only the filler shrinks; score lines are never cut.
    n = len(lines)
    per = max((budget - sum(len(x) for x in lines) - 2 * n + 1) // n, 0)
    reason = (_FILLER * (per // len(_FILLER) + 1))[:per]
    text = "\n".join(f"{reason}\n{line}" if per else line for line in lines)
    return text


# ----------------------------------------------------------------------------------
# Optional GPU load
# ----------------------------------------------------------------------------------
class _GpuLoad:
    """Reserve GPU memory and keep the GPU busy a fraction of time while requested."""

    def __init__(
        self, busy_fraction: float, memory_fraction: float, window_s: float = 0.1
    ):
        import torch  # only for gpu: load

        self.torch = torch
        self.busy_fraction = busy_fraction
        self.window_s = window_s
        dev = torch.device("cuda", 0)
        free, total = torch.cuda.mem_get_info(dev)
        n_bytes = int(min(memory_fraction * total, free * 0.95))
        self.reserved = torch.empty(n_bytes, dtype=torch.uint8, device=dev)
        self.a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        self.reserved_bytes = n_bytes
        self.active = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="llm-emulator-gpu", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        torch = self.torch
        while not self.stop.is_set():
            if not self.active.wait(timeout=0.05):
                continue
            t_end = time.perf_counter() + self.busy_fraction * self.window_s
            while time.perf_counter() < t_end:
                self.a @ self.a  # noqa: B018 - the kernel is the point
                torch.cuda.synchronize()
            rest = self.window_s * (1 - self.busy_fraction)
            if rest > 0:
                time.sleep(rest)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        del self.reserved, self.a
        self.torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------
# AsyncFlow backend
# ----------------------------------------------------------------------------------
class EmulatedLLMBackend(NoopExecutionBackend):
    """LLM-service emulator with the measured startup, latency, answer and GPU profile.

    Create with :meth:`create` (awaits the emulated startup).
    """

    def __init__(
        self,
        name: str = "llm",
        profile: dict | None = None,
        max_tokens: int = 512,
        seed: int = 0,
        gpu: str = "none",
        gpu_memory_utilization: float | None = None,
    ):
        super().__init__(name=name)
        self.profile = profile or load_profile()
        self.decode = self.profile["decode"]
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)
        self._pending: list[dict] = []
        self._new_request = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._gpu = None
        self._inflight = 0  # submitted and not yet answered (the GPU-busy window)
        self.gpu_mode = gpu
        g = self.profile["gpu"]
        self._gpu_memory_fraction = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else g["memory_fraction_config"]
        )

    @classmethod
    async def create(
        cls,
        name: str = "llm",
        profile_path: str | None = None,
        startup: str | float = "warm",
        startup_scale: float = 1.0,
        **kw,
    ) -> EmulatedLLMBackend:
        profile = load_profile(profile_path)
        self = cls(name=name, profile=profile, **kw)
        if isinstance(startup, str):
            startup_s = self.rng.choice(profile["service"]["startup_s"][startup])
        else:
            startup_s = float(startup)
        if self.gpu_mode == "load":
            self._gpu = await asyncio.to_thread(
                _GpuLoad,
                profile["gpu"]["busy_percent_inflight"] / 100,
                self._gpu_memory_fraction,
            )
        self.startup_s = startup_s * startup_scale
        await asyncio.sleep(self.startup_s)
        self._loop_task = asyncio.get_running_loop().create_task(self._batch_loop())
        return self

    async def submit_tasks(self, tasks: list) -> None:
        for task in tasks:
            self._callback_func(task, "RUNNING")
            task["_emu_answer"] = sample_answer(
                task.get("prompt", ""), self.profile, self.max_tokens, self.rng
            )
            self._pending.append(task)
            self._inflight += 1
        self._set_gpu_active()
        self._new_request.set()

    async def _batch_loop(self) -> None:
        wait, max_size = self.decode["batch_wait_s"], int(self.decode["max_batch_size"])
        while True:
            await self._new_request.wait()
            self._new_request.clear()
            await asyncio.sleep(wait)  # collect more requests
            batch, self._pending = self._pending[:max_size], self._pending[max_size:]
            if self._pending:
                self._new_request.set()
            if not batch:
                continue
            longest = max(len(t["_emu_answer"]) for t in batch) / CHARS_PER_TOKEN
            await asyncio.sleep(batch_time(longest, self.decode, self.rng))
            for task in batch:  # the whole batch returns together
                text = task.pop("_emu_answer")
                task["stdout"] = text
                task["return_value"] = text
                self._inflight -= 1
                self._callback_func(task, "DONE")
            self._set_gpu_active()

    def _set_gpu_active(self) -> None:
        if self._gpu is not None:
            if self._inflight > 0:
                self._gpu.active.set()
            else:
                self._gpu.active.clear()

    async def shutdown(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        if self._gpu is not None:
            self._gpu.close()
        await super().shutdown()
