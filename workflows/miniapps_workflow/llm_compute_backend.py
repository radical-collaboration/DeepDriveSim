"""AsyncFlow steering backend driven by actual synthetic inference computation.

Unlike ``llm_emulator.py``, no H100-calibrated wait or independent load generator
is used. Workload dimensions and device are explicit. Replies are seeded
pseudo-random scores, not scientific reasoning. Unequal prompts are padded to the
longest in a batch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

from radical.asyncflow import NoopExecutionBackend

from workflows.miniapps_workflow.inference_compute import (
    FIDELITY,
    InferenceSpec,
    InferenceWorkload,
)
from workflows.miniapps_workflow.inference_profile import (
    load_profile,
    profile_sha256,
    spec_from_profile,
)

#: Same rough sizing convention as the existing steering interface.
CHARS_PER_TOKEN = 4

_CONTEXT_RE = re.compile(
    r"^Steering context: replica=(\S+) iteration=(\d+)$", re.MULTILINE
)
_SIM_RE = re.compile(r"sim_(\d+)")
_MANTISSA_BITS = 53  # a float64 holds 53 bits: scores are uniform on [0, 1)


def seeded_steering_scores(prompt: str, seed: int) -> dict[int, float]:
    """Return a uniform pseudo-random score in ``[0, 1)`` for every sim in ``prompt``.

    Scores are keyed by ``(seed, replica, iteration, simulation)``, taken from the
    prompt's ``Steering context`` line. They do not depend on timing, device,
    batch membership or request order. This is a systems mini-app policy, not
    scientific reasoning.
    """
    context = _CONTEXT_RE.search(prompt)
    replica, iteration = context.groups() if context else ("", "0")
    scores = {}
    for sim in sorted({int(k) for k in _SIM_RE.findall(prompt)}):
        key = json.dumps([seed, replica, int(iteration), sim], separators=(",", ":"))
        digest = hashlib.sha256(key.encode()).digest()
        bits = int.from_bytes(digest[:8], "big") >> (64 - _MANTISSA_BITS)
        scores[sim] = bits / 2**_MANTISSA_BITS
    return scores


def format_scores(scores: dict[int, float]) -> str:
    """Render scores as the ``sim_<id>: <score>`` lines the workflow parses."""
    lines = [f"sim_{sim}: {score:.8f}" for sim, score in scores.items()]
    return "\n".join(lines) or "no simulations"


def _check_settings(
    spec: InferenceSpec, new_tokens: object, max_pending: object, steering_seed: object
) -> None:
    """Raise ``ValueError`` for unusable backend settings.

    ``type(...) is int`` is deliberate: ``bool`` subclasses ``int`` and must not
    be accepted as a count.
    """
    if type(new_tokens) is not int or not 1 <= new_tokens <= spec.max_new_tokens:
        raise ValueError("new_tokens exceeds the workload bounds")
    if type(max_pending) is not int or max_pending <= 0:
        raise ValueError("max_pending must be a positive integer")
    if type(steering_seed) is not int or steering_seed < 0:
        raise ValueError("steering_seed must be a nonnegative integer")


class ComputeLLMBackend(NoopExecutionBackend):
    """Inference backend that executes a synthetic decoder for every request.

    One worker owns the tensor workload and runs batches on a thread, so tensor
    execution never blocks the event loop. Requests already queued are grouped up
    to ``spec.max_batch_size``. The queue is bounded, and an oversize request
    fails instead of being truncated.

    Construct it inside a running event loop, normally through :meth:`create`.
    """

    def __init__(
        self,
        workload: InferenceWorkload,
        new_tokens: int = 16,
        max_pending: int = 64,
        name: str = "llm",
        steering_seed: int = 0,
        profile: dict[str, str] | None = None,
    ) -> None:
        """Wrap an existing workload; call :meth:`start` to begin serving.

        ``profile`` is the name and hash of the inference profile the workload's
        spec came from, or None for hand-chosen dimensions; it is only recorded.
        """
        super().__init__(name=name)
        _check_settings(workload.spec, new_tokens, max_pending, steering_seed)
        self.workload: InferenceWorkload | None = workload
        self.spec = workload.spec
        self.steering_seed = steering_seed
        self.new_tokens = new_tokens
        # An item is (task, estimated prompt tokens); None stops the worker.
        self._queue: asyncio.Queue[tuple[dict, int] | None] = asyncio.Queue(
            maxsize=max_pending
        )
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._description = {
            "spec": asdict(self.spec),
            "device": str(workload.device),
            "new_tokens": new_tokens,
            "max_pending": max_pending,
            "profile": profile,
            "fidelity": FIDELITY,
            "batching": "opportunistic_pad_to_longest",
            "prompt_size": "estimated_characters_divided_by_four_not_tokenized",
            "steering_policy": "seeded_hash_uniform_after_computation",
            "steering_seed": steering_seed,
        }

    @classmethod
    async def create(
        cls, config: dict[str, Any], name: str = "llm"
    ) -> ComputeLLMBackend:
        """Build the workload described by ``config`` and start serving.

        Every setting is validated before any tensor is allocated.

        Args:
            config: The ``llm.compute`` section: ``spec``, ``device``,
                ``new_tokens``, ``max_pending`` and ``steering_seed``. With
                ``profile`` (a packaged profile name or a path), the architecture
                comes from that inference profile and ``spec`` holds only the
                per-run limits.
            name: Backend name used by AsyncFlow to route prompt tasks.

        Raises:
            ValueError: A setting is out of bounds.
        """
        recorded = None
        if "profile" in config:
            profile = load_profile(config["profile"])
            spec = spec_from_profile(profile, **config.get("spec", {}))
            recorded = {"name": profile["name"], "sha256": profile_sha256(profile)}
        else:
            spec = InferenceSpec(**config.get("spec", {}))
        new_tokens = config.get("new_tokens", 16)
        max_pending = config.get("max_pending", 64)
        steering_seed = config.get("steering_seed", 0)
        _check_settings(spec, new_tokens, max_pending, steering_seed)
        workload = await asyncio.to_thread(
            InferenceWorkload, spec, config.get("device", "cpu")
        )
        backend = cls(
            workload,
            new_tokens=new_tokens,
            max_pending=max_pending,
            name=name,
            steering_seed=steering_seed,
            profile=recorded,
        )
        backend.start()
        return backend

    def start(self) -> None:
        """Start the worker that serves queued requests; safe to call once."""
        if self._worker is None and not self._closed:
            self._worker = asyncio.create_task(self._serve())

    def description(self) -> dict[str, Any]:
        """Return the effective configuration, recorded with each run."""
        return dict(self._description)

    def state(self) -> str:
        """Return ``INITIALIZED``, ``RUNNING`` or ``SHUTDOWN``."""
        if self._closed:
            return "SHUTDOWN"
        return "RUNNING" if self._worker is not None else "INITIALIZED"

    def _fail(self, task: dict, error: Exception) -> None:
        """Report ``task`` as failed with ``error`` through the state callback."""
        task["exception"] = error
        task["stderr"] = str(error)
        task["exit_code"] = 1
        self._callback_func(task, "FAILED")

    async def submit_tasks(self, tasks: list[dict]) -> None:
        """Queue prompt tasks; reject any that cannot be served as submitted."""
        for task in tasks:
            self._callback_func(task, "RUNNING")
            if self._closed:
                self._fail(task, RuntimeError("compute inference backend is closed"))
                continue
            prompt_chars = len(task.get("prompt", ""))
            # Round up: a partial token still occupies a position.
            prompt_tokens = max(
                (prompt_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN, 1
            )
            # Never silently truncate a request to make it fit the compute budget.
            if prompt_tokens > self.spec.max_prompt_tokens:
                self._fail(task, ValueError("prompt exceeds max_prompt_tokens"))
                continue
            try:
                self._queue.put_nowait((task, prompt_tokens))
            except asyncio.QueueFull:
                self._fail(task, RuntimeError("compute inference queue is full"))

    async def _serve(self) -> None:
        """Serve queued requests in batches until the stop sentinel arrives."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if self._closed or self.workload is None:
                self._fail(item[0], RuntimeError("backend shut down before execution"))
                continue
            batch = [item]
            while len(batch) < self.spec.max_batch_size:
                try:
                    queued = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if queued is None:
                    # Defensive: shutdown() marks the backend closed before it
                    # queues the sentinel, so this is not expected. Keep the
                    # sentinel for the next iteration rather than batch it.
                    self._queue.put_nowait(None)
                    break
                batch.append(queued)
            await self._run_batch(self.workload, batch)

    async def _run_batch(
        self, workload: InferenceWorkload, batch: list[tuple[dict, int]]
    ) -> None:
        """Execute one padded batch and complete or fail each of its tasks."""
        longest = max(prompt_tokens for _, prompt_tokens in batch)
        try:
            result = await asyncio.to_thread(
                workload.run, len(batch), longest, self.new_tokens
            )
        except Exception as error:
            # Deliberately broad: any workload failure must become a failed
            # request. A narrower clause could kill the worker and hang callers.
            for task, _ in batch:
                self._fail(task, error)
            return
        compute = {key: value for key, value in result.items() if key != "token_ids"}
        for task, prompt_tokens in batch:
            # Scores are produced only AFTER the tensor workload has run, and
            # ignore its token content, so rounding or device differences cannot
            # alter the policy. The computation itself is unchanged.
            scores = seeded_steering_scores(task.get("prompt", ""), self.steering_seed)
            text = format_scores(scores)
            task["stdout"] = task["return_value"] = text
            task["inference_compute"] = {
                **compute,
                "original_prompt_tokens": prompt_tokens,
            }
            task["exit_code"] = 0
            self._callback_func(task, "DONE")

    async def shutdown(self) -> None:
        """Finish the active batch, fail queued requests and release the workload.

        A running tensor call cannot safely be interrupted, so shutdown waits for
        it rather than orphaning a compute thread. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None
        self.workload = None
        await super().shutdown()
