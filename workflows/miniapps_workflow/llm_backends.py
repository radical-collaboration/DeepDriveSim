"""LLM backends for the agentic MiniApps workflow.

The agentic workflow sends its steering prompt as an AsyncFlow ``prompt_task`` routed to
a named LLM backend, so every LLM request is an ordinary AsyncFlow task (uid, workflow
scope, lifecycle telemetry).

Backends (chosen explicitly in the config, ``llm.backend``):

* ``vllm``: RHAPSODY's ``DragonVllmInferenceBackend`` (Dragon Inference + vLLM on GPUs).
  This is the intended backend.
* ``standin``: :class:`StandInLLMBackend`, a CPU-only stand-in for testing the workflow
  without a GPU. It answers in the same format after a configurable delay; it is not an
  LLM and its numbers must not be reported as LLM measurements.
* ``emulated``: :class:`llm_emulator.EmulatedLLMBackend`, an emulator calibrated on the
  measured vLLM service (startup, RHAPSODY batching and batch times, answer size and
  format, optional GPU load). See ``llm_emulator_calibration/README.md``. Its numbers
  are emulated, not measured; runs record the profile and its sha256.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time

from radical.asyncflow import NoopExecutionBackend


class PromptTaskBridge:
    """Copy an AI backend's ``task["response"]`` into ``task["stdout"]`` on DONE.

    AsyncFlow resolves a prompt task's future with ``task["stdout"]``, but RHAPSODY's
    ``DragonVllmInferenceBackend`` stores the model output only in ``task["response"]``
    (its own ``AITask`` API reads that). Without this bridge AsyncFlow's run loop raises
    ``KeyError: 'stdout'`` and the prompt task never returns (radical.asyncflow d97585e,
    rhapsody 142f396). Remove once fixed upstream.
    """

    def register_callback(self, func):
        def bridged(task, state):
            if state == "DONE" and task.get("stdout") is None and "response" in task:
                response = task["response"]
                if isinstance(response, list) and len(response) == 1:
                    response = response[0]
                text = response if isinstance(response, str) else str(response)
                task["stdout"] = text
                task["return_value"] = text
            return func(task, state)

        return super().register_callback(bridged)


class StandInLLMBackend(NoopExecutionBackend):
    """CPU-only stand-in for an LLM inference backend (testing without a GPU).

    Waits ``latency_s + output_tokens * per_token_s`` per request, then answers with one
    ``sim_<k>: <score>`` line for every ``sim_<k>`` mentioned in the prompt.
    """

    def __init__(
        self,
        name: str = "llm",
        latency_s: float = 0.2,
        per_token_s: float = 0.002,
        output_tokens: int = 128,
        seed: int = 0,
    ):
        super().__init__(name=name)
        self.latency_s = latency_s
        self.per_token_s = per_token_s
        self.output_tokens = output_tokens
        self._rng = random.Random(seed)
        self._inflight: set[asyncio.Task] = set()

    async def submit_tasks(self, tasks: list) -> None:
        for task in tasks:
            t = asyncio.get_running_loop().create_task(self._answer(task))
            self._inflight.add(t)
            t.add_done_callback(self._inflight.discard)

    async def _answer(self, task: dict) -> None:
        self._callback_func(task, "RUNNING")
        await asyncio.sleep(self.latency_s + self.output_tokens * self.per_token_s)
        sims = sorted(
            {int(s) for s in re.findall(r"sim_(\d+)", task.get("prompt", ""))}
        )
        text = "\n".join(f"sim_{k}: {self._rng.random():.3f}" for k in sims)
        task["stdout"] = text or "no simulations"
        task["return_value"] = task["stdout"]
        self._callback_func(task, "DONE")

    async def shutdown(self) -> None:
        for t in list(self._inflight):
            t.cancel()
        await super().shutdown()


async def make_llm_backend(llm_cfg: dict, name: str = "llm"):
    """Build and initialize the LLM backend described by the ``llm:`` config section.

    Returns ``(backend, info)``; ``info`` records the service identity and its start-up
    timestamps (``t_init_start``, ``t_ready``) for the service metrics.
    """
    kind = llm_cfg.get("backend", "vllm")
    info = {"service": name, "backend": kind, "model": llm_cfg.get("model", "")}
    info["t_init_start"] = time.time()
    if kind == "vllm":
        from rhapsody.backends import DragonVllmInferenceBackend
        from rhapsody.backends.ai.config import (
            BatchingConfig,
            HardwareConfig,
            ModelConfig,
        )

        class BridgedDragonVllmInferenceBackend(
            PromptTaskBridge, DragonVllmInferenceBackend
        ):
            pass

        backend = BridgedDragonVllmInferenceBackend(
            name=name,
            model=ModelConfig(
                model_name=llm_cfg["model"],
                hf_token=llm_cfg.get("hf_token", ""),
                tp_size=int(llm_cfg.get("tp_size", 1)),
                max_tokens=int(llm_cfg.get("max_tokens", 128)),
                max_model_len=int(llm_cfg.get("max_model_len", 8192)),
                temperature=float(llm_cfg.get("temperature", 0.5)),
                ignore_eos=bool(llm_cfg.get("ignore_eos", True)),
                gpu_memory_utilization=float(
                    llm_cfg.get("gpu_memory_utilization", 0.85)
                ),
            ),
            hardware=HardwareConfig(
                num_nodes=int(llm_cfg.get("num_nodes", 1)),
                num_gpus=int(llm_cfg.get("num_gpus", 1)),
                node_offset=int(llm_cfg.get("node_offset", 0)),
            ),
            batching=BatchingConfig(enabled=True, batch_type="pre-batch"),
            use_service=bool(llm_cfg.get("http_service", False)),
            port=int(llm_cfg.get("port", 8000)),
        )
        info["num_gpus"] = int(llm_cfg.get("num_gpus", 1))
        backend = await backend  # initialize(): load the model, start the pipeline
    elif kind == "emulated":
        import hashlib

        from workflows.miniapps_workflow.llm_emulator import (
            DEFAULT_PROFILE,
            EmulatedLLMBackend,
        )

        profile_path = llm_cfg.get("emulator_profile") or DEFAULT_PROFILE
        startup = llm_cfg.get("emulator_startup", "warm")  # warm | cold | seconds
        gpu = llm_cfg.get("emulator_gpu", "none")  # none | load
        backend = await EmulatedLLMBackend.create(
            name=name,
            profile_path=profile_path,
            startup=startup if startup in ("warm", "cold") else float(startup),
            startup_scale=float(llm_cfg.get("emulator_startup_scale", 1.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 512)),
            seed=int(llm_cfg.get("emulator_seed", 0)),
            gpu=gpu,
            gpu_memory_utilization=llm_cfg.get("gpu_memory_utilization"),
        )
        with open(profile_path, "rb") as fh:
            info["emulator_profile_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        info["emulator_profile"] = os.path.basename(profile_path)
        info["model"] = "emulated:" + backend.profile["description"].split(",")[0]
        info["num_gpus"] = 1 if gpu == "load" else 0
    elif kind == "standin":
        backend = StandInLLMBackend(
            name=name,
            latency_s=float(llm_cfg.get("standin_latency_s", 0.2)),
            per_token_s=float(llm_cfg.get("standin_per_token_s", 0.002)),
            output_tokens=int(llm_cfg.get("max_tokens", 128)),
        )
        info["num_gpus"] = 0
    else:
        raise ValueError(
            f"llm.backend={kind!r}: expected 'vllm', 'emulated' or 'standin'"
        )
    info["t_ready"] = time.time()
    return backend, info
