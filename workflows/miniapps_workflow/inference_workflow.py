"""Inference-only steering loop; the stock training-inclusive baseline is untouched."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

from workflows.miniapps_workflow.agentic_workflow import AgenticMiniAppsWorkflow

#: Simulation kernel sizes that an inference-only run may override.
_SIMULATION_SIZES = ("mat_size", "read_size", "write_size")
_SIMULATION_KEYS = frozenset((*_SIMULATION_SIZES, "steps_by_sim"))


def _is_int(value: object) -> bool:
    """True for a real ``int``; ``bool`` is rejected although it subclasses int."""
    return type(value) is int


def _validated_interval(cfg: dict[str, Any]) -> float:
    """Return ``agentic.steering_interval_s``; raise ``ValueError`` if unusable."""
    interval = (cfg.get("agentic", {}) or {}).get("steering_interval_s", 0.1)
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("steering_interval_s must be finite and positive")
    return float(interval)


def _validated_simulation(cfg: dict[str, Any]) -> tuple[dict[str, int], dict[int, int]]:
    """Return the kernel size overrides and the per-simulation step counts.

    Raises:
        ValueError: The ``simulation`` section has an unknown key or a value
            that is not a positive integer.
    """
    simulation = cfg.get("simulation", {}) or {}
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be a mapping")
    if set(simulation) - _SIMULATION_KEYS:
        raise ValueError("unsupported simulation setting")
    sizes = {}
    for key in _SIMULATION_SIZES:
        if key not in simulation:
            continue
        if not _is_int(simulation[key]) or simulation[key] <= 0:
            raise ValueError(f"simulation.{key} must be a positive integer")
        sizes[key] = simulation[key]
    steps = simulation.get("steps_by_sim", {})
    if not isinstance(steps, dict) or any(
        not str(sim).isdigit() or not _is_int(count) or count <= 0
        for sim, count in steps.items()
    ):
        raise ValueError("steps_by_sim must map simulation IDs to positive counts")
    return sizes, {int(sim): count for sim, count in steps.items()}


class InferenceOnlyMiniAppsWorkflow(AgenticMiniAppsWorkflow):
    """Simulations -> observed outputs -> inference -> cancel/continue -> repeat.

    No training task is registered or submitted. Steering begins once a
    simulation has finished and left its output, and repeats at a configurable
    interval while simulations remain. The interval is workflow polling, not
    inference latency.
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Validate the inference-only settings, then build the workflow.

        Raises:
            ValueError: A setting is missing a usable value.
        """
        cfg = config or {}
        interval = _validated_interval(cfg)
        if not _is_int(cfg.get("max_sim_batch", 4)) or cfg.get("max_sim_batch", 4) <= 0:
            raise ValueError("max_sim_batch must be a positive integer")
        if (
            not _is_int(cfg.get("total_num_sim", 25))
            or cfg.get("total_num_sim", 25) < 0
        ):
            raise ValueError("total_num_sim must be a nonnegative integer")
        # Set before super().__init__(), which calls register_tasks().
        self.simulation_sizes, self.steps_by_sim = _validated_simulation(cfg)
        super().__init__(config=cfg, **kwargs)
        self.steering_interval_s = interval
        self.retrain_model = False
        self.free_resources_for_train = False
        self.training_cores = 0
        self.sim_batch_size = self.max_sim_batch
        self.finished_sim_ids: list[int] = []
        self.stopped_sim_ids: list[int] = []
        self.keep_outputs = bool(
            (cfg.get("agentic", {}) or {}).get("keep_outputs", False)
        )
        self._simulation_failure: BaseException | None = None

    def register_tasks(self) -> None:
        """Register only the simulation and steering tasks; no training."""
        if self.engine == "dragon":
            desc = {"process_templates": [[1]], "cores_per_rank": 1, "gpus_per_rank": 0}
        else:
            desc = {"cpus": 1}

        # ``task_description`` is read by AsyncFlow from the signature default.
        @self.flow.executable_task(capture_stdio=True)
        async def simulation(task_description: dict = desc, **kwargs: Any) -> str:
            return self._sim_cmd(kwargs["sim_inputs"]["sim_idx"])

        @self.flow.prompt_task(backend=self.llm_backend_name)
        async def llm_steering(prompt: str) -> str:
            return prompt

        self.simulation = simulation
        self.llm_steering = llm_steering

    def _sim_cmd(self, sim_idx: int) -> str:
        """Return the stock simulation command plus any validated overrides."""
        command = super()._sim_cmd(sim_idx)
        # argparse keeps the last occurrence, so the stock command is preserved
        # when nothing is overridden. Every appended value is a validated int.
        if sim_idx in self.steps_by_sim:
            command += f" --num_step {self.steps_by_sim[sim_idx]}"
        for key, value in self.simulation_sizes.items():
            command += f" --{key} {value}"
        return command

    def _build_prompt(self, data: list[dict]) -> str:
        """Prefix the prompt with the replica and iteration used for scoring."""
        context = f"Steering context: replica={self.name} iteration={self.iteration}\n"
        return context + super()._build_prompt(data)

    def _on_sim_done(self, task: asyncio.Task, sim_idx: int) -> None:
        """Record how a simulation ended before the manager unregisters it."""
        if task.cancelled():
            if sim_idx in self._perm_cancelled and not self.shutting_down.is_set():
                self.stopped_sim_ids.append(sim_idx)
        elif task.exception() is not None:
            self._simulation_failure = task.exception()
        else:
            self.finished_sim_ids.append(sim_idx)
        super()._on_sim_done(task, sim_idx)

    async def close(self) -> None:
        """Emit the request summary; remove outputs unless ``keep_outputs``."""
        if self.keep_outputs:
            self._emit_summary()
        else:
            await super().close()

    def _collect_data(self) -> list[dict]:
        """Describe each simulation, labelling it by task state, not by files."""
        data = super()._collect_data()
        for row in data:
            # File existence alone does not establish that a task has finished.
            running = row["sim"] in self.registered_sims
            row["state"] = "running" if running else "finished"
        return data

    async def cancel_sims(self) -> None:
        """Cancel low-scoring simulations that are still actually running."""
        # A simulation can finish just before its done-callback unregisters it.
        # Do not count such a task again as a prediction-based cancellation.
        self.sim_predictions = {
            sim: score
            for sim, score in self.sim_predictions.items()
            if sim in self.registered_sims and not self.registered_sims[sim].done()
        }
        await super().cancel_sims()

    def _has_finished_output(self) -> bool:
        """True once a successfully finished simulation has left its output.

        A file can appear before its producer closes it, so a visible path is
        not enough: the producer must have completed.
        """
        root = Path(self.sim_output_dir, f"phase{self.phase}")
        return any(
            (root / f"data_0_{sim}.h5").is_file() for sim in self.finished_sim_ids
        )

    async def _steer_once(self) -> None:
        """Run one inference request and apply its cancellation decisions.

        Raises:
            RuntimeError: The inference request failed.
        """
        self.iteration += 1
        await self.evaluate_simulations()
        latest = self._requests[-1]
        if latest["status"] != "completed":
            raise RuntimeError(latest["error"])
        await self.cancel_sims()
        await self.finalize_results()

    async def start(self) -> None:
        """Run the inference-only loop until every simulation is accounted for.

        Raises:
            RuntimeError: A simulation or an inference request failed, or the
                simulation submitter stopped unexpectedly.
        """

        @self.flow.block
        async def run_workflow() -> None:
            await self.init_sim_queue()
            submitter = asyncio.create_task(self.submit_sims())
            try:
                while self.run_workflow:
                    if self._simulation_failure is not None:
                        raise RuntimeError(
                            "simulation failed"
                        ) from self._simulation_failure
                    if submitter.done():
                        await submitter  # propagate submitter errors, do not hang
                        raise RuntimeError("simulation submitter stopped unexpectedly")
                    await self.finalize_results()
                    if not self.run_workflow:
                        break
                    if self._has_finished_output():
                        await self._steer_once()
                    if self.run_workflow:
                        await asyncio.sleep(self.steering_interval_s)
            finally:
                self.shutting_down.set()
                submitter.cancel()
                await asyncio.gather(submitter, return_exceptions=True)
                remaining = list(self.registered_sims.items())
                # Final cleanup is not a resource-freeing pause: never requeue it.
                self._perm_cancelled.update(sim for sim, _ in remaining)
                for _, task in remaining:
                    task.cancel()
                await asyncio.gather(
                    *(task for _, task in remaining), return_exceptions=True
                )

        try:
            await run_workflow()
        finally:
            await self.close()
