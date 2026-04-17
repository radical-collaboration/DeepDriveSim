import asyncio
import ctypes
import random
from multiprocessing import Array as mp_Array
from pathlib import Path

from ddsim.ddsim_manager import DDSimManager


class ExchangeWorkflow(DDSimManager):
    """Replica-exchange workflow: runs an ensemble of MD simulations and
    periodically triggers coordinate/velocity exchanges between replicas
    once all active simulations have reached a common step checkpoint.

    Extends DDSimManager for scheduling and lifecycle management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate ExchangeWorkflow w/o asyncflow")

        self.config = kwargs.get("config")
        self.workflow_id = "exchange_workflow"

        # task_types must be populated before _generate_task_description is called.
        self.task_types = list(self.config["tasks"].keys())
        self.task_descriptions = self._generate_task_description(self.config)

        # All simulations run concurrently with no resource sharing for training.
        self.sim_batch_size = float("inf")
        self.max_sim_batch = float("inf")

        self.retrain_model = False
        self.call_finalize_results = True
        # If True, call evaluate_simulations after each sim; else call post_process_sim
        self.call_evaluate_simulations = True

        self.num_steps = self.config.get("num_steps", 1000)
        self.target_temp = self.config.get("target_temp", 300.0)

        # Shared arrays updated in-flight by simulation tasks (not only at completion).
        #   steps_reached[i]       — latest step checkpoint reported by sim i
        #   target_temp_reached[i] — True once sim i reaches target_temp
        num_sims = self.config.get("num_init_sims", 1)
        self.steps_reached = mp_Array(ctypes.c_int, num_sims)
        self.target_temp_reached = mp_Array(ctypes.c_bool, num_sims)

        # Exchange fires when every active sim has reached a multiple of this interval.
        self.exchange_step_interval = self.config.get("exchange_step_interval", 100)
        self._next_exchange_threshold = self.exchange_step_interval

        # Counter per sim type — used to generate unique sim_idx values.
        self.all_sims = {}

        # Optional callback invoked by _signal_ready; set by caller if needed.
        self._on_ready = None

        self.register_tasks()
        self.sim_inputs = {}

    # --------------------------------------------------------------------------
    async def _signal_ready(self) -> None:
        """Signal the CM that this workflow has produced enough data."""
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    # --------------------------------------------------------------------------
    def _init_experiment_dir(self) -> None:
        """Create the experiment directory from config."""
        self.experiment_dir = Path(self.config["experiment_dir"])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    def _generate_task_description(self, config):
        """Build resource-requirement dicts for each task type."""
        task_description = {}
        for t in self.task_types:
            task_cfg = config["tasks"][t]
            task_description[t] = {
                "ranks": 1,
                "cores_per_rank": task_cfg.get("cpu_reqs", 1),
                "gpus_per_rank": task_cfg.get("gpu_reqs", 0),
                "pre_exec": task_cfg.get("pre_exec", []),
                "shell": True,
            }
        return task_description

    # --------------------------------------------------------------------------
    async def _add_sims_to_queue(self, sim_type, sim_input=None):
        """Enqueue a new simulation task with a unique index."""
        n = self.all_sims.get(sim_type, 0)
        sim_idx = f"{sim_type}_{n}"
        self.all_sims[sim_type] = n + 1
        await self.sim_task_queue.put({"sim_idx": sim_idx})
        self.sim_inputs[sim_idx] = sim_input

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Populate the simulation task queue with the initial replica set."""
        num_init_sims = self.config["num_init_sims"]
        for _ in range(num_init_sims):
            await self._add_sims_to_queue("MD")

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """Return False — this workflow does not cancel simulations early."""
        return False

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        """Shut down once all queued simulations have completed."""
        if self.sim_task_queue.empty():
            self.shutting_down.set()
            self.run_workflow = False

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register MD simulation and exchange tasks.

        md_simulation runs as a function_task so it can update the shared
        steps_reached/target_temp_reached arrays in-flight during the step loop.
        exchange runs as a function_task invoked by evaluate_simulations once
        all active replicas have reached the current exchange threshold.
        """

        # --- MD simulation ---
        task_description = self.task_descriptions["MD"]
        steps_reached = self.steps_reached
        target_temp_reached = self.target_temp_reached
        num_steps = self.num_steps
        target_temp = self.target_temp

        @self.flow.function_task
        async def md_simulation(task_description=task_description, **kwargs):
            """Run one replica for num_steps, updating shared progress arrays.

            Replace the placeholder step body with real MD integration code.
            sim_slot is derived from the numeric suffix of sim_idx (e.g. 'MD_3' → 3).
            """
            sim_inputs = kwargs.get("sim_inputs", {})
            sim_idx = sim_inputs.get("sim_idx", "MD_0")
            sim_slot = int(sim_idx.split("_")[-1])

            # Replace with real MD initialisation.
            current_temp = 0.0

            for step in range(1, num_steps + 1):
                # Replace with one real MD step.
                current_temp += random.uniform(-0.5, 1.0)

                if step % 100 == 0:
                    steps_reached[sim_slot] = step
                    print(
                        f"[md_simulation] {sim_idx} step={step}/{num_steps}"
                        f"  temp={current_temp:.2f}",
                        flush=True,
                    )

                if current_temp >= target_temp and not target_temp_reached[sim_slot]:
                    target_temp_reached[sim_slot] = True
                    print(
                        f"[md_simulation] {sim_idx} reached target_temp"
                        f" {target_temp} K at step {step}",
                        flush=True,
                    )

            steps_reached[sim_slot] = num_steps

            return {
                "sim_idx": sim_idx,
                "steps_completed": num_steps,
                "final_temp": current_temp,
                "target_temp_reached": bool(target_temp_reached[sim_slot]),
            }

        self.simulation = md_simulation

        # --- Exchange ---
        task_description = self.task_descriptions["EXCHANGE"]

        @self.flow.function_task
        async def exchange(task_description=task_description, **kwargs):
            print("Add your code for exchange here")

        self.exchange = exchange

    # --------------------------------------------------------------------------
    async def add_sims_to_queue(self, sim_ids):
        """Re-queue simulations that were cancelled to free resources."""
        for sim_idx in sim_ids:
            sim_type = sim_idx.split("_")[0]
            sim_input = self.sim_inputs.get(sim_idx)
            await self._add_sims_to_queue(sim_type, sim_input)

    # --------------------------------------------------------------------------
    async def post_process(self):
        """Called after each inference iteration; advance workflow or shut down."""
        await self.finalize_results()

    # --------------------------------------------------------------------------
    def _all_sims_reached_threshold(self, completing_sim_idx: str) -> bool:
        """Return True when every active sim has reached the next exchange threshold.

        completing_sim_idx is the sim that just finished; it has already been
        removed from registered_sims and is counted as having reached the
        threshold (a completed sim has by definition run all num_steps).
        Returns False if no sim has ever been active.
        """
        threshold = self._next_exchange_threshold
        for sim_idx in self.registered_sims:
            try:
                slot = int(sim_idx.split("_")[-1])
            except ValueError:
                return False
            if self.steps_reached[slot] < threshold:
                return False
        return bool(self.registered_sims) or completing_sim_idx is not None

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self, sim_idx):
        """Called after each simulation completes.

        Triggers an exchange once all active replicas have reached the current
        step threshold, then advances the threshold for the next exchange window.
        """
        del self.sim_inputs[sim_idx]

        sim_type = sim_idx.split("_")[0]
        print(f"[evaluate_simulations] {sim_idx} completed (type={sim_type})")

        if self._all_sims_reached_threshold(completing_sim_idx=sim_idx):
            print(
                f"[evaluate_simulations] All sims reached step"
                f" {self._next_exchange_threshold} — triggering exchange"
            )
            await self.exchange()
            self._next_exchange_threshold += self.exchange_step_interval

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shut down the asyncflow engine."""
        await self.flow.shutdown()
