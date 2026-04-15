import asyncio
import ctypes
import random
from multiprocessing import Array as mp_Array
from pathlib import Path

from ddsim.ddsim_manager import DDSimManager


class ExchangeWorkflow(DDSimManager):
    """DeepDriveMD workflow: orchestrates MD simulations, ML training,
    aggregation, model selection, and agent-based inference stages.

    Extends DDSimManager to implement the full DeepDriveMD workflow
    using asyncflow for task scheduling and execution.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class (sets up logger, queues, event flags, etc.)
        super().__init__()

        # Asyncflow engine for registering and dispatching tasks
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate ExchangeWorkflow w/o asyncflow")

        self.config = kwargs.get("config")

        self.workflow_id = "exchange_workflow"

        #self._init_experiment_dir()

        # task_types must be set before _generate_task_description is called
        self.task_types = list(self.config["tasks"].keys())
        self.task_descriptions = self._generate_task_description(self.config)

        # Batch size equals max since no resource sharing between sim and train
        self.sim_batch_size = float("inf")
        self.max_sim_batch = float("inf")  # required by DDSimManager._on_sim_done

        # If False, skip retraining (e.g. when model accuracy is sufficient)
        self.retrain_model = False
        self.call_finalize_results = True
        self.call_evaluate_simulations = True  # If True, call evaluate_simulations after each sim; else call post_process_sim

        # ── MD simulation parameters ──────────────────────────────────────────
        # Total number of MD steps per simulation run
        self.num_steps = self.config.get("num_steps", 1000)
        # Target temperature (K); sim marks target_temp_reached when hit
        self.target_temp = self.config.get("target_temp", 300.0)

        # ── Shared progress arrays (one slot per concurrent simulation) ───────
        # Updated in-flight by md_simulation tasks (not just at completion).
        # steps_reached[i]       — last step checkpoint reported by sim i
        #                          (updated every 100 steps)
        # target_temp_reached[i] — True once sim i reaches target_temp
        num_sims = self.config.get("num_init_sims", 1)
        self.steps_reached = mp_Array(ctypes.c_int, num_sims)
        self.target_temp_reached = mp_Array(ctypes.c_bool, num_sims)

        # ── Exchange trigger ──────────────────────────────────────────────────
        # Exchange runs when every active simulation has reported at least
        # exchange_step_interval steps.  Checked in post_process_sim.
        self.exchange_step_interval = self.config.get("exchange_step_interval", 100)
        # Tracks the next exchange threshold that must be reached before firing.
        self._next_exchange_threshold = self.exchange_step_interval

        # Counter per sim type — used to generate unique sim_idx values
        self.all_sims = {}

        # Optional callback invoked by _signal_ready; set by caller if needed
        self._on_ready = None

        # Register simulation, training, aggregation, inference, selection tasks
        self.register_tasks()
        # Dict to store inputs for simulation
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
        """Create the experiment directory from config and store as self.experiment_dir."""
        self.experiment_dir = Path(self.config["experiment_dir"])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    def _generate_task_description(self, config):
        """Build a single task resource description from a stage config."""
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
        """Add new simulation tasks to the queue based on sim_idx."""
        n = self.all_sims.get(sim_type, 0)
        sim_idx = f"{sim_type}_{n}"
        self.all_sims[sim_type] = n + 1
        await self.sim_task_queue.put({"sim_idx": sim_idx})
        self.sim_inputs[sim_idx] = sim_input

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Populate the simulation task queue with initial PDB inputs.

        Cycles through available PDB files so that each simulation task
        gets an input structure. If there are more tasks than PDB files,
        files are reused in round-robin order.
        """
        num_init_sims = self.config["num_init_sims"]
        for _ in range(num_init_sims):
            await self._add_sims_to_queue('MD')

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """Decide whether to cancel a simulation based on prediction.

        Returns False by default (no early cancellation).
        This workflow designed to replicate original ENTK workflow
        """
        return False

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        """Advance to next iteration or signal shutdown if max reached.

        Increments stage_idx and queues the next batch of simulation
        tasks (with pdb_file=None, so the simulation task will use
        restart PDBs from the previous iteration's predictions).
        """
        if self.sim_task_queue.empty():
            self.shutting_down.set()
            self.run_workflow = False

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register function for all types of simulations."""

        # ── MD simulation ─────────────────────────────────────────────────────
        # function_task: runs Python code directly on the worker (not a shell
        # command), which allows the step loop to update shared arrays in-flight.
        task_description = self.task_descriptions["MD"]
        steps_reached = self.steps_reached
        target_temp_reached = self.target_temp_reached

        num_steps   = self.num_steps
        target_temp = self.target_temp

        @self.flow.function_task
        async def md_simulation(task_description=task_description, **kwargs):
            """Run MD simulation for num_steps, updating shared arrays in-flight.

            Called by DDSimManager.submit_sims as:
                self.simulation(sim_inputs={"sim_idx": "MD_0"})

            num_steps and target_temp come from the ExchangeWorkflow closure.
            sim_slot is derived from the numeric suffix of sim_idx (e.g. 'MD_3' → 3).
            """
            sim_inputs = kwargs.get("sim_inputs", {})
            sim_idx    = sim_inputs.get("sim_idx", "MD_0")
            # Derive array slot from the trailing integer in sim_idx (e.g. 'MD_3' → 3)
            sim_slot   = int(sim_idx.split("_")[-1])

            # ── placeholder: replace with real MD initialisation ──────────────
            current_temp = 0.0

            for step in range(1, num_steps + 1):
                # ── placeholder: replace with one real MD step ────────────────
                current_temp += random.uniform(-0.5, 1.0)

                # Every 100 steps record the latest checkpoint
                if step % 100 == 0:
                    steps_reached[sim_slot] = step
                    print(
                        f"[md_simulation] {sim_idx} step={step}/{num_steps}"
                        f"  temp={current_temp:.2f}",
                        flush=True,
                    )

                # Record the first time the target temperature is reached
                if current_temp >= target_temp and not target_temp_reached[sim_slot]:
                    target_temp_reached[sim_slot] = True
                    print(
                        f"[md_simulation] {sim_idx} reached target_temp"
                        f" {target_temp} K at step {step}",
                        flush=True,
                    )

            # Final checkpoint for the last partial 100-step block
            steps_reached[sim_slot] = num_steps

            return {
                "sim_idx": sim_idx,
                "steps_completed": num_steps,
                "final_temp": current_temp,
                "target_temp_reached": bool(target_temp_reached[sim_slot]),
            }

        self.simulation = md_simulation

        # ── Exchange ──────────────────────────────────────────────────────────
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
        """Return True when every active simulation has reached the next exchange threshold.

        completing_sim_idx is the sim that just finished and triggered this check.
        It has already been removed from registered_sims by _on_sim_done, so it
        is passed explicitly and counted as having reached the threshold (a sim
        that completed has by definition run all num_steps >= threshold).

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
        # At least one sim must have been active: either still running or the
        # one that just completed.
        return bool(self.registered_sims) or completing_sim_idx is not None

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self, sim_idx):
        del self.sim_inputs[sim_idx]

        sim_type = sim_idx.split("_")[0]
        print(f'[evaluate_simulations] {sim_idx} completed (type={sim_type})')

        # Trigger exchange when all active simulations have reached the
        # current step threshold (a multiple of exchange_step_interval).
        if self._all_sims_reached_threshold(completing_sim_idx=sim_idx):
            print(
                f'[evaluate_simulations] All sims reached step {self._next_exchange_threshold}'
                f' — triggering exchange'
            )
            await self.exchange()
            # Advance threshold to the next interval so exchange fires once per window
            self._next_exchange_threshold += self.exchange_step_interval

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shut down the asyncflow engine."""
        await self.flow.shutdown()
