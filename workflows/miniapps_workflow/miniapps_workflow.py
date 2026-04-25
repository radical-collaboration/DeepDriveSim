"""
miniapps_workflow_asyncflow.py — Decorator-based variant for asyncflow developer review.

This file shows the INTENDED design using @self.flow.executable_task /
@self.flow.function_task decorators for simulation, training, and prediction.
It is NOT the production version (see miniapps_workflow.py for the working
workaround).

ISSUES OBSERVED on Dragon backend V3 (DragonExecutionBackendV3):
------------------------------------------------------------------------

1. executable_task for simulation/training/prediction (the natural design):
   Dragon Batch worker subprocesses silently hang when the subprocess uses
   cupy/wfMiniAPI GPU operations, even when:
     - CUDA_VISIBLE_DEVICES is set via process_template → env
       (per asyncflow 06-dragon_execution_backend.py example)
     - CUDA_VISIBLE_DEVICES is also set explicitly in the shell command
     - The head process has correct CUDA_HOME / LD_LIBRARY_PATH (from sbatch)
   The subprocess never returns; no error is raised.

2. function_task for simulation (to run asyncio.create_subprocess_shell inside):
   Dragon must pickle the entire function closure to send to a worker.
   The closure captures `self`, which holds `self.flow` — the asyncflow
   WorkflowEngine — which contains non-picklable asyncio.Future objects.
   Error: "cannot pickle '_asyncio.Future' object"

3. function_task with no task_description:
   All tasks are immediately cancelled (t.cancelled() == True), causing
   DDSimManager._on_sim_done() to re-queue them in an infinite loop.
   Root cause: asyncflow/Dragon cannot schedule a task with no resource spec.

WORKING WORKAROUND (miniapps_workflow.py):
   All three tasks run as asyncio.create_subprocess_shell() on the head-process
   event loop, bypassing Dragon workers.  simulation wraps the coroutine in
   asyncio.ensure_future() so submit_sims() can call add_done_callback() on the
   returned asyncio.Task.  training and prediction are plain async functions
   awaited directly in train_model() / evaluate_simulations().

ENVIRONMENT:
   Platform  : Bridges-2 (PSC), 4x V100 GPUs per node
   Dragon    : HPE Dragon HPC runtime
   Backend   : DragonExecutionBackendV3
   asyncflow : radical.asyncflow / rhapsody
"""

import asyncio
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

from ddsim.ddsim_manager import DDSimManager


class MiniAppsWorkflow(DDSimManager):
    """
    Decorator-based MiniAppsWorkflow for asyncflow developer review.

    Identical to MiniAppsWorkflow except register_tasks() uses
    @self.flow.executable_task / @self.flow.function_task as intended.
    See module docstring for observed failure modes.
    """

    def __init__(self, config: dict = None, **kwargs):
        cfg = config or {}
        super().__init__(name=kwargs.get("name", "miniapps"))
        self.debug = config.get("debug", False)

        self.flow = kwargs.get("asyncflow", None)
        self._on_ready = kwargs.get("on_ready", None)
        self._data_ready_signaled = False
        self.miniapps_data_ready = int(cfg.get("miniapps_data_ready", 3))

        _home_base = Path(
            kwargs.get("home_dir", cfg.get("home_dir", Path.home() / "MiniApps"))
        )
        self.home_dir = self._ensure_dir(_home_base / self.name)
        self.clean_dir(self.home_dir)

        self.sim_output_dir = self._ensure_dir(self.home_dir / "sim_output")

        _default_src = str(Path(__file__).parent)
        self.src_dir = cfg.get("src_dir") or os.getenv("WORK_DIR", _default_src)

        _default_exe = os.path.expandvars(cfg.get("executable") or "") or sys.executable

        def _exe(key):
            val = cfg.get(key)
            return os.path.expandvars(val) if val else _default_exe

        self.sim_executable = _exe("sim_executable")
        self.train_executable = _exe("train_executable")
        self.predict_executable = _exe("predict_executable")
        self.selection_executable = _exe("selection_executable")

        self.prediction_file = self.home_dir / "predictions.yaml"

        self.max_sim_batch = int(cfg.get("max_sim_batch", 4))
        self.training_cores = int(cfg.get("training_cores", 1))
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        self.free_resources_for_train = bool(cfg.get("free_resources_for_train", True))

        self.call_cancel_simulations = True
        self.call_finalize_results = True
        self.call_evaluate_simulations = True

        self.total_num_sim = int(cfg.get("total_num_sim", 25))
        self.iteration = 0
        self.phase = int(cfg.get("phase", 0))
        self.num_step = int(cfg.get("num_step", 1000))
        self.num_epochs = int(cfg.get("num_epochs", 1000))
        self.num_mult = int(cfg.get("num_epochs", 1))
        self.retrain_model = True
        self.sim_predictions = {}

        self.clean_unregistered_sims = bool(cfg.get("clean_unregistered_sims", False))

        policies = kwargs.get("policies", None)
        if policies is not None:
            self.policy = policies[0] if policies else None
        else:
            self.policy = kwargs.get("policy", None)

        # ── task_description ──────────────────────────────────────────────────
        # Per asyncflow Dragon backend example (06-dragon_execution_backend.py),
        # the correct format is simply {"process_template": {...}} at top level —
        # no ranks/cores_per_rank/gpus_per_rank/shell/task_backend_specific_kwargs.
        #
        #   task_description = {
        #       "process_template": {
        #           "policy": <Policy>,
        #           "env": {
        #               "CUDA_VISIBLE_DEVICES": "0",
        #               "HDF5_USE_FILE_LOCKING": "FALSE",
        #           },
        #       }
        #   }
        #
        # OBSERVED FAILURE: even with correct structure, executable_task worker
        # subprocesses hang when running cupy GPU ops.  See module docstring.
        _gpu_id = self.policy.gpu_affinity[0] if self.policy else None

        _env = {"HDF5_USE_FILE_LOCKING": "FALSE"}
        if _gpu_id is not None:
            _env["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)

        _process_template = {"env": _env}
        if self.policy is not None:
            _process_template["policy"] = self.policy

        self.task_description = {"process_template": _process_template}

        if self.policy is not None:
            self.logger.info(
                f"Task policy: host={self.policy.host_name} "
                f"gpu_affinity={self.policy.gpu_affinity} "
                f"CUDA_VISIBLE_DEVICES={_gpu_id}",
                component=self.name,
            )
        else:
            self.logger.info("Task policy: none (no GPU affinity)", component=self.name)

        self.logger.info(
            f"Task description: {self.task_description}", component=self.name
        )

        self.register_tasks()
        self.sim_inputs = {}

    # --------------------------------------------------------------------------
    @staticmethod
    def _ensure_dir(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def clean_dir(dir_name):
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

    # --------------------------------------------------------------------------
    def _sim_cmd(self, sim_idx: int) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {sim_idx} "
            f"--phase {self.phase} "
            f"--num_step {self.num_step} "
        )
        return f"{self.sim_executable} {self.src_dir}/simulation.py {args}"

    def _training_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase} "
            f"--num_epochs {self.num_epochs}"
        )
        return f"{self.train_executable} {self.src_dir}/training.py {args}"

    def _prediction_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase} "
            f"--num_epochs {self.num_epochs} "
            f"--num_mult_outlier 1 "
            f"--num_mult {self.num_mult} "
            f"--output_file {self.prediction_file}"
        )
        return f"{self.predict_executable} {self.src_dir}/agent.py {args}"

    def _selection_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase}"
        )
        return f"{self.selection_executable} {self.src_dir}/selection.py {args}"

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """
        INTENDED design: all tasks use @self.flow.executable_task with a
        task_description that sets CUDA_VISIBLE_DEVICES via:
          {"process_template": {"policy": <Policy>,
           "env": {"CUDA_VISIBLE_DEVICES": "0", ...}}}

        FAILURE MODE for simulation/training/prediction:
          Dragon Batch worker subprocesses hang when the subprocess uses cupy GPU
          ops, even with correct task_description (process_template → env →
          CUDA_VISIBLE_DEVICES set, policy.gpu_affinity set).
          The head-process sbatch environment (CUDA_HOME, LD_LIBRARY_PATH) appears to
          reach Dragon workers (docs: "inherits parent environment"), but something in
          Dragon's worker sandbox prevents cupy from successfully initialising the GPU.
          No error is raised — the subprocess simply never exits.

        FAILURE MODE for simulation with function_task:
          Dragon must pickle the closure to send to a worker.  The closure captures
          `self` → `self.flow` (asyncflow WorkflowEngine) → asyncio.Future objects
          which are not picklable.
          Error: "cannot pickle '_asyncio.Future' object"

        FAILURE MODE for function_task with no task_description:
          asyncflow/Dragon immediately cancels all tasks (t.cancelled() == True)
          because there is no resource specification to schedule against.
        """

        # ── Simulation ────────────────────────────────────────────────────────
        # ISSUE: hangs — Dragon worker subprocess cannot run cupy GPU ops.
        # CUDA_VISIBLE_DEVICES is set via process_template → env, but the
        # subprocess still hangs on cupy GPU init.
        @self.flow.executable_task
        async def simulation(task_description=self.task_description, **kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            return self._sim_cmd(sim_idx)

        self.simulation = simulation

        # ── Training ──────────────────────────────────────────────────────────
        # ISSUE: same hang — training.py uses wfMiniAPI cupy GPU ops.
        @self.flow.executable_task
        async def training(task_description=self.task_description):
            return self._training_cmd()

        self.training = training

        # ── Prediction ────────────────────────────────────────────────────────
        # ISSUE: same hang — agent.py uses wfMiniAPI cupy GPU ops.
        @self.flow.executable_task
        async def prediction(task_description=self.task_description):
            return self._prediction_cmd()

        self.prediction = prediction

        # ── Selection ─────────────────────────────────────────────────────────
        # No GPU ops — function_task works here (no cupy, no pickling issue
        # since selection() is awaited, not used with add_done_callback).
        @self.flow.function_task
        async def selection(*args, **kwargs):
            return self._selection_cmd()

        self.selection = selection

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self):
        await self.prediction()
        with open(self.prediction_file) as f:
            predictions = yaml.safe_load(f)
        self.sim_predictions = predictions

    def stop_simulation(self, *args, **kwargs):
        if random.random() < 0.5:
            return False
        return True

    async def init_sim_queue(self):
        for sim_idx, s in enumerate(range(self.total_num_sim)):
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.sim_inputs[sim_idx] = s

    async def add_sims_to_queue(self, resubmitted_sims):
        for sim_idx in resubmitted_sims:
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.logger.info(
                f"Re-added Sim {sim_idx} back the queue", component=self.name
            )
            if sim_idx not in self.sim_inputs:
                raise ValueError(f"Unable to add sim {sim_idx} to queue")

    async def check_train_status(self):
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            ranks = comm.Get_size()
        except Exception:
            ranks = 1

        root_path = Path(self.sim_output_dir, "phase0")
        next_iter = self.iteration + 1
        filenames = [
            Path(root_path, f"data_{rank}_{next_iter}.h5") for rank in range(ranks)
        ]

        self.logger.info(
            f"Waiting for {len(filenames)} file to start training...",
            component=self.name,
        )
        start_trainig = False
        while True:
            if start_trainig:
                break
            start_trainig = True
            for filename in filenames:
                if not filename.exists():
                    if self.debug:
                        self.logger.info(
                            f"File {filename} not found yet, wait...",
                            component=self.name,
                        )
                    start_trainig = False
                    await asyncio.sleep(5)
                    break

        self.logger.info(
            "All required files are available. Starting training...",
            component=self.name,
        )
        return True

    async def train_model(self):
        self.iteration += 1
        if self.debug:
            self.logger.info(
                f"\nTraining Iteration {self.iteration}", component=self.name
            )
            self.logger.info(
                f"{len(self.registered_sims)} simulation(s) running....",
                component=self.name,
            )
        await self.training()
        if self.debug:
            self.logger.task_completed("Training Completed", component=self.name)

    async def _signal_ready(self) -> None:
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    async def finalize_results(self):
        n_done = len(self.completed_sims)
        if not self._data_ready_signaled and n_done >= self.miniapps_data_ready:
            self._data_ready_signaled = True
            if self.debug:
                self.logger.info(
                    f"{n_done} sims completed"
                    " — signaling ready for downstream workflows",
                    component=self.name,
                )
            await self._signal_ready()
        if n_done >= self.total_num_sim:
            self.shutting_down.set()
            self.run_workflow = False
            self.logger.info("All sim have completed...", component=self.name)
        else:
            if self.debug:
                self.logger.info(
                    f"sim {n_done} out of {self.total_num_sim} have completed...",
                    component=self.name,
                )

    async def post_process_sim(self, sim_idx):
        pass

    async def close(self):
        # Temporary cleanup to test campaign manager and aid Disk quota exceeded.
        if self.home_dir.exists():
            shutil.rmtree(self.home_dir, ignore_errors=True)
            self.logger.info(
                f"Removed home directory: {self.home_dir}", component=self.name
            )

    async def stop(self):
        await self.close()
