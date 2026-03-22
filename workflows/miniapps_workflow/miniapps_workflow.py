import asyncio
import os
import random
import shutil
import sys
from pathlib import Path

import yaml
try:
    from rose import Learner
except ModuleNotFoundError:
    Learner = None

from ddsim.ddsim_manager import DDSimManager

class MiniAppsWorkflow(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, config: dict = None, **kwargs):
        """
        Parameters
        ----------
        config : dict, optional
            Application-level parameters loaded from the campaign YAML.
            All MiniAppsWorkflow-specific keys are read from here.
        kwargs : dict
            Framework-level params set by the campaign/miniapps_workflow layer:
              asyncflow, name, home_dir, on_ready
        """
        cfg = config or {}
        # Initialize parent class (sets up logger, queues, etc.)
        super().__init__(name=kwargs.get("name", "miniapps"))
        self.debug = config.get("debug", False)

        self.flow = kwargs.get("asyncflow", None)
        self.learner = Learner(self.flow) if Learner is not None else None

        # on_ready: async callable injected by AsyncCampaignManager so that
        # _signal_ready() can unblock dependent workflow groups (e.g. inference).
        self._on_ready = kwargs.get("on_ready", None)
        self._data_ready_signaled = False
        self.miniapps_data_ready = int(cfg.get("miniapps_data_ready", 3))

        # Default home directory
        home_dir = Path(kwargs.get("home_dir", Path.home() / "DDMD"))
        self.clean_dir(home_dir)  # ❗Careful: deletes everything in home_dir!

        # Create output directory for simulations
        self.sim_output_dir = self._ensure_dir(home_dir / f"{self.name}/sim_output")
        self.clean_dir(self.sim_output_dir)

        # ── Executable paths ──────────────────────────────────────────────────
        # Default src_dir to the directory that contains miniapps_workflow.py so
        # simulation.py, training.py, etc. are found regardless of WORK_DIR or cwd.
        _default_src = str(Path(__file__).parent)
        self.src_dir = cfg.get("src_dir") or os.getenv("WORK_DIR", _default_src)

        # Per-task python executables — each can point to a different conda env.
        # Falls back to the current interpreter if not set.
        _default_exe = sys.executable
        self.sim_executable = cfg.get("sim_executable") or _default_exe
        self.train_executable = cfg.get("train_executable") or _default_exe
        self.predict_executable = cfg.get("predict_executable") or _default_exe
        self.selection_executable = cfg.get("selection_executable") or _default_exe

        # Prediction (agent) is running as executable for
        # miniapps and writes all scores to file
        self.prediction_file = home_dir / "predictions.yaml"

        # Simulation/training config
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

        # Dragon Policy for GPU/CPU affinity (injected by AsyncCampaignManager).
        # CM passes a list (`policies`); we use the first entry for this single-GPU
        # workflow.  A bare `policy` kwarg is also accepted for standalone use.
        policies = kwargs.get("policies", None)
        if policies is not None:
            self.policy = policies[0] if policies else None
        else:
            self.policy = kwargs.get("policy", None)

        # When a Dragon Policy is provided, GPU assignment is handled by the
        # policy's gpu_affinity.  Setting gpus_per_rank simultaneously causes
        # Dragon to double-count the GPU requirement and may result in
        # CUDA_VISIBLE_DEVICES="" or a Dragon IPC deadlock.
        self.task_description = {
            "ranks": 1,
            "cores_per_rank": 1,
            "gpus_per_rank": 0 if self.policy is not None else 1,
            "shell": True,
        }
        if self.policy is not None:
            self.task_description["process_template"] = {"policy": self.policy}
            self.logger.info(
                f"Task policy: host={self.policy.host_name} "
                f"gpu_affinity={self.policy.gpu_affinity}",
                component=self.name,
            )
        else:
            self.logger.info("Task policy: none (no GPU affinity)", component=self.name)

        # Register learner tasks
        self.register_tasks()
        # To store input files
        self.sim_inputs = {}

    # --------------------------------------------------------------------------
    @staticmethod
    def _ensure_dir(path):
        """Create directory if it does not exist."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --------------------------------------------------------------------------
    @staticmethod
    def clean_dir(dir_name):
        """Delete an existing directory (used for a clean workflow run)."""
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self):
        await self.prediction()
        with open(self.prediction_file) as f:
            predictions = yaml.safe_load(f)
        self.sim_predictions = predictions

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """
        Check prediction score: If it returns True
        then simulation will be canceled
        """
        if random.random() < 0.5:
            return False
        else:
            return True

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Collect all simulation input files into task queue."""
        for sim_idx, s in enumerate(range(self.total_num_sim)):
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.sim_inputs[sim_idx] = s

    # --------------------------------------------------------------------------
    async def add_sims_to_queue(self, resubmitted_sims):
        for sim_idx in resubmitted_sims:
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.logger.info(
                f"Re-added Sim {sim_idx} back the queue", component=self.name
            )
            if sim_idx not in self.sim_inputs:
                raise ValueError(f"Unable to add  sim {sim_idx} to queue ")

    # --------------------------------------------------------------------------
    async def check_train_status(self):
        """Check if enough training data is available to start training."""

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            ranks = comm.Get_size()
        except Exception:
            ranks = 1

        root_path = Path(self.sim_output_dir, "phase0")
        # train_model() increments self.iteration before calling training.py, so
        # training.py reads data_{rank}_{iteration+1}.h5. Check for that file.
        next_iter = self.iteration + 1
        filenames = [
            Path(root_path, f"data_{rank}_{next_iter}.h5") for rank in range(ranks)
        ]

        self.logger.info(
            f"Waiting for {len(filenames)} file to start training... ",
            component=self.name,
        )
        start_trainig = False
        while True:
            if start_trainig:
                # await asyncio.sleep(5)
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

    # --------------------------------------------------------------------------
    # Command builders — called at task execution time so self.iteration and
    # self.phase always reflect the current workflow state, not init values.
    # --------------------------------------------------------------------------

    def _sim_cmd(self, sim_idx: int) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {sim_idx} "
            f"--phase {self.phase} "
            f"--num_step {self.num_step} "
        )
        # Explicitly set CUDA_VISIBLE_DEVICES so Dragon Batch subprocesses use
        # the correct GPU.  Without this, Dragon may set CUDA_VISIBLE_DEVICES=-1
        # (no GPU) for tasks with gpus_per_rank=0, causing cupy to fall back to
        # CPU and making each simulation ~16x slower.
        gpu_id = self.policy.gpu_affinity[0] if self.policy else None
        cuda_env = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        return f"env HDF5_USE_FILE_LOCKING=FALSE {cuda_env}{self.sim_executable} {self.src_dir}/simulation.py {args}"

    def _training_cmd(self) -> str:
        gpu_id = self.policy.gpu_affinity[0] if self.policy else None
        cuda_env = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase} "
            f"--num_epochs {self.num_epochs}"
        )
        return f"env HDF5_USE_FILE_LOCKING=FALSE {cuda_env}{self.train_executable} {self.src_dir}/training.py {args}"

    def _prediction_cmd(self) -> str:
        gpu_id = self.policy.gpu_affinity[0] if self.policy else None
        cuda_env = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase} "
            f"--num_epochs {self.num_epochs} "
            f"--num_mult_outlier 1 "
            f"--num_mult {self.num_mult} "
            f"--output_file {self.prediction_file}"
        )
        return f"env HDF5_USE_FILE_LOCKING=FALSE {cuda_env}{self.predict_executable} {self.src_dir}/agent.py {args}"

    def _selection_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase}"
        )
        return f"{self.selection_executable} {self.src_dir}/selection.py {args}"

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        # Simulation runs via asyncio.ensure_future on the head-process event loop,
        # bypassing Dragon workers entirely.
        #
        # Dragon worker subprocesses hang for this simulation regardless of
        # CUDA_VISIBLE_DEVICES or process_template.env settings — likely due to
        # process isolation or resource limits imposed by Dragon's Batch pool.
        # executable_task and function_task both route through Dragon workers.
        # function_task additionally fails because Dragon must pickle the closure,
        # which captures self.flow (asyncflow engine with non-picklable asyncio.Future).
        #
        # asyncio.create_subprocess_shell() from the head process works: it inherits
        # the full sbatch environment (CUDA_HOME, LD_LIBRARY_PATH) and CUDA access.
        # submit_sims() needs self.simulation() to return a future with
        # add_done_callback(); asyncio.ensure_future() wraps the coroutine as a Task.
        async def _sim_coro(**kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            cmd = self._sim_cmd(sim_idx)
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                print(stdout.decode(), end="", flush=True)
            if proc.returncode != 0:
                self.logger.error(
                    f"Simulation {sim_idx} failed with exit code {proc.returncode}",
                    component=self.name,
                )

        # simulation cannot use @self.flow.function_task: a sync function decorated
        # with function_task causes asyncflow to schedule it as a Dragon worker task.
        # In that context asyncio.ensure_future() fails (no running event loop),
        # silently crashing submit_sims() so no simulations are ever submitted.
        # It must remain a plain sync function returning an asyncio.Task directly.
        def simulation(**kwargs):
            return asyncio.ensure_future(_sim_coro(**kwargs))

        self.simulation = simulation

        # Training and prediction must NOT use @self.flow.function_task:
        # when a function_task completes, asyncflow marks it done and Dragon
        # tears down the workflow, cancelling all remaining replicas mid-run.
        # Plain async functions awaited directly in train_model() /
        # evaluate_simulations() avoid any asyncflow lifecycle tracking.
        async def training():
            cmd = self._training_cmd()
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                print(stdout.decode(), end="", flush=True)
            if proc.returncode != 0:
                self.logger.error(
                    f"Training failed with exit code {proc.returncode}",
                    component=self.name,
                )

        self.training = training

        async def prediction():
            cmd = self._prediction_cmd()
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                print(stdout.decode(), end="", flush=True)
            if proc.returncode != 0:
                self.logger.error(
                    f"Prediction failed with exit code {proc.returncode}",
                    component=self.name,
                )

        self.prediction = prediction

        @self.flow.function_task
        async def selection(*args, **kwargs):
            return self._selection_cmd()

        self.selection = selection

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
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

    # --------------------------------------------------------------------------
    async def _signal_ready(self) -> None:
        """Signal the CM that this workflow has produced enough data."""
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        n_done = len(self.completed_sims)

        if not self._data_ready_signaled and n_done >= self.miniapps_data_ready:
            self._data_ready_signaled = True
            if self.debug:
                self.logger.info(
                    f"{n_done} sims completed — signaling ready for downstream workflows",
                    component=self.name,
                )
            await self._signal_ready()

        if n_done >= self.total_num_sim:
            self.shutting_down.set()
            self.run_workflow = False
            self.logger.info("All sim have completed...", component=self.name)
        else:
            if self.debug:
                self.logger.info(f"sim {n_done} out of {self.total_num_sim} have completed...", component=self.name)

    async def post_process_sim(self, sim_idx):
        pass

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner."""
        try:
            await self.learner.shutdown()
        except Exception:
            pass

    async def stop(self):
        """Alias for close(), can be used for external termination."""
        try:
            await self.learner.shutdown()
        except Exception:
            pass
