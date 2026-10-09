"""MiniApps workflow — Dragon/asyncflow backend, Delta HPC."""

import asyncio
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

from ddsim.ddsim_manager import DDSimManager


class MiniAppsWorkflow(DDSimManager):
    """MiniApps workflow — Dragon/asyncflow backend."""

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
        self.src_dir = cfg.get("src_dir") or _default_src

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
        # ── Simulation ────────────────────────────────────────────────────────
        @self.flow.executable_task(capture_stdio=True)
        async def simulation(**kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            return self._sim_cmd(sim_idx)

        self.simulation = simulation

        # ── Training ──────────────────────────────────────────────────────────
        @self.flow.executable_task(capture_stdio=True)
        async def training():
            return self._training_cmd()

        self.training = training

        # ── Prediction ────────────────────────────────────────────────────────
        @self.flow.executable_task(capture_stdio=True)
        async def prediction():
            return self._prediction_cmd()

        self.prediction = prediction

        # ── Selection ─────────────────────────────────────────────────────────
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
