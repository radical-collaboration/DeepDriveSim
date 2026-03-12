import asyncio
import os
import shutil
import sys
from pathlib import Path

try:
    from rose import Learner
except ModuleNotFoundError:
    Learner = None

import numpy as np
import yaml
from rose.metrics import MODEL_ACCURACY

from ddsim.ddsim_manager import DDSimManager

task_description = {
    "shell": True,
}


class DummyWorkflow(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    workflow_id = "dummy_workflow"

    def __init__(self, config: dict = None, **kwargs):
        """
        Parameters
        ----------
        config : dict, optional
            Application-level parameters loaded from the campaign YAML.
            All DummyWorkflow-specific keys are read from here.
        kwargs : dict
            Framework-level params set by the campaign/ddsim_workflow layer:
              asyncflow, name, home_dir, on_ready
        """
        cfg = config or {}
        # on_ready: async callable injected by AsyncCampaignManager so that
        # _signal_ready() can unblock dependent workflow groups (e.g. inference).
        self._on_ready = kwargs.get("on_ready", None)

        super().__init__(name=kwargs.get("name", "ddsim"))

        self.debug = config.get("debug", False)
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate DummyWorkflow w/o asyncflow")
        self.learner = Learner(self.flow)

        home_dir = Path(kwargs.get("home_dir", Path.home() / "DDSim"))
        self._clean_dir(home_dir)  # ❗Careful: deletes everything in home_dir!

        # Create workflow directories
        self.sim_output_dir = self._ensure_dir(home_dir / "sim_output")
        self.sim_inputs_dir = self._ensure_dir(home_dir / "sim_input")
        self.train_dir      = self._ensure_dir(home_dir / "train")
        self.train_al_dir   = self._ensure_dir(home_dir / "train_al")
        self.val_dir        = self._ensure_dir(home_dir / "val")

        # ── Simulation / training parameters ──────────────────────────────────
        self.num_inputs             = int(cfg.get("num_inputs", 25))
        self.max_sim_batch          = int(cfg.get("max_sim_batch", 24))
        self.training_cores         = int(cfg.get("training_cores", 1))
        self.sim_batch_size         = self.max_sim_batch + self.training_cores
        self.training_threshold     = float(cfg.get("training_threshold", 0.5))
        self.prediction_threshold   = float(cfg.get("prediction_threshold", 0.5))
        self.start_training_threshold = int(cfg.get("start_training_threshold", 1))
        self.training_epochs        = int(cfg.get("training_epochs", 1))
        self.free_resources_for_train = bool(cfg.get("free_resources_for_train", True))
        # Must exceed simulation.py's asyncio.sleep so evaluation doesn't
        # always preempt the last running sim before it can finish.
        self.sleep_time             = float(cfg.get("sleep_time", 30))

        self.call_cancel_simulations  = True
        self.call_finalize_results    = True
        self.call_evaluate_simulations = True

        self.iteration    = 0
        self.retrain_model = self.training_epochs > 0

        # ── Dependent workflow trigger ─────────────────────────────────────────────────
        # After this many sims complete, signal the CM that enough data has been
        # produced for dependent workflows (e.g. inference) to start.
        # Defaults to num_inputs (i.e. signal only when all sims finish) so that
        # dep_threshold remains the fallback strategy when not explicitly set.
        self.ddsim_data_ready = int(
            cfg.get("ddsim_data_ready", self.num_inputs)
        )
        self._data_ready_signaled = False  # fire the signal at most once

        # ── Executable paths ──────────────────────────────────────────────────
        # Default src_dir to the directory that contains dummy_workflow.py so
        # simulation.py, train.py, etc. are found regardless of WORK_DIR or cwd.
        _default_src = str(Path(__file__).parent)
        self.src_dir   = cfg.get("src_dir", os.getenv("WORK_DIR", _default_src))
        self.code_path = cfg.get("code_path", f"{sys.executable} {self.src_dir}")

        self.model_filename  = home_dir / "model.pkl"
        self.prediction_file = home_dir / "predictions.yml"

        # Register learner tasks then generate dummy input files
        self.register_tasks()
        self.sim_inputs = {}
        self._generate_sim_inputs(self.sim_inputs_dir, num_inputs=self.num_inputs)

    # --------------------------------------------------------------------------
    async def _signal_ready(self) -> None:
        """Signal the CM that this workflow has produced enough data."""
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    # --------------------------------------------------------------------------
    @staticmethod
    def _ensure_dir(path):
        """Create directory if it does not exist."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --------------------------------------------------------------------------
    @staticmethod
    def _clean_dir(dir_name):
        """Delete an existing directory (used for a clean workflow run)."""
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

    # --------------------------------------------------------------------------
    @staticmethod
    def _generate_sim_inputs(sim_inputs_dir, num_inputs: int = 5):
        """
        Generate dummy input `.npz` files for simulations.
        """
        sim_inputs_path = Path(sim_inputs_dir)
        for i in range(num_inputs):
            file_path = sim_inputs_path / f"config_{i}.npz"
            x = np.random.rand(100, 1)  # Dummy input data
            np.savez(file_path, X=x)

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        #@self.learner.simulation_task()
        @self.flow.executable_task
        async def simulation(task_description=task_description, **kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            filename = self.sim_inputs[sim_idx]
            args = (
                f"--output_dir {self.sim_output_dir} --sim_tag {sim_idx} "
                f"--filename {filename}"
            )
            return f"{self.code_path}/simulation.py {args}"

        self.simulation = simulation

        @self.learner.training_task()
        async def training(task_description=task_description, **kwargs):
            args = (
                f"--model_filename {self.model_filename} "
                f"--sim_output_dir {self.sim_output_dir} "
                f"--train_dir {self.train_al_dir} --val_dir {self.val_dir}"
            )
            return f"{self.code_path}/train.py {args}"

        self.training = training

        @self.learner.active_learn_task()
        async def active_learn(task_description=task_description, **kwargs):
            args = (
                f"--model_filename {self.model_filename} "
                f"--train_dir {self.train_dir} "
                f"--train_al_dir {self.train_al_dir}"
            )
            return f"{self.code_path}/active_learn.py {args}"

        self.active_learn = active_learn

        #@self.learner.prediction_task(as_executable=True)
        @self.flow.executable_task
        async def prediction(task_description=task_description, **kwargs):
            args = (
                f"--model_filename {self.model_filename} "
                f"--sim_output_dir {self.sim_output_dir} "
                f"--output_file {self.prediction_file}"
            )
            return f"{self.code_path}/predict.py {args}"

        self.prediction = prediction

        @self.learner.as_stop_criterion(
            metric_name=MODEL_ACCURACY, threshold=self.training_threshold
        )
        async def check_accuracy(task_description=task_description, **kwargs):
            args = f"--model_filename {self.model_filename} --val_dir {self.val_dir}"
            return f"{self.code_path}/check_accuracy.py {args}"

        self.check_accuracy = check_accuracy

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs) -> bool:
        """Return True if prediction < threshold (cancel simulation)."""
        if self.debug:
            self.logger.info(f"Prediction is { kwargs['prediction']}", component=self.name)
        return kwargs["prediction"] < self.prediction_threshold

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self) -> dict:
        if self.debug:
            self.logger.task_started(f"Model Prediction", component="prediction")
        await self.prediction()
        if self.debug:
            self.logger.task_completed(f"Model Prediction", component="prediction")
        with open(self.prediction_file) as f:
            predictions = yaml.safe_load(f)

        self.sim_predictions = predictions

    # --------------------------------------------------------------------------
    async def init_sim_queue(self) -> None:
        """Collect all simulation input files into task queue."""
        filenames = await asyncio.to_thread(lambda: list(self.sim_inputs_dir.iterdir()))
        for sim_idx, filename in enumerate(filenames):
            if filename.is_file():
                sim_name = filename.stem
                sim_idx = f"{sim_name}"
                await self.sim_task_queue.put({"sim_idx": sim_idx})
                self.sim_inputs[sim_idx] = filename
        await self.sim_task_queue.put(None)

    # --------------------------------------------------------------------------
    async def add_sims_to_queue(self, resubmitted_sims):
        for sim_idx in resubmitted_sims:
            if sim_idx not in self.sim_inputs:
                raise ValueError(f"Unable to add sim {sim_idx} to queue")
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.logger.info(f"Re-added Sim {sim_idx} back to queue", component=self.name)

    # --------------------------------------------------------------------------
    async def check_train_status(self) -> bool:
        """Check if enough training data is available to start training."""
        filenames = await asyncio.to_thread(lambda: list(self.sim_inputs_dir.iterdir()))
        return len(filenames) >= self.start_training_threshold

    # --------------------------------------------------------------------------
    async def post_process_sim(self, sim_idx):
        """
        Asynchronously delete all files associated
        with a simulation index (safe parallel cleanup).
        """
        del self.sim_inputs[sim_idx]

        async def _delete_file(file_path):
            try:
                await asyncio.to_thread(os.remove, file_path)
            except FileNotFoundError:
                self.logger.warning(f"File already removed: {file_path}", component=self.name)
            except Exception as e:
                self.logger.error(f"Error deleting {file_path}: {e}", component=self.name)

        # Collect all deletion tasks (parallel file cleanup)
        tasks = []
        for directory in [self.train_al_dir, self.train_dir, self.val_dir]:
            for filename in directory.iterdir():
                if sim_idx in filename.name:
                    tasks.append(_delete_file(filename))
        if tasks:
            await asyncio.gather(*tasks)

        # Remove simulation output directory after files are gone
        sim_dir = Path(self.sim_output_dir, sim_idx)
        try:
            if sim_dir.exists():
                await asyncio.to_thread(shutil.rmtree, sim_dir)
                if self.debug:
                    self.logger.warning(
                        f"Simulation directory has been removed: {sim_dir}",
                        component=self.name,
                    )
            else:
                if self.debug:
                    self.logger.warning(
                        f"Simulation directory already removed: {sim_dir}", component=self.name
                    )
        except Exception as e:
            self.logger.error(f"Error deleting directory {sim_dir}: {e}", component=self.name)
        if self.debug:
            self.logger.info(
                f"Removed all files related to simulation {sim_idx}", component=self.name
            )

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
        # If all simulations have already completed, no new training data will
        # ever appear in sim_output_dir (post_process_sim deletes everything).
        # Stop retraining immediately to prevent train.py from hanging on empty
        # directories. DDSimManager.start() already decremented sim_batch_size
        # by training_cores before calling us, so restore it here.
        if len(self.completed_sims) >= self.num_inputs:
            # self.retrain_model = False
            # self.sim_batch_size += self.training_cores
            # self.training_cores = 0
            return

        self.iteration += 1
        for epoch in range(self.training_epochs):
            if self.debug:
                self.logger.info(
                    f"Iteration {self.iteration} / Epoch {epoch + 1}", component=self.name
                )

            if self.debug:
                self.logger.task_started("Model Training", component=self.name)

            await self.training()

            if self.debug:
                self.logger.task_completed("Model Training", component=self.name)
                self.logger.task_started("Check Accuracy", component=self.name)

            should_stop, metric_val = await self.check_accuracy()

            if should_stop:
                if self.debug:
                    self.logger.info(
                        f"Accuracy ({metric_val}) reached threshold → stopping training",
                        component=self.name,
                    )
                self.retrain_model = False
                self.sim_batch_size += self.training_cores
                self.training_cores = 0
                break
            if self.debug:
                self.logger.task_completed("Check Accuracy", component=self.name)

            if self.debug:
                self.logger.task_started("Active Learning", component=self.name)

            await self.active_learn()

            if self.debug:
                self.logger.task_completed("Active Learning", component=self.name)

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        n_done = len(self.completed_sims)

        # Signal dependents (e.g. inference) when enough data is produced.
        if not self._data_ready_signaled and n_done >= self.ddsim_data_ready:
            self._data_ready_signaled = True
            self.logger.info(
                f"{n_done} sims completed — signaling ready for downstream workflows",
                component=self.name,
            )
            await self._signal_ready()

        if n_done >= self.num_inputs:
            if self.registered_sims:
                self.logger.warning(f"All sim have completed BUT there are still registered sims {self.registered_sims.keys()}", component=self.name)
            else:
                self.shutting_down.set()
                self.run_workflow = False
                self.logger.info("All sim have completed...", component=self.name)
        else:
            if self.debug:
                self.logger.warning(
                    f" {n_done} sim have completed out of {self.num_inputs}", component=self.name
                )

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner.

        NOTE: self.flow (the asyncflow WorkflowEngine) is intentionally NOT shut
        down here — and neither is self.learner, since Learner.shutdown() delegates
        directly to asyncflow.shutdown().  Calling either from one replica while
        other replicas are still running destroys shared asyncio subprocess
        infrastructure and causes concurrent replicas' asyncflow task submissions
        to hang indefinitely.  The AsyncCampaignManager shuts down the shared
        engine via cm.close() after all replicas have finished.
        """
        pass

    # --------------------------------------------------------------------------
    async def stop(self):
        """Alias for close(), can be used for external termination."""
        await self.close()
