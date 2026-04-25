import asyncio
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from rose import Learner
    from rose.metrics import MODEL_ACCURACY
except ModuleNotFoundError:
    Learner = None
    MODEL_ACCURACY = None


from ddsim.ddsim_manager import DDSimManager

task_description = {
    "shell": True,
}


class DummyWorkflow(DDSimManager):
    """Dummy workflow for managing simulations, training, and predictions."""

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
        # Callback injected by AsyncCampaignManager to unblock dependent workflows.
        self._on_ready = kwargs.get("on_ready", None)

        super().__init__(name=kwargs.get("name", "dummy_workflow"))

        self.debug = cfg.get("debug", False)
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate DummyWorkflow w/o asyncflow")
        self.learner = Learner(self.flow) if Learner is not None else None

        _home_base = Path(kwargs.get("home_dir", cfg.get("home_dir", Path.home() / "Dummy")))
        self.home_dir = self._ensure_dir(_home_base / self.name)
        self._clean_dir(self.home_dir)  # ❗Careful: deletes everything in home_dir!

        self.sim_output_dir = self._ensure_dir(self.home_dir / "sim_output")
        self.sim_inputs_dir = self._ensure_dir(self.home_dir / "sim_input")
        self.train_dir = self._ensure_dir(self.home_dir / "train")
        self.train_al_dir = self._ensure_dir(self.home_dir / "train_al")
        self.val_dir = self._ensure_dir(self.home_dir / "val")

        # kwargs take precedence over config dict (useful for tests/overrides).
        def _get(key, default, alias=None):
            if key in kwargs:
                return kwargs[key]
            if alias and alias in kwargs:
                return kwargs[alias]
            return cfg.get(key, cfg.get(alias, default) if alias else default)

        self.num_inputs = int(_get("num_inputs", 5, alias="num_files"))
        self.max_sim_batch = int(_get("max_sim_batch", 4))
        self.training_cores = int(_get("training_cores", 1))
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        self.training_threshold = float(_get("training_threshold", 0.5))
        self.prediction_threshold = float(_get("prediction_threshold", 0.5))
        self.start_training_threshold = int(_get("start_training_threshold", 1))
        self.training_epochs = int(_get("training_epochs", 1))
        self.free_resources_for_train = bool(_get("free_resources_for_train", True))
        self.force_start_training = bool(_get("force_start_training", False))
        # sleep_time should exceed a single simulation's runtime so the main
        # loop doesn't evaluate before the last sim has had a chance to finish.
        self.sleep_time = float(cfg.get("sleep_time", 30))

        self.call_cancel_simulations = True
        self.call_finalize_results = True
        self.call_evaluate_simulations = True

        self.iteration = 0
        self.retrain_model = self.training_epochs > 0

        # Signal dependent workflows after this many sims complete.
        # Defaults to num_inputs so the signal fires only when all sims finish.
        self.ddsim_data_ready = int(cfg.get("ddsim_data_ready", self.num_inputs))
        self._data_ready_signaled = False

        # Default src_dir to the workflow directory so scripts are found
        # regardless of the working directory.  Using `or` ensures an empty
        # string in config also falls back to the default.
        _default_src = str(Path(__file__).parent)
        self.src_dir = cfg.get("src_dir") or os.getenv("WORK_DIR", _default_src)

        # Python executable for all tasks; falls back to the current interpreter.
        self.executable = os.path.expandvars(cfg.get("executable") or sys.executable)

        # GPU/CPU affinity policy injected by AsyncCampaignManager.
        policies = kwargs.get("policies", None)
        if policies is not None:
            self.policy = policies[0] if policies else None
        else:
            self.policy = kwargs.get("policy", None)

        self.model_filename = self.home_dir / "model.pkl"
        self.prediction_file = self.home_dir / "predictions.yml"

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
            shutil.rmtree(dir_path, ignore_errors=True)

    # --------------------------------------------------------------------------
    @staticmethod
    def _generate_sim_inputs(sim_inputs_dir, num_inputs: int = 5):
        """
        Generate dummy input `.npz` files for simulations.
        """
        sim_inputs_path = Path(sim_inputs_dir)
        sim_inputs_path.mkdir(parents=True, exist_ok=True)
        for i in range(num_inputs):
            file_path = sim_inputs_path / f"config_{i}.npz"
            x = np.random.rand(100, 1)
            np.savez(file_path, X=x)

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""
        _task_desc = dict(task_description)
        if self.policy is not None:
            _task_desc["process_template"] = {"policy": self.policy}
            self.logger.info(
                f"Task policy: host={self.policy.host_name} "
                f"gpu_affinity={self.policy.gpu_affinity}",
                component=self.name,
            )
        else:
            self.logger.info("Task policy: none (no GPU affinity)", component=self.name)

        @self.flow.executable_task
        async def simulation(task_description=_task_desc, **kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            filename = self.sim_inputs[sim_idx]
            args = (
                f"--output_dir {self.sim_output_dir} --sim_tag {sim_idx} "
                f"--filename {filename}"
            )
            return f"{self.executable} {self.src_dir}/simulation.py {args}"

        self.simulation = simulation

        _training_dec = (
            self.learner.training_task() if self.learner else self.flow.executable_task
        )

        @_training_dec
        async def training(task_description=_task_desc, **kwargs):
            args = (
                f"--model_filename {self.model_filename} "
                f"--sim_output_dir {self.sim_output_dir} "
                f"--train_dir {self.train_al_dir} --val_dir {self.val_dir}"
            )
            return f"{self.executable} {self.src_dir}/train.py {args}"

        self.training = training

        _active_learn_dec = (
            self.learner.active_learn_task()
            if self.learner
            else self.flow.executable_task
        )

        @_active_learn_dec
        async def active_learn(task_description=_task_desc, **kwargs):
            args = (
                f"--model_filename {self.model_filename} "
                f"--train_dir {self.train_dir} "
                f"--train_al_dir {self.train_al_dir}"
            )
            return f"{self.executable} {self.src_dir}/active_learn.py {args}"

        self.active_learn = active_learn

        @self.flow.executable_task
        async def prediction(task_description=_task_desc, **kwargs):
            args = (
                f"--model_filename {self.model_filename} "
                f"--sim_output_dir {self.sim_output_dir} "
                f"--output_file {self.prediction_file}"
            )
            return f"{self.executable} {self.src_dir}/predict.py {args}"

        self.prediction = prediction

        _accuracy_dec = (
            self.learner.as_stop_criterion(
                metric_name=MODEL_ACCURACY, threshold=self.training_threshold
            )
            if self.learner
            else self.flow.executable_task
        )

        @_accuracy_dec
        async def check_accuracy(task_description=_task_desc, **kwargs):
            args = f"--model_filename {self.model_filename} --val_dir {self.val_dir}"
            return f"{self.executable} {self.src_dir}/check_accuracy.py {args}"

        self.check_accuracy = check_accuracy

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs) -> bool:
        """Return True if prediction < threshold (cancel simulation)."""
        if self.debug:
            self.logger.info(
                f"Prediction is {kwargs['prediction']}", component=self.name
            )
        return kwargs["prediction"] < self.prediction_threshold

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self) -> dict:
        if self.debug:
            self.logger.task_started("Model Prediction", component="prediction")
        await self.prediction()
        if self.debug:
            self.logger.task_completed("Model Prediction", component="prediction")
        try:
            with open(self.prediction_file) as f:
                predictions = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            predictions = {}

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
            self.logger.info(
                f"Re-added Sim {sim_idx} back to queue", component=self.name
            )

    # --------------------------------------------------------------------------
    async def check_train_status(self) -> bool:
        """
        Return True when enough simulation outputs have
        accumulated to start training.
        """
        outputs = list(self.sim_output_dir.iterdir())
        return len(outputs) >= self.start_training_threshold

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
                self.logger.warning(
                    f"File already removed: {file_path}", component=self.name
                )
            except Exception as e:
                self.logger.error(
                    f"Error deleting {file_path}: {e}", component=self.name
                )

        tasks = []
        for directory in [self.train_al_dir, self.train_dir, self.val_dir]:
            for filename in directory.iterdir():
                if sim_idx in filename.name:
                    tasks.append(_delete_file(filename))
        if tasks:
            await asyncio.gather(*tasks)

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
                        f"Simulation directory already removed: {sim_dir}",
                        component=self.name,
                    )
        except Exception as e:
            self.logger.error(
                f"Error deleting directory {sim_dir}: {e}", component=self.name
            )
        if self.debug:
            self.logger.info(
                f"Removed all files related to simulation {sim_idx}",
                component=self.name,
            )

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
        # Skip training if all sims are already done; no new data will appear.
        if len(self.completed_sims) >= self.num_inputs:
            return

        self.iteration += 1
        for epoch in range(self.training_epochs):
            if self.debug:
                self.logger.info(
                    f"Iteration {self.iteration} / Epoch {epoch + 1}",
                    component=self.name,
                )

            if self.debug:
                self.logger.task_started("Model Training", component=self.name)

            await self.training()

            if self.debug:
                self.logger.task_completed("Model Training", component=self.name)
                self.logger.task_started("Check Accuracy", component=self.name)

            result = await self.check_accuracy()
            # learner.as_stop_criterion returns (bool, float); fall back to no-stop
            # when rose is not installed and the task returns None.
            try:
                should_stop, metric_val = result
            except (TypeError, ValueError):
                should_stop, metric_val = False, 0.0

            if should_stop:
                if self.debug:
                    self.logger.info(
                        f"Accuracy ({metric_val}) reached threshold"
                        " → stopping training",
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
                self.logger.warning(
                    f"All sim have completed BUT there are still registered sims"
                    f" {self.registered_sims.keys()}",
                    component=self.name,
                )
            else:
                self.shutting_down.set()
                self.run_workflow = False
                self.logger.info("All sim have completed...", component=self.name)
        else:
            if self.debug:
                self.logger.warning(
                    f" {n_done} sim have completed out of {self.num_inputs}",
                    component=self.name,
                )

    # --------------------------------------------------------------------------
    async def close(self):
        # Temporary cleanup to test campaign manager and aid Disk quota exceeded.
        if self.home_dir.exists():
            shutil.rmtree(self.home_dir, ignore_errors=True)
            self.logger.info(
                f"Removed home directory: {self.home_dir}", component=self.name
            )

    # --------------------------------------------------------------------------
    async def stop(self):
        """Alias for close(), can be used for external termination."""
        await self.close()
