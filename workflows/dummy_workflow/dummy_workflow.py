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

        _home_base = Path(
            kwargs.get("home_dir", cfg.get("home_dir", Path.home() / "Dummy"))
        )
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
        self.call_evaluate_simulations = True
        self.call_finalize_results = True
        # call_post_process_sim controls whether _on_sim_done() fires
        # post_process_sim(sim_idx) immediately after each sim completes.
        # It is kept False here because evaluate_simulations() (predict.py)
        # still needs the sim output files in the same iteration — enabling it
        # would race with predict.py, deleting files it is trying to read.
        # Set to True only if your workflow does NOT use evaluate_simulations,
        # or if you move cleanup to finalize_results / close() instead.
        self.call_post_process_sim = False

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

        # ── Secondary parallel training tasks (train_models) ──────────────────
        # DDSimManager runs everything in self.train_models concurrently with
        # train_model() on every training tick:
        #
        #   await asyncio.gather(train_model(), *(t() for t in self.train_models))
        #
        # Use this list to attach auxiliary steps that should run alongside
        # primary training — for example a separate validation pass, a data
        # pre-processing job, or a secondary model on a different feature set.
        # Each entry must be a zero-argument async callable.
        #
        # Example — register a standalone validation task and add it:
        #
        #   @self.flow.executable_task
        #   async def validate(task_description=_task_desc, **kwargs):
        #       args = (
        #           f"--model_filename {self.model_filename} "
        #           f"--val_dir {self.val_dir}"
        #       )
        #       return f"{self.executable} {self.src_dir}/validate.py {args}"
        #
        #   self.train_models.append(validate)
        #
        # Leave the list empty (the default) to skip secondary training entirely.

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
        """Generate sim inputs (in a thread) then queue them for submission."""
        await asyncio.to_thread(
            self._generate_sim_inputs, self.sim_inputs_dir, self.num_inputs
        )
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
        Return True when training should begin.

        Two paths:
        - Normal: wait until at least `start_training_threshold` sim outputs
          have accumulated (data-driven gate).
        - Force: if `force_start_training` is True, bypass the data gate and
          start training immediately regardless of how many outputs exist.
          Useful when you want the first training round to start as soon as
          any sim finishes, e.g. to warm-start the model early.
        """
        if self.force_start_training:
            return True
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

        # ── Inter-iteration queue management ─────────────────────────────────────
        # This block runs after every training cycle but before the campaign
        # decides whether to shut down.  Use it to reshape the pending work queue
        # so the next simulation batch reflects the latest model knowledge.
        #
        # Two complementary operations are shown:
        #   1. PRUNE  — drain the queue and drop inputs the model predicts are
        #               unlikely to pass the scoring threshold (saves GPU time).
        #   2. ENRICH — add new candidate inputs derived from training output
        #               (active-learning or model-guided generation).
        #
        # Both are no-ops when predictions are unavailable or the queue is empty,
        # so this block is safe to leave in place even in stub/test runs.

        predictions = getattr(self, "sim_predictions", {})

        # 1. PRUNE: remove low-confidence candidates from the pending queue.
        #    Drain all items, keep only those whose prediction clears the
        #    threshold (or those not yet scored — give them the benefit of doubt).
        if predictions and not self.sim_task_queue.empty():
            kept = []
            while not self.sim_task_queue.empty():
                item = self.sim_task_queue.get_nowait()
                if item is None:
                    # Sentinel — will be re-inserted after pruning.
                    continue
                sim_idx = item.get("sim_idx")
                score = predictions.get(sim_idx)
                if score is None or score >= self.prediction_threshold:
                    kept.append(item)
                else:
                    self.logger.info(
                        f"Pruning sim {sim_idx} (predicted score {score:.3f} "
                        f"< threshold {self.prediction_threshold})",
                        component=self.name,
                    )
            for item in kept:
                await self.sim_task_queue.put(item)
            # Restore the sentinel so the consumer knows the queue end.
            await self.sim_task_queue.put(None)

        # 2. ENRICH: inject new candidate inputs for the next iteration.
        #    In a real campaign these would come from model-guided generation,
        #    a screening library, or an external oracle.  Here we show the
        #    pattern — replace `new_candidates` with your actual source.
        #
        #    new_candidates = generate_candidates(self.model_filename)
        #    for candidate in new_candidates:
        #        sim_idx = candidate["id"]
        #        self.sim_inputs[sim_idx] = candidate["path"]
        #        await self.sim_task_queue.put({"sim_idx": sim_idx})
        #        self.num_inputs += 1   # extend the completion target accordingly
        # ── End inter-iteration queue management ──────────────────────────────

        if n_done >= self.num_inputs:
            # Defensive cleanup: any sims still registered here are perm-cancelled
            # tasks whose done-callbacks haven't fired yet.  cancel_sims() now
            # awaits cancelled tasks before returning, so this branch should not
            # normally be reached.  If it is (e.g. a non-prediction cancel path
            # left stragglers), cancel them and wait rather than just warning.
            if self.registered_sims:
                straggler_ids = list(self.registered_sims.keys())
                for sim_idx in straggler_ids:
                    self._perm_cancelled.add(sim_idx)
                    task = self.registered_sims[sim_idx]
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    *self.registered_sims.values(), return_exceptions=True
                )
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
