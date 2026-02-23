import asyncio
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from rose import Learner

import numpy as np
import yaml
from rose.metrics import MODEL_ACCURACY

from ddsim.ddsim_manager import DDSimManager

task_description = {"shell": True}


class DummyWorkflow(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, **kwargs):
        # Initialize parent class (sets up logger, queues, etc.)
        super().__init__()

        # Default home directory
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate DummyWorkflow w/o asyncflow")
        self.learner = Learner(self.flow)

        home_dir = Path(kwargs.get("home_dir", Path.home() / "DDSim"))
        self._clean_dir(home_dir)  # ❗Careful: deletes everything in home_dir!

        # Create workflow directories
        self.sim_output_dir = kwargs.get("sim_output_dir", home_dir / "sim_output")
        self.sim_output_dir = self._clean_dir(self.sim_output_dir)
        self.sim_output_dir = self._ensure_dir(
            kwargs.get("sim_output_dir", home_dir / "sim_output")
        )
        self.sim_inputs_dir = self._ensure_dir(
            kwargs.get("sim_inputs_dir", home_dir / "sim_input")
        )
        self.train_dir = self._ensure_dir(kwargs.get("train_dir", home_dir / "train"))
        self.train_al_dir = self._ensure_dir(
            kwargs.get("train_al_dir", home_dir / "train_al")
        )
        self.val_dir = self._ensure_dir(kwargs.get("val_dir", home_dir / "val"))

        # Simulation/training config
        # Max number of simulation to run at once
        self.max_sim_batch = kwargs.get("max_sim_batch", 4)
        # Number of cores reserved for training
        self.training_cores = kwargs.get("training_cores", 1)
        # Initial size of simulation batch before training starts
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        self.training_threshold = kwargs.get("training_threshold", 0.5)
        self.prediction_threshold = kwargs.get("prediction_threshold", 0.5)
        self.start_training_threshold = kwargs.get("start_training_threshold", 1)
        self.training_epochs = kwargs.get("training_epochs", 1)
        self.free_resources_for_train = bool(
            kwargs.get("free_resources_for_train", True)
        )

        self.iteration = 0
        self.retrain_model = self.training_epochs > 0

        # Paths for executables and model
        self.src_dir = kwargs.get("src_dir", os.getcwd())
        self.code_path = kwargs.get("code_path", f"{sys.executable} {self.src_dir}")
        self.model_filename = home_dir / "model.pkl"
        self.prediction_file = home_dir / "predictions.yml"  # fixed typo ("predicions")

        # Register learner tasks
        self.register_tasks()
        self.num_files = kwargs.get("num_files", 5)
        # Generate dummy input files
        self.sim_inputs = {}
        self._generate_sim_inputs(self.sim_inputs_dir, num_files=self.num_files)

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
    def _generate_sim_inputs(sim_inputs_dir, num_files: int = 5):
        """
        Generate dummy input `.npz` files for simulations.
        """
        sim_inputs_path = Path(sim_inputs_dir)
        for i in range(num_files):
            file_path = sim_inputs_path / f"config_{i}.npz"
            x = np.random.rand(100, 1)  # Dummy input data
            np.savez(file_path, X=x)

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        @self.learner.simulation_task()
        async def simulation(task_description=task_description, **kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            filename = self.sim_inputs[sim_idx]
            args = f"--output_dir {self.sim_output_dir} --sim_tag {sim_idx} --filename {filename}"
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

        @self.learner.prediction_task(as_executable=True)
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
        return kwargs["prediction"] < self.prediction_threshold

    # --------------------------------------------------------------------------
    async def run_inference(self) -> dict:
        await self.prediction()
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
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.logger.info(f"Re-added Sim {sim_idx} back the queue")
            if sim_idx not in self.sim_inputs:
                raise ValueError(f"Unable to add  sim {sim_idx} to queue ")

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
                self.logger.warning(f"File already removed: {file_path}")
            except Exception as e:
                self.logger.error(f"Error deleting {file_path}: {e}")

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
                        f"Simulation directory has been removed: {sim_dir}"
                    )
            else:
                self.logger.warning(f"Simulation directory already removed: {sim_dir}")
        except Exception as e:
            self.logger.error(f"Error deleting directory {sim_dir}: {e}")
        if self.debug:
            self.logger.info(f"Removed all files related to simulation {sim_idx}")

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
        self.iteration += 1
        for epoch in range(self.training_epochs):
            self.logger.info(
                f"Iteration {self.iteration} / Epoch {epoch + 1}", component="training"
            )

            train_task = await self.training()
            self.logger.task_started("Model Training", component="training")

            should_stop, metric_val = await self.check_accuracy()
            self.logger.task_completed("Model Training", component="training")
            self.logger.task_started("Check Accuracy", component="training")

            if should_stop:
                self.logger.info(
                    f"Accuracy ({metric_val}) reached threshold → stopping training"
                )
                self.retrain_model = False
                self.training_cores = 0
                break
            self.logger.task_completed("Check Accuracy", component="training")

            self.logger.task_started("Active Learning", component="training")
            await self.active_learn()
            self.logger.task_completed("Active Learning", component="training")

    # --------------------------------------------------------------------------
    async def post_process(self):
        if len(self.completed_sims) >= self.num_files:
            self.shutting_down.set()
            self.run_pipeline = False
            self.logger.info("All sim have completed...")

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner."""
        try:
            await self.learner.shutdown()
        except Exception:
            pass
        try:
            await self.flow.shutdown()
        except Exception:
            pass

    # --------------------------------------------------------------------------
    async def stop(self):
        """Alias for close(), can be used for external termination."""
        await self.close()
