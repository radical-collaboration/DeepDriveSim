import asyncio
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

from ddsim.ddsim_manager import DDSimManager

TASK_PRE_EXEC = [
    "module load anaconda",
    "source activate base",
    "conda activate /anvil/scratch/x-mgoliyad1/conda_env/rose_env",
    'export RADICAL_PROFILE="TRUE"',
]


class DDMdWorkflow(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, **kwargs):
        # Initialize parent class (sets up logger, queues, etc.)
        super().__init__()

        # Default home directory
        self.flow = kwargs.get("asyncflow", None)
        home_dir = Path(kwargs.get("home_dir", Path.home() / "DDMD"))
        self.clean_dir(home_dir)  # ❗Careful: deletes everything in home_dir!

        # Create output directoriy for simulations
        self.sim_output_dir = self._ensure_dir(
            kwargs.get("sim_output_dir", home_dir / "sim_output")
        )

        # Paths for executables and model
        self.code_path = kwargs.get("code_path", f"{sys.executable} {os.getcwd()}")
        # Prediction (agent) is running as executable for
        # miniapps and writes all scores to file
        self.prediction_file = home_dir / "predictions.yaml"

        # Simulation/training config
        #################################
        # Max number of simulation to run at once
        self.max_sim_batch = kwargs.get("max_sim_batch", 4)
        # Number of cores reserved for training
        self.training_cores = kwargs.get("training_cores", 1)
        # Initial size of simulation batch before training starts
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        # Set to True if training data is available at start
        self.force_start_training = bool(kwargs.get("force_start_training", False))
        # Stop pipeline after all simulation are done
        self.total_num_sim = kwargs.get("total_num_sim", 25)
        # Training iteration
        self.iteration = 0
        self.phase = kwargs.get("phase", 0)  # required by miniapps
        self.retrain_model = True  # Stop training model if accuracy is good
        self.sim_predictions = {}  # To Store predictions scores

        # Delete sims that are not running because they were completed or canceled
        self.clean_unregistered_sims = bool(
            kwargs.get("clean_unregistered_sims", False)
        )

        if self.device == "gpu":
            self.task_description = {
                "ranks": 1,
                "gpus_per_rank": 1,
                "pre_exec": TASK_PRE_EXEC,
            }
        else:
            self.task_description = {
                "ranks": 1,
                "cores_per_rank": 1,
                "pre_exec": TASK_PRE_EXEC,
            }

        # Register learner tasks
        self._register_learner_tasks()

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
    async def run_inference(self):
        with open(self.prediction_file) as f:
            predictions = yaml.safe_load(f)
        return predictions

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
        for s in range(self.total_num_sim):
            await self.sim_task_queue.put({"sim_tag": s})

    # --------------------------------------------------------------------------
    async def check_train_data(self):
        """Check if enough training data is available to start training."""

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            ranks = comm.Get_size()
        except Exception:
            ranks = 1

        root_path = Path(self.sim_output_dir, "phase0")
        filenames = [
            Path(root_path, f"data_{rank}_{self.iteration}.h5") for rank in range(ranks)
        ]

        print(f"Waiting for {len(filenames)} file to start training... ")
        start_trainig = False
        while True:
            if start_trainig:
                # await asyncio.sleep(5)
                break
            start_trainig = True
            for filename in filenames:
                if not filename.exists():
                    if self.debug:
                        print(f"File {filename} not found yet, wait...")
                    start_trainig = False
                    await asyncio.sleep(1)
                    break

        print("All required files are available. Starting training...")
        return True

    # --------------------------------------------------------------------------
    def _register_learner_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        @self.flow.function_task
        async def simulation(task_description=self.task_description, **kwargs):
            sim_tag = kwargs["sim_inputs"]["sim_tag"]
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {sim_tag} "
                f"--phase {self.phase} "
                f"--num_step 50 "
            )
            return f"{self.code_path}/simulation.py {args}"

        self.simulation = simulation

        @self.flow.function_task
        async def training(task_description=self.task_description):
            if len(self.completed_sims) > 0:
                sim_tag = list(self.completed_sims)[0]
            else:
                sim_tag = self.registered_sims.keys()[0]
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {sim_tag} "
                f"--phase {self.phase} "
                f"--num_epochs 1"
            )

            return f"{self.code_path}/training.py {args}"

        self.training = training

        @self.learner.prediction_task(as_executable=True)
        async def prediction(task_description=self.task_description):
            if len(self.completed_sims) > 0:
                sim_tag = list(self.completed_sims)[0]
            else:
                sim_tag = self.registered_sims.keys()[0]
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {sim_tag} "
                f"--phase {self.phase} "
                f"--num_epochs 1 "
                f" --num_mult_outlier 1 "
                f" --num_mult 1 "
                f"--output_file {self.prediction_file}"
            )

            return f"{self.code_path}/agent.py {args}"

        self.prediction = prediction

        @self.learner.utility_task(as_executable=True)
        async def selection(*args, **kwargs):
            """Dummy selection: assign random score to each sim."""
            if len(self.completed_sims) > 0:
                sim_tag = list(self.completed_sims)[0]
            else:
                sim_tag = self.registered_sims.keys()[0]
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {sim_tag} "
                f"--phase {self.phase}"
            )
            return f"{self.code_path}/selection.py {args}"

        self.selection = selection

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
        self.iteration += 1
        self.logger.info(f"\nTraining Iteration {self.iteration}")
        self.logger.info(f"{len(self.registered_sims)} simulation(s) running....")

        await self.training()
        self.logger.task_completed("Training Completed")
