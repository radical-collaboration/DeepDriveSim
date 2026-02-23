import asyncio
import os
import random
import shutil
import sys
from pathlib import Path

from rose import Learner

import yaml

from ddsim.ddsim_manager import DDSimManager

TASK_PRE_EXEC = [
    "module load anaconda3",
    "source activate base",
    "conda activate /anvil/scratch/x-mgoliyad1/conda_env/rose_env",
    'export RADICAL_PROFILE="TRUE"',
]


class MiniAppsWorkflow(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, **kwargs):
        # Initialize parent class (sets up logger, queues, etc.)
        super().__init__()

        self.flow = kwargs.get("asyncflow", None)
        self.learner = Learner(self.flow)

        # Default home directory
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
        self.free_resources_for_train = bool(
            kwargs.get("free_resources_for_train", True)
        )
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

        self.task_description = {
            "ranks": 1,
             "cores_per_rank": 1,
            "gpus_per_rank": 1,
            "pre_exec": TASK_PRE_EXEC,
            "shell": True,
        }

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
    async def run_inference(self):
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
            self.logger.info(f"Re-added Sim {sim_idx} back the queue")
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
        filenames = [
            Path(root_path, f"data_{rank}_{self.iteration}.h5") for rank in range(ranks)
        ]

        self.logger.info(f"Waiting for {len(filenames)} file to start training... ")
        start_trainig = False
        while True:
            if start_trainig:
                # await asyncio.sleep(5)
                break
            start_trainig = True
            for filename in filenames:
                if not filename.exists():
                    if self.debug:
                        self.logger.info(f"File {filename} not found yet, wait...")
                    start_trainig = False
                    await asyncio.sleep(1)
                    break

        self.logger.info("All required files are available. Starting training...")
        return True

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        @self.learner.simulation_task
        async def simulation(task_description=self.task_description, **kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            instance_index = self.sim_inputs[sim_idx]
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {instance_index} "
                f"--phase {self.phase} "
                f"--num_step 50 "
            )
            return f"{self.code_path}/simulation.py {args}"

        self.simulation = simulation

        @self.learner.training_task
        async def training(task_description=self.task_description):
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {self.iteration} "
                f"--phase {self.phase} "
                f"--num_epochs 1"
            )

            return f"{self.code_path}/training.py {args}"

        self.training = training

        @self.learner.prediction_task(as_executable=True)
        async def prediction(task_description=self.task_description):
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {self.iteration} "
                f"--phase {self.phase} "
                f"--num_epochs 1 "
                f" --num_mult_outlier 1 "
                f" --num_mult 1 "
                f"--output_file {self.prediction_file}"
            )

            return f"{self.code_path}/agent.py {args}"

        self.prediction = prediction

        @self.learner.utility_task(as_executable=False)
        async def selection(*args, **kwargs):
            """Dummy selection: assign random score to each sim."""
            args = (
                f"--data_root_dir {self.sim_output_dir} "
                f"--instance_index {self.iteration} "
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

    # --------------------------------------------------------------------------
    async def post_process(self):
        if len(self.completed_sims) >= self.total_num_sim:
            self.shutting_down.set()
            self.run_pipeline = False
            self.logger.info("All sim have completed...")

    async def post_process_sim(self, sim_idx, ):
        pass
    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner."""
        try:
            await self.learner.shutdown()
            await self.flow.shutdown()
        except Exception:
            pass

    async def stop(self):
        """Alias for close(), can be used for external termination."""
        try:
            await self.learner.shutdown()
            await self.flow.shutdown()
        except Exception:
            pass
