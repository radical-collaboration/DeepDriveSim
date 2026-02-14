import asyncio
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

from ddsim.ddsim_manager import DDSimManager

from deepdrivemd.config import ExperimentConfig  #BaseStageConfig, 
from deepdrivemd.data.api import DeepDriveMD_API


TASK_PRE_EXEC = [
    "module load anaconda",
    "source activate base",
    "conda activate /anvil/scratch/x-mgoliyad1/conda_env/rose_env",
    'export RADICAL_PROFILE="TRUE"',
]


class DDMdWorkflow(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, config, **kwargs):
        # Initialize parent class (sets up logger, queues, etc.)
        super().__init__()

        cfg = ExperimentConfig.from_yaml(config)
        self.api = DeepDriveMD_API(cfg.experiment_directory)

        # Default home directory
        self.flow = kwargs.get("asyncflow", None)

        self.stage_idx = 0

        self._init_experiment_dir()

        self.sim_task_description = self._generate_task_description(cfg.molecular_dynamics_stage)
        self.training_task_description = self._generate_task_description(cfg.machine_learning_stage)
        self.prediction_task_description = self._generate_task_description(cfg.agent_stage)
        self.aggregation_task_description = self. _generate_task_description(cfg.aggregation_stage)
        self.selection_task_description = self. _generate_task_description(cfg.model_selection_stage)

        # Back up configuration file (PipelineManager must create cfg.experiment_dir)
        shutil.copy(config, cfg.experiment_directory)
            


        # Simulation/training config
        #################################
        # Setting simulation resource req
        sim_cpus = cfg.molecular_dynamics_stage.cpu_reqs.processes
        sim_gpus = cfg.molecular_dynamics_stage.gpu_reqs.processes

        max_resource = max(sim_cpus, sim_gpus)

        self.max_sim_batch = cfg.molecular_dynamics_stage.num_tasks * max_resource
        # Setting training resource req
        sim_cpus = cfg.machine_learning_stage:.cpu_reqs.processes
        sim_gpus = cfg.machine_learning_stage.gpu_reqs.processes

        max_resource = max(sim_cpus, sim_gpus)
        self.training_cores = max_resource

        # Initial size of simulation batch before training starts
        self.sim_batch_size = self.max_sim_batch

        # Set to True if training data is available at start
        self.force_start_training = bool(kwargs.get("force_start_training", False))

        # Training iteration
        self.iteration = 0

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

        # Register asyncflow tasks
        self._register_tasks()

    def _init_experiment_dir(self) -> None:
        # Make experiment directories
        self.cfg.experiment_directory.mkdir(parents=True, exist_ok=True)
        self.api.molecular_dynamics_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.aggregation_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.machine_learning_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.model_selection_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.agent_stage.runs_dir.mkdir(parents=True, exist_ok=True)

    # # --------------------------------------------------------------------------
    # @staticmethod
    # def _ensure_dir(path):
    #     """Create directory if it does not exist."""
    #     path = Path(path)
    #     path.mkdir(parents=True, exist_ok=True)
    #     return path

    # # --------------------------------------------------------------------------
    # @staticmethod
    # def clean_dir(dir_name):
    #     """Delete an existing directory (used for a clean workflow run)."""
    #     dir_path = Path(dir_name)
    #     if dir_path.exists() and dir_path.is_dir():
    #         shutil.rmtree(dir_path)

    def _generate_task_description(self, config):

        task_description = {
                "ranks": 1,
                "cores_per_rank": config.cpu_reqs,
                "gpus_per_rank": config.gpu_reqs,
                "pre_exec": config.pre_exec,
            }

    # --------------------------------------------------------------------------
    async def run_inference(self):
        with open(self.prediction_file) as f:
            predictions = yaml.safe_load(f)
        return predictions

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
       return False

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Collect all simulation input files into task queue."""
        cfg = self.config.molecular_dynamics_stage
        num_sim = cfg.num_tasks
        initial_pdbs = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
        filenames: Optional[itertools.cycle[Path]] = itertools.cycle(initial_pdbs)
        for task_idx in range(num_sim):
            next_file = next(filenames)
            await self.sim_task_queue.put({"sim_tag": {task_idx: next_file}})


    async def post_processed(self):
        for task_idx in range(num_sim):
            await self.sim_task_queue.put({"sim_tag": {task_idx: None}})


    # --------------------------------------------------------------------------
    async def check_train_data(self):
        """Check if enough training data is available to start training."""
        return True

    # --------------------------------------------------------------------------
    def _register_tasks(self):
        """Register asyncflow tasks: simulation, training, active learning, prediction."""

        @self.flow.function_task
        async def simulation(task_description=self.sim_task_description, **kwargs):
            
            cfg = self.config.molecular_dynamics_stage
            sim_tag = kwargs["sim_inputs"]["sim_tag"]

            task_idx, cfg.task_config.pdb_file = next(iter(sim_tag.items()))
            cfg = self.config.molecular_dynamics_stage
            stage_api = self.api.molecular_dynamics_stage

            if self.stage_idx == 0:
                initial_pdbs = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
                filenames: Optional[itertools.cycle[Path]] = itertools.cycle(initial_pdbs)
            else:
                filenames = None

            #for task_idx in range(cfg.num_tasks):

            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path
            # if self.stage_idx == 0:
            #     assert filenames is not None
            #     cfg.task_config.pdb_file = next(filenames)
            # else:
            #     cfg.task_config.pdb_file = None

            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)
            # task = generate_task(cfg)
            # task.arguments += ["-c", cfg_path.as_posix()]
            # stage.add_tasks(task)

            
                
            return f"{cfg.executable} {cfg.arguments} -c {cfg_path.as_posix()} "

        self.simulation = simulation

        @self.flow.function_task
        async def aggregation(task_description=self.aggregation_task_description):
            cfg = self.cfg.aggregation_stage
            stage_api = self.api.aggregation_stage

            task_idx = 0
            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path

            # Write yaml configuration
            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)
            return f"{cfg.executable} {cfg.arguments} -c {cfg_path.as_posix()} "

        @self.flow.function_task
        async def training(task_description=self.training_task_description):
            cfg = self.cfg.machine_learning_stage
            stage_api = self.api.machine_learning_stage

            task_idx = 0
            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path
            cfg.task_config.model_tag = stage_api.unique_name(output_path)
            if self.stage_idx > 0:
                # Machine learning should use model selection API
                cfg.task_config.init_weights_path = None

            # Write yaml configuration
            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)

            return f"{cfg.executable} {cfg.arguments} -c {cfg_path.as_posix()} "

        self.training = training

        @self.flow.function_task
        async def prediction(task_description=self.prediction_task_description):
            cfg = self.cfg.agent_stage
            stage_api = self.api.agent_stage

            task_idx = 0
            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path

            # Write yaml configuration
            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)

            return f"{cfg.executable} {cfg.arguments} -c {cfg_path.as_posix()} "

        self.prediction = prediction

        @self.flow.function_task
        async def selection(task_description=self.selection_task_description):
            cfg = self.cfg.model_selection_stage
            stage_api = self.api.model_selection_stage

            task_idx = 0
            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path

            # Write yaml configuration
            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)


            return f"{cfg.executable} {cfg.arguments} -c {cfg_path.as_posix()} "

        self.selection = selection

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
        self.iteration += 1
        self.logger.info(f"\nTraining Iteration {self.iteration}")
        self.logger.info(f"{len(self.registered_sims)} simulation(s) running....")

        await self.training()
        self.logger.task_completed("Training Completed")
