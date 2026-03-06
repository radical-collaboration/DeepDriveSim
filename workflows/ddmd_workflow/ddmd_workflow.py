import itertools
import shutil

from ddsim.ddsim_manager import DDSimManager
from workflows.ddmd_workflow.config import ExperimentConfig
from workflows.ddmd_workflow.data.api import DeepDriveMD_API


class DDMdWorkflow(DDSimManager):
    """DeepDriveMD workflow: orchestrates MD simulations, ML training,
    aggregation, model selection, and agent-based inference stages.

    Extends DDSimManager to implement the full DeepDriveMD workflow
    using asyncflow for task scheduling and execution.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class (sets up logger, queues, event flags, etc.)
        resource_manager = kwargs.get("resource_manager", None)
        super().__init__(resource_manager=resource_manager)

        # Asyncflow engine for registering and dispatching tasks
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate DDMdWorkflow w/o asyncflow")

        config = kwargs.get("config")

        # Load and validate experiment configuration from YAML
        self.experiment_config = ExperimentConfig.from_yaml(config)

        agg_stage = self.experiment_config.aggregation_stage
        self.skip_aggregation = agg_stage.skip_aggregation
        self.api = DeepDriveMD_API(self.experiment_config.experiment_directory)

        self.tasks_config = {
            "simulation": {
                "priority": 10,
                "ranks": 1,
                "cores_per_rank": 1,
                "gpus_per_rank": 0.5,
            },
            "train_model": {
                "priority": 10,
                "ranks": 1,
                "cores_per_rank": 1,
                "gpus_per_rank": 1,
            },
            "selection": {
                "priority": 10,
                "ranks": 1,
                "cores_per_rank": 1,
                "gpus_per_rank": 0,
                "on_completion": "inference",
            },
            "aggregation": {
                "priority": 10,
                "ranks": 1,
                "cores_per_rank": 1,
                "gpus_per_rank": 0,
                "on_completion": "inference",
            },
            "inference": {
                "priority": 10,
                "ranks": 1,
                "cores_per_rank": 1,
                "gpus_per_rank": 0,
            },
            "finalize_results": {
                "priority": 10,
                "ranks": 1,
                "cores_per_rank": 1,
                "gpus_per_rank": 0,
            },
        }

        self.workflow_id = "ddsim_workflow"
        # Stage index tracks the current DeepDriveMD iteration (0-based)
        self.stage_idx = 0

        self._init_experiment_dir()
        self.task_descriptions = self._generate_task_descriptions()
        self.stage_config = self._generate_stage_config()

        # Back up YAML config into the experiment directory for reproducibility
        shutil.copy(config, self.experiment_config.experiment_directory)

        # Number of parallel simulation tasks per iteration
        self.num_sims = self.experiment_config.molecular_dynamics_stage.num_tasks
        self.max_sim_batch = self.num_sims

        # Batch size equals max since no resource sharing between sim and train
        self.sim_batch_size = self.max_sim_batch

        # No resource freeing needed: sims and training don't share resources
        self.free_resources_for_train = False
        # By default, do not call cancel_sims() after inference
        self.call_cancel_simulations = True
        self.call_finalize_results = True
        self.call_evaluate_simulations = True

        # Number of cores to free for training (used by monitor_training_data)
        self.training_cores = 1

        # Training iteration counter (incremented each train_model call)
        self.iteration = 0

        # If False, skip retraining (e.g. when model accuracy is sufficient)
        self.retrain_model = True

        # Register simulation, training, aggregation, inference, selection tasks
        self.register_tasks()
        # Dict to store inputs for simulation
        self.sim_inputs = {}
        self.train_models = []

    # --------------------------------------------------------------------------
    def _generate_stage_config(self):
        """Initialize each stage's task_config with shared experiment settings.

        Sets experiment_directory and node_local_path on every stage's
        task_config, then returns all configs in a dict keyed by stage name.
        """
        stage_config = {}

        cfg = self.experiment_config.molecular_dynamics_stage
        self._init_update_config(cfg)
        stage_config["molecular_dynamics_stage"] = cfg

        cfg = self.experiment_config.machine_learning_stage
        self._init_update_config(cfg)
        stage_config["machine_learning_stage"] = cfg

        cfg = self.experiment_config.aggregation_stage
        self._init_update_config(cfg)
        stage_config["aggregation_stage"] = cfg

        cfg = self.experiment_config.agent_stage
        self._init_update_config(cfg)
        stage_config["agent_stage"] = cfg

        cfg = self.experiment_config.model_selection_stage
        self._init_update_config(cfg)
        stage_config["model_selection_stage"] = cfg

        return stage_config

    # --------------------------------------------------------------------------
    def _generate_task_descriptions(self):
        """Build resource-requirement dicts for each workflow stage.

        Returns a dict keyed by stage name, each value describing
        ranks, cores, GPUs, and pre-exec commands for that stage.
        """
        task_descriptions = {}
        task_descriptions["molecular_dynamics_stage"] = self._generate_task_description(
            self.experiment_config.molecular_dynamics_stage
        )
        task_descriptions["machine_learning_stage"] = self._generate_task_description(
            self.experiment_config.machine_learning_stage
        )
        task_descriptions["aggregation_stage"] = self._generate_task_description(
            self.experiment_config.aggregation_stage
        )
        task_descriptions["agent_stage"] = self._generate_task_description(
            self.experiment_config.agent_stage
        )
        task_descriptions["model_selection_stage"] = self._generate_task_description(
            self.experiment_config.model_selection_stage
        )
        return task_descriptions

    # --------------------------------------------------------------------------
    def _init_experiment_dir(self) -> None:
        """Create the experiment directory tree for all workflow stages."""
        self.experiment_config.experiment_directory.mkdir(parents=True, exist_ok=True)
        self.api.molecular_dynamics_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.aggregation_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.machine_learning_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.model_selection_stage.runs_dir.mkdir(parents=True, exist_ok=True)
        self.api.agent_stage.runs_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    def _generate_task_description(self, config):
        """Build a single task resource description from a stage config."""
        task_description = {
            "ranks": 1,
            "cores_per_rank": config.cpu_reqs,
            "gpus_per_rank": config.gpu_reqs,
            "pre_exec": config.pre_exec,
            "shell": True,
        }
        return task_description

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Populate the simulation task queue with initial PDB inputs.

        Cycles through available PDB files so that each simulation task
        gets an input structure. If there are more tasks than PDB files,
        files are reused in round-robin order.
        """
        cfg = self.experiment_config.molecular_dynamics_stage
        initial_pdbs = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
        if not initial_pdbs:
            raise FileNotFoundError(
                f"No PDB files found in '{cfg.task_config.initial_pdb_dir}'. "
                f"Expected PDB files matching pattern '*/*.pdb' "
                f"(files must be inside subdirectories)."
            )
        filenames = itertools.cycle(initial_pdbs)
        for sim_idx in range(self.num_sims):
            next_file = next(filenames)
            # Key must be "sim_idx" to match parent's submit_sims()
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.sim_inputs[sim_idx] = next_file

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """Decide whether to cancel a simulation based on prediction.

        Returns False by default (no early cancellation).
        This workflow designed to replicate original ENTK workflow
        """
        return False

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        """Advance to next iteration or signal shutdown if max reached.

        Increments stage_idx and queues the next batch of simulation
        tasks (with pdb_file=None, so the simulation task will use
        restart PDBs from the previous iteration's predictions).
        """
        self.stage_idx += 1
        if self.stage_idx == self.experiment_config.max_iteration:
            self.shutting_down.set()
            self.run_workflow = False
            return

        for sim_idx in range(self.num_sims):
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.sim_inputs[sim_idx] = None

    # --------------------------------------------------------------------------
    async def check_train_status(self):
        """Check if all simulations for the current iteration have completed.

        Training starts only when all num_sims tasks across all iterations
        up to and including the current stage_idx have finished.
        Called by parent's monitor_training_data() and start() loop.
        """
        return len(self.completed_sims) == self.num_sims * (self.stage_idx + 1)

    # --------------------------------------------------------------------------
    def _init_update_config(self, cfg):
        """Set shared experiment-level fields on a stage's task_config."""
        exp = self.experiment_config
        cfg.task_config.experiment_directory = exp.experiment_directory
        cfg.task_config.node_local_path = exp.node_local_path

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """Register all workflow stages as asyncflow executable tasks.

        Each task builds a shell command string from its stage config
        and returns it for execution by the asyncflow backend.
        Tasks registered: simulation, aggregation, training,
        agent (inference), and model selection.
        """

        def _update_stage_config(stage_api, cfg, task_idx=0):
            """Write per-task YAML config and build the shell command.

            Creates the output directory, updates stage/task indices,
            dumps the config YAML, and returns (output_path, command_str).
            """
            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.output_path = output_path
            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)
            str_argument = " ".join(str(argument) for argument in cfg.arguments)
            cmd = f"{cfg.executable} {str_argument} -c {cfg_path.as_posix()} "
            return output_path, cmd

        stage_config = self.stage_config
        api = self.api

        # --- Simulation task: runs MD for each input PDB ---
        task_description = self.task_descriptions["molecular_dynamics_stage"]

        @self.flow.executable_task
        async def simulation(task_description=task_description, **kwargs):
            cfg = stage_config["molecular_dynamics_stage"]
            # Extract sim index and PDB file from the queued sim_idx
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            cfg.task_config.pdb_file = self.sim_inputs[sim_idx]
            stage_api = api.molecular_dynamics_stage
            _, cmd = _update_stage_config(stage_api, cfg, sim_idx)
            return cmd

        self.simulation = simulation

        # --- Aggregation task: combines MD outputs (optional) ---
        if not self.skip_aggregation:
            task_description = self.task_descriptions["aggregation_stage"]

            @self.flow.executable_task
            async def aggregation(task_description=task_description):
                cfg = stage_config["aggregation_stage"]
                stage_api = api.aggregation_stage
                _, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
                return cmd

            self.aggregation = aggregation
        else:
            self.aggregation = None

        # --- Training task: trains ML model on aggregated data ---
        task_description = self.task_descriptions["machine_learning_stage"]

        @self.flow.executable_task
        async def training(task_description=task_description):
            cfg = stage_config["machine_learning_stage"]
            stage_api = api.machine_learning_stage
            output_path, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            cfg.task_config.model_tag = stage_api.unique_name(output_path)
            if self.stage_idx > 0:
                # After first iteration, use model selection instead of init weights
                cfg.task_config.init_weights_path = None
            return cmd

        self.training = training

        # --- Agent task: runs inference/active learning ---
        task_description = self.task_descriptions["agent_stage"]

        @self.flow.executable_task
        async def agent_stage(task_description=task_description):
            cfg = stage_config["agent_stage"]
            stage_api = api.agent_stage
            _, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            return cmd

        self.evaluate_simulations = agent_stage

        # --- Model selection task: picks best model checkpoint ---
        task_description = self.task_descriptions["model_selection_stage"]

        @self.flow.executable_task
        async def selection(task_description=task_description):
            cfg = stage_config["model_selection_stage"]
            stage_api = api.model_selection_stage
            _, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            return cmd

        self.selection = selection

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Execute one training iteration: aggregate, train, select best model.

        Runs aggregation (if enabled), then ML training, then model selection.
        Called by the parent's start() loop after check_train_status() is True.
        """
        self.iteration += 1
        self.logger.task_started(f"Iteration {self.iteration}", component="training")
        self.logger.info(f"{len(self.registered_sims)} simulation(s) running....")

        if self.aggregation:
            await self.aggregation()

        await self.training()
        self.logger.task_completed(f"Iteration {self.iteration}", component="training")

        await self.selection()

    # --------------------------------------------------------------------------
    async def post_process_sim(self, sim_idx):
        del self.sim_inputs[sim_idx]

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shut down the asyncflow engine."""
        await self.flow.shutdown()
