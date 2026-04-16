import asyncio
import itertools
import os
import shutil
from pathlib import Path

import yaml

from ddsim.ddsim_manager import DDSimManager
from workflows.ddmd_workflow.config import ExperimentConfig
from workflows.ddmd_workflow.data.api import DeepDriveMD_API


def _load_camp_config(config_file) -> dict:
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw.items()}
    return {}


class DDMdWorkflow(DDSimManager):
    """DeepDriveMD workflow: orchestrates MD simulations, ML training,
    aggregation, model selection, and agent-based inference stages.

    Extends DDSimManager to implement the full DeepDriveMD workflow
    using asyncflow for task scheduling and execution.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class (sets up logger, queues, event flags, etc.)
        super().__init__(name=kwargs.get("name", "ddsim"))

        # on_ready: async callable injected by AsyncCampaignManager so that
        # _signal_ready() can unblock dependent workflow groups (e.g. inference).
        self._on_ready = kwargs.get("on_ready", None)
        self._data_ready_signaled = False

        # Asyncflow engine for registering and dispatching tasks
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate DDMdWorkflow w/o asyncflow")

        # One Dragon Policy per assigned GPU (injected by AsyncCampaignManager).
        # The CM passes a single compound policy with gpu_affinity=[gpu0, gpu1, ...].
        # Expand it here into per-GPU policies so tasks can cycle round-robin;
        # a standalone caller may already pass N per-GPU policies, which we keep as-is.
        _raw_policies = kwargs.get("policies", []) or []
        if (len(_raw_policies) == 1
                and len(getattr(_raw_policies[0], "gpu_affinity", [])) > 1):
            try:
                from dragon.infrastructure.policy import Policy as _Policy
                _p = _raw_policies[0]
                self.policies = [
                    _Policy(
                        placement=_p.placement,
                        host_name=_p.host_name,
                        gpu_affinity=[gid],
                    )
                    for gid in _p.gpu_affinity
                ]
            except Exception:
                self.policies = _raw_policies
        else:
            self.policies = _raw_policies

        # Load camp-level config (config.yaml) and extract workflow parameters.
        # Falls back to direct kwargs for callers that don't use a camp config file.
        camp_cfg = _load_camp_config(kwargs["camp_config"]) if "camp_config" in kwargs else {}
        _here = Path(__file__).parent
        config = camp_cfg.get("ddsim_config") or kwargs.get("config")
        self.debug = camp_cfg.get("debug", kwargs.get("debug", False))
        self.tf_gpu_wrapper = (
            camp_cfg.get("tf_gpu_wrapper") or kwargs.get("tf_gpu_wrapper")
            or str(_here / "tf_gpu_wrapper.sh")
        )
        self.tf_cpu_wrapper = (
            camp_cfg.get("tf_cpu_wrapper") or kwargs.get("tf_cpu_wrapper")
            or str(_here / "tf_cpu_wrapper.sh")
        )

        # Load and validate experiment configuration from YAML.
        # When running multiple replicas they all share the same ddsim_config YAML,
        # so experiment_directory must be unique per replica to pass the validator
        # (which rejects pre-existing directories) and avoid data collisions.
        # We load the raw YAML, suffix experiment_directory with the replica name,
        # then construct ExperimentConfig directly — same as from_yaml() but patched.
        with open(config) as _fp:
            _raw = yaml.safe_load(_fp)
        for _k, _v in _raw.items():
            if isinstance(_v, str):
                _raw[_k] = os.path.expandvars(_v)
        _exp_dir = Path(_raw["experiment_directory"])
        _replica_name = kwargs.get("name", "ddsim")
        _raw["experiment_directory"] = str(_exp_dir.parent / f"{_exp_dir.name}-{_replica_name}")
        # Make node_local_path unique per replica so concurrent replicas don't
        # collide when writing sim scratch files (workdir = node_local_path/stage_task).
        if _raw.get("node_local_path"):
            _raw["node_local_path"] = str(Path(_raw["node_local_path"]) / _replica_name)
        self.experiment_config = ExperimentConfig(**_raw)

        agg_stage = self.experiment_config.aggregation_stage
        self.skip_aggregation = agg_stage.skip_aggregation
        self.api = DeepDriveMD_API(self.experiment_config.experiment_directory)

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
    async def _signal_ready(self) -> None:
        """Signal the CM that this workflow has produced enough data."""
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

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
        stages = [
            (
                "molecular_dynamics_stage",
                self.experiment_config.molecular_dynamics_stage,
            ),
            ("machine_learning_stage", self.experiment_config.machine_learning_stage),
            ("aggregation_stage", self.experiment_config.aggregation_stage),
            ("agent_stage", self.experiment_config.agent_stage),
            ("model_selection_stage", self.experiment_config.model_selection_stage),
        ]
        for idx, (name, cfg) in enumerate(stages):

            task_descriptions[name] = self._generate_task_description(
                cfg, stage_idx=idx
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
    def _generate_task_description(self, config, stage_idx: int = 0):
        """Build a single task resource description from a stage config."""
        # Apply a Dragon policies (and set gpus_per_rank=0) only for stages that
        # actually require GPU.  Applying a policies to CPU-only stages (training,
        # selection) pins them to a GPU-pinned Dragon worker, which can freeze
        # Dragon's IPC when the stage is long-running (e.g. CPU Keras training).
        stage_needs_gpu = config.gpu_reqs.processes > 0
        # pre_exec = list(config.pre_exec)
        # if not stage_needs_gpu:
        #     pass  # pre_exec is ignored by DragonExecutionBackendV3; env setup via tf_gpu_wrapper.sh
        task_description = {
            # "ranks": 1,
            # "cores_per_rank": config.cpu_reqs.processes,
            # "gpus_per_rank": 0
            # if (self.policies and stage_needs_gpu)
            # else config.gpu_reqs.processes,
            # "pre_exec": pre_exec,
            # "shell": True,
        }

        if self.policies and stage_needs_gpu:
            policy = self.policies[stage_idx % len(self.policies)]
            task_description["process_template"] = {"policy": policy}
        
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

        # Signal dependent workflows (e.g. inference) once after the first
        # complete iteration — enough data exists for downstream processing.
        if not self._data_ready_signaled:
            self._data_ready_signaled = True
            self.logger.info(
                f"Iteration {self.stage_idx} complete — "
                "signaling ready for downstream workflows",
                component=self.name,
            )
            await self._signal_ready()

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
        # If GPU policy are assigned, register one task function per GPU and
        # cycle by sim_idx so that the 8 simulation tasks spread across all
        # assigned GPUs, rather than all landing on the same GPU worker.
        _md_base = {
            k: v
            for k, v in self.task_descriptions["molecular_dynamics_stage"].items()
            if k != "process_template"
        }  # strip the single pre-baked policy

        if self.policies:
            _sim_fns = []
            for _gpu_idx, _policy in enumerate(self.policies):
                _td = {**_md_base, "process_template": {"policy": _policy}}
                self.logger.info(
                    f"Registering sim func [{_gpu_idx}] "
                    f"gpu_affinity={_policy.gpu_affinity}",
                    component=self.name,
                )

                @self.flow.executable_task
                async def _sim_gpu(task_description=_td, **kwargs):
                    cfg = stage_config["molecular_dynamics_stage"]
                    sim_idx = kwargs["sim_inputs"]["sim_idx"]
                    cfg.task_config.pdb_file = self.sim_inputs[sim_idx]
                    stage_api = api.molecular_dynamics_stage
                    _, cmd = _update_stage_config(stage_api, cfg, sim_idx)
                    return cmd

                _sim_fns.append(_sim_gpu)

            _n_gpus = len(_sim_fns)

            def simulation(**kwargs):
                sim_idx = kwargs["sim_inputs"]["sim_idx"]
                return _sim_fns[sim_idx % _n_gpus](**kwargs)

        else:
            self.logger.info(
                f"Using task_description {_md_base} for sim (no GPU policy)",
                component=self.name,
            )

            @self.flow.executable_task
            async def simulation(task_description=_md_base, **kwargs):
                cfg = stage_config["molecular_dynamics_stage"]
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

        _TF_GPU_WRAPPER = self.tf_gpu_wrapper

        # Training runs via asyncio subprocess directly (not Dragon executable_task).
        # Dragon worker subprocesses hang during TF's CUDA initialization even with
        # CUDA_VISIBLE_DEVICES=-1; the Dragon head process does not have this issue.
        async def training():
            cfg = stage_config["machine_learning_stage"]
            stage_api = api.machine_learning_stage
            output_path, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            cfg.task_config.model_tag = stage_api.unique_name(output_path)
            if self.stage_idx > 0:
                cfg.task_config.init_weights_path = None
            full_cmd = f"{_TF_GPU_WRAPPER} {cmd}"
            #self.logger.info(f"cmd= {full_cmd}", component=self.name)
            proc = await asyncio.create_subprocess_shell(
                full_cmd,
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

        # --- Agent task: runs inference/active learning ---
        # lof.py creates a CVAE model that hits the cuDNN 9.2.0/V100 status 5003
        # bug on every conv op.  Like training, we bypass Dragon's executable_task
        # entirely and run as a direct asyncio subprocess so Dragon's CUDA IPC does
        # not interfere with TF initialisation.  tf_cpu_wrapper.sh sets
        # CUDA_VISIBLE_DEVICES=-1 to force CPU execution.
        _TF_CPU_WRAPPER = self.tf_cpu_wrapper

        async def agent_stage():
            cfg = stage_config["agent_stage"]
            stage_api = api.agent_stage
            _, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            full_cmd = f"{_TF_CPU_WRAPPER} {cmd}"
            proc = await asyncio.create_subprocess_shell(
                full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                print(stdout.decode(), end="", flush=True)
            if proc.returncode != 0:
                self.logger.error(
                    f"Agent stage failed with exit code {proc.returncode}",
                    component=self.name,
                )

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
        if self.debug:
            self.logger.task_started(f"Iteration {self.iteration}", component=self.name)
            self.logger.info(
                f"{len(self.registered_sims)} simulation(s) running....",
                component=self.name,
            )

        if self.aggregation:
            await self.aggregation()

        await self.training()
        if self.debug:
            self.logger.task_completed(f"Iteration {self.iteration}", component=self.name)

        await self.selection()

    # --------------------------------------------------------------------------
    async def post_process_sim(self, sim_idx):
        del self.sim_inputs[sim_idx]

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shut down the asyncflow engine."""
        pass
