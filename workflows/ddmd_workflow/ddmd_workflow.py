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
        return {
            k: os.path.expandvars(v) if isinstance(v, str) else v
            for k, v in raw.items()
        }
    return {}


class DDMdWorkflow(DDSimManager):
    """DeepDriveMD workflow: orchestrates MD simulations, ML training,
    aggregation, model selection, and agent-based inference stages.

    Extends DDSimManager to implement the full DeepDriveMD adaptive loop
    using asyncflow for task scheduling and execution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name=kwargs.get("name", "ddsim"))

        # Callback injected by AsyncCampaignManager to unblock dependent workflows.
        self._on_ready = kwargs.get("on_ready", None)
        self._data_ready_signaled = False

        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate DDMdWorkflow w/o asyncflow")

        # Expand a compound policy (multiple GPUs in one policy) into per-GPU
        # policies so tasks can be distributed across GPUs round-robin.
        _raw_policies = kwargs.get("policies", []) or []
        if (
            len(_raw_policies) == 1
            and len(getattr(_raw_policies[0], "gpu_affinity", [])) > 1
        ):
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

        # Load campaign config and extract workflow parameters.
        camp_cfg = (
            _load_camp_config(kwargs["camp_config"]) if "camp_config" in kwargs else {}
        )
        _here = Path(__file__).parent
        config = camp_cfg.get("ddsim_config") or kwargs.get("config")
        self.debug = camp_cfg.get("debug", kwargs.get("debug", False))
        self.tf_gpu_wrapper = (
            camp_cfg.get("tf_gpu_wrapper")
            or kwargs.get("tf_gpu_wrapper")
            or str(_here / "tf_gpu_wrapper.sh")
        )
        self.tf_cpu_wrapper = (
            camp_cfg.get("tf_cpu_wrapper")
            or kwargs.get("tf_cpu_wrapper")
            or str(_here / "tf_cpu_wrapper.sh")
        )

        # Load experiment config, making experiment_directory and node_local_path
        # unique per replica to avoid data collisions when running multiple replicas.
        with open(config) as _fp:
            _raw = yaml.safe_load(_fp)
        for _k, _v in _raw.items():
            if isinstance(_v, str):
                _raw[_k] = os.path.expandvars(_v)
        _exp_dir = Path(_raw["experiment_directory"])
        _replica_name = kwargs.get("name", "ddsim")
        _raw["experiment_directory"] = str(
            _exp_dir.parent / f"{_exp_dir.name}-{_replica_name}"
        )
        if _raw.get("node_local_path"):
            _raw["node_local_path"] = str(Path(_raw["node_local_path"]) / _replica_name)
        self.experiment_config = ExperimentConfig(**_raw)

        agg_stage = self.experiment_config.aggregation_stage
        self.skip_aggregation = agg_stage.skip_aggregation
        self.api = DeepDriveMD_API(self.experiment_config.experiment_directory)

        self.workflow_id = "ddsim_workflow"
        self.stage_idx = 0

        self._init_experiment_dir()
        self.task_descriptions = self._generate_task_descriptions()
        self.stage_config = self._generate_stage_config()

        # Copy the config YAML into the experiment directory for reproducibility.
        shutil.copy(config, self.experiment_config.experiment_directory)

        self.num_sims = self.experiment_config.molecular_dynamics_stage.num_tasks
        self.max_sim_batch = self.num_sims
        self.sim_batch_size = self.max_sim_batch

        self.free_resources_for_train = False
        self.call_cancel_simulations = False
        self.call_finalize_results = True
        self.call_evaluate_simulations = True

        self.training_cores = 1
        self.iteration = 0
        self.retrain_model = True

        self.register_tasks()
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
        """Initialise each stage's task_config with shared experiment settings."""
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
        """Build resource-requirement dicts for each workflow stage."""
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
        """Build a single task resource description from a stage config.

        GPU policies are applied only to stages that request GPUs; attaching
        a policy to CPU-only stages can cause long-running tasks to stall.
        """
        stage_needs_gpu = config.gpu_reqs.processes > 0
        task_description = {}

        if self.policies and stage_needs_gpu:
            policy = self.policies[stage_idx % len(self.policies)]
            task_description["process_template"] = {"policy": policy}

        return task_description

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Populate the simulation task queue with initial PDB inputs.

        Cycles through available PDB files in round-robin order so tasks
        are spread evenly across the available input structures.
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
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.sim_inputs[sim_idx] = next_file

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """Decide whether to cancel a simulation based on prediction.

        Returns False — this workflow does not use early cancellation.
        """
        return False

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        """Advance to the next iteration or shut down when the max is reached.

        On completion, re-queues the next batch of simulations (pdb_file=None
        so the simulation task uses restart files from the previous iteration).
        Signals dependent workflows after the first complete iteration.
        """
        self.stage_idx += 1

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
    async def add_sims_to_queue(self, *args, **kwargs):
        """
        For subseq iterations, new simulations are added
        to the queue in finalize_results()
        """
        pass

    # --------------------------------------------------------------------------
    async def check_train_status(self):
        """Return True when all simulations for the current iteration have finished."""
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

        Each task function builds a shell command from its stage config and
        returns it for execution by the asyncflow backend.  Stages registered:
        simulation, aggregation (optional), training, agent (inference),
        and model selection.

        When GPU policies are present, one simulation function is registered
        per GPU and tasks are distributed round-robin across them.
        """

        def _update_stage_config(stage_api, cfg, task_idx=0):
            """Write per-task YAML config and build the shell command."""
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

        # --- Simulation ---
        _md_base = {
            k: v
            for k, v in self.task_descriptions["molecular_dynamics_stage"].items()
            if k != "process_template"
        }

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

        # --- Aggregation (optional) ---
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

        # --- Training ---
        # Training runs as a direct asyncio subprocess on the head process so
        # that it inherits the full environment and GPU access.
        tf_gpu_wrapper = self.tf_gpu_wrapper

        async def training():
            cfg = stage_config["machine_learning_stage"]
            stage_api = api.machine_learning_stage
            output_path, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            cfg.task_config.model_tag = stage_api.unique_name(output_path)
            if self.stage_idx > 0:
                cfg.task_config.init_weights_path = None
            full_cmd = f"{tf_gpu_wrapper} {cmd}"
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

        # --- Agent (inference / active learning) ---
        # The agent stage runs as a direct asyncio subprocess with CPU-only
        # execution to avoid interfering with ongoing GPU simulations.
        tf_cpu_wrapper = self.tf_cpu_wrapper

        async def agent_stage():
            cfg = stage_config["agent_stage"]
            stage_api = api.agent_stage
            _, cmd = _update_stage_config(stage_api, cfg, task_idx=0)
            full_cmd = f"{tf_cpu_wrapper} {cmd}"
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

        # --- Model selection ---
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
        """Execute one training iteration: aggregate, train, select best model."""
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
            self.logger.task_completed(
                f"Iteration {self.iteration}", component=self.name
            )

        await self.selection()

    # --------------------------------------------------------------------------
    async def post_process_sim(self, sim_idx):
        del self.sim_inputs[sim_idx]

    # --------------------------------------------------------------------------
    async def close(self):
        # Temporary cleanup to test campaign manager and aid Disk quota exceeded.
        exp_dir = self.experiment_config.experiment_directory
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)
            self.logger.info(
                f"Removed experiment directory: {exp_dir}", component=self.name
            )

    # --------------------------------------------------------------------------
    async def stop(self):
        """Alias for close(), can be used for external termination."""
        await self.close()
