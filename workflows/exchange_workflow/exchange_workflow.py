import random

from ddsim.ddsim_manager import DDSimManager


class ExchangeWorkflow(DDSimManager):
    """DeepDriveMD workflow: orchestrates MD simulations, ML training,
    aggregation, model selection, and agent-based inference stages.

    Extends DDSimManager to implement the full DeepDriveMD workflow
    using asyncflow for task scheduling and execution.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class (sets up logger, queues, event flags, etc.)
        super().__init__()

        # Asyncflow engine for registering and dispatching tasks
        self.flow = kwargs.get("asyncflow", None)
        if self.flow is None:
            raise ValueError("Unable to initiate ExchangeWorkflow w/o asyncflow")

        self.config = kwargs.get("config")

        self.workflow_id = "exchange_workflow"

        self._init_experiment_dir()
        self.task_descriptions = self._generate_task_description(self.config)

        # Batch size equals max since no resource sharing between sim and train
        self.sim_batch_size = float("inf")

        # If False, skip retraining (e.g. when model accuracy is sufficient)
        self.retrain_model = False
        self.call_finalize_results = True

        # Register simulation, training, aggregation, inference, selection tasks
        self.register_tasks()
        # Dict to store inputs for simulation
        self.sim_inputs = {}

        # Get all tasks from config
        self.task_types = self.config.get('tasks').keys()

    # --------------------------------------------------------------------------
    async def _signal_ready(self) -> None:
        """Signal the CM that this workflow has produced enough data."""
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    # --------------------------------------------------------------------------
    def _init_experiment_dir(self) -> None:
        """Create the experiment directory from config and store as self.experiment_dir."""
        self.experiment_dir = self.config["experiment_dir"]
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    def _generate_task_description(self, config):
        """Build a single task resource description from a stage config."""
        task_description = {}
        for t in self.task_types:
            task_description[t] = {
                "ranks": 1,
                "cores_per_rank": config[t].cpu_reqs,
                "gpus_per_rank": config[t].gpu_reqs,
                "pre_exec": config[t].pre_exec,
                "shell": True,
            }
        return task_description

    # --------------------------------------------------------------------------
    async def _add_sims_to_queue(self, sim_type, sim_input=None):
        """Add new simulation tasks to the queue based on sim_idx."""
        #n = self.all_sims[sim_type]
        #n += 1
        sim_idx = f"{sim_type}_{n}"
        #self.all_sims[sim_type] += 1
        await self.sim_task_queue.put({"sim_idx": sim_idx})
        self.sim_inputs[sim_idx] = sim_input

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Populate the simulation task queue with initial PDB inputs.

        Cycles through available PDB files so that each simulation task
        gets an input structure. If there are more tasks than PDB files,
        files are reused in round-robin order.
        """
        num_init_sims = self.config["num_init_sims"]
        for _ in range(num_init_sims):
            await self._add_sims_to_queue('MD')

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
        if self.sim_task_queue.empty():
            self.shutting_down.set()
            self.run_workflow = False

    # --------------------------------------------------------------------------
    def register_tasks(self):
        """
            Register function for all types of simulations 
        """
        task_description = self.task_descriptions["MD"]
        @self.flow.executable_task
        async def md_simulation(task_description=task_description, **kwargs):
            print("Add your code for MD simulation here")
        self.simulation = md_simulation

        task_description = self.task_descriptions["EXCHANGE"]
        @self.flow.executable_task
        async def exchange(task_description=task_description, **kwargs):
            print("Add your code for MD simulation here")
        self.exchange = exchange

    # --------------------------------------------------------------------------
    async def add_sims_to_queue(self, sim_ids):
        """Re-queue simulations that were cancelled to free resources."""
        for sim_idx in sim_ids:
            sim_type = sim_idx.split("_")[0]
            sim_input = self.sim_inputs.get(sim_idx)
            await self._add_sims_to_queue(sim_type, sim_input)

    # --------------------------------------------------------------------------
    async def post_process(self):
        """Called after each inference iteration; advance workflow or shut down."""
        await self.finalize_results()

    # --------------------------------------------------------------------------
    async def post_process_sim(self, sim_idx):
        del self.sim_inputs[sim_idx]

        sim_type = sim_idx.split("_")[0]
        print(f'Add your code to retrive sim status: {sim_type}')
        sim_result = random.random()
        if sim_result == 'susspended':
            print(f'Sim {sim_idx} has been susspended')

            await self.exchange()

            if next_type is not None:
                await self._add_sims_to_queue(next_type)
        
    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shut down the asyncflow engine."""
        await self.flow.shutdown()