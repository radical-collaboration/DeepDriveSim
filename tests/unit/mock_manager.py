import asyncio
import random
from unittest.mock import MagicMock

import pytest

DDSimManager = pytest.importorskip(
    "ddsim.ddsim_manager",
    reason="ddsim.ddsim_manager not importable (missing radical.asyncflow)",
).DDSimManager

# from unittest.mock import Mock


# ---------------------------
# Minimal stubs for logger/learner
# ---------------------------
class DummyLogger:
    def __init__(self):
        self.info = MagicMock()
        self.warning = MagicMock()
        self.error = MagicMock()
        self.task_started = MagicMock()
        self.task_completed = MagicMock()
        self.task_killed = MagicMock()
        self.manager_exiting = MagicMock()
        self.separator = MagicMock()


class MockLearner(DDSimManager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, **kwargs):
        # Simulation/training config
        self.max_sim_batch = kwargs.get("max_sim_batch", 4)
        self.training_cores = kwargs.get("training_cores", 1)
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        self.training_threshold = kwargs.get("training_threshold", 0.5)
        self.prediction_threshold = kwargs.get("prediction_threshold", 0.5)
        self.start_training_threshold = kwargs.get("start_training_threshold", 10)
        self.training_epochs = kwargs.get("training_epochs", 1)
        self.force_start_training = bool(kwargs.get("force_start_training", True))
        self.clean_unregistered_sims = bool(kwargs.get("clean_unregistered_sims", True))

        self.iteration = 0
        self.retrain_model = self.training_epochs > 0
        self.sim_predictions = {}
        self.selection = None
        self.run_prediction_as_exe = False

        # Initialize parent class (sets up asyncflow, logger, queues, etc.)
        asyncflow = kwargs.get("asyncflow")
        super().__init__(asyncflow)

        # Override timing for faster tests
        self.time_between_predictions = 0.1
        self.time_before_shutdown = 0.1

        # Register learner tasks
        self._register_learner_tasks()
        self.logger = DummyLogger()

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """Return True if prediction < threshold (cancel simulation)."""
        return kwargs["prediction"] < self.prediction_threshold

    # --------------------------------------------------------------------------
    async def collect_sim_inputs(self, n=5):
        """Collect all simulation input files into task queue."""
        for i in range(n):
            sim_tag = f"sim_{i}"
            await self.sim_task_queue.put({"sim_tag": sim_tag})

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Initialize simulation queue (uses collect_sim_inputs for tests)."""
        pass

    # --------------------------------------------------------------------------
    async def check_train_data(self):
        """Check if enough training data is available to start training."""
        return True

    # --------------------------------------------------------------------------
    async def clean_sim_data(self, sim_ind):
        """Clean up simulation data files."""
        pass

    # --------------------------------------------------------------------------
    async def del_files(self, sim_ind):
        """Alias for backwards compatibility in tests."""
        await self.clean_sim_data(sim_ind)

    # --------------------------------------------------------------------------
    def _register_learner_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        @self.learner.simulation_task(as_executable=False)
        async def simulation(*args, **kwargs):
            await asyncio.sleep(5)
            return True

        self.simulation = simulation

        # will work after UQ branch of ROSE is finalized
        # @self.learner.prediction_task(as_executable=False)
        @self.learner.utility_task(as_executable=False)
        async def prediction(*args, **kwargs):
            """Dummy prediction: assign random score to each sim."""
            sim_inds = list(self.registered_sims.keys())
            return {sim_ind: random.random() for sim_ind in sim_inds}

        self.prediction = prediction

    # --------------------------------------------------------------------------
    async def train_model(self):
        pass


# @pytest.fixture
# def mock_execution_backend():
#     """Mock execution backend"""
#     return Mock()


# @pytest.fixture
# def ddmd_workflow(mock_execution_backend):
#     """Create an ImpressManager instance for testing"""
#     manager = MockLearner(asyncflow=mock_execution_backend)
#     manager.logger = Mock()  # Mock the logger
#     return manager
