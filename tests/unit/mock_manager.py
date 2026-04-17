import asyncio
from unittest.mock import MagicMock

import pytest

DDSimManager = pytest.importorskip(
    "ddsim.ddsim_manager",
    reason="ddsim.ddsim_manager not importable",
).DDSimManager


# ---------------------------
# Minimal stubs for logger
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
    """Minimal workflow subclass for unit and integration tests."""

    def __init__(self, **kwargs):
        # Initialize parent class first so its __init__ doesn't overwrite the
        # values we set below (DDSimManager.__init__ resets sim_batch_size and
        # max_sim_batch to 0).
        super().__init__()

        # Simulation/training config — must come after super().__init__()
        self.max_sim_batch = kwargs.get("max_sim_batch", 4)
        self.training_cores = kwargs.get("training_cores", 1)
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        self.training_threshold = kwargs.get("training_threshold", 0.5)
        self.prediction_threshold = kwargs.get("prediction_threshold", 0.5)
        self.start_training_threshold = kwargs.get("start_training_threshold", 10)
        self.training_epochs = kwargs.get("training_epochs", 1)
        self.force_start_training = bool(kwargs.get("force_start_training", True))

        self.iteration = 0
        self.retrain_model = self.training_epochs > 0

        # Override timing for faster tests
        self.sleep_time = 0.01

        # Enable finalize_results so start() has an exit path
        self.call_finalize_results = True

        # Register simulation callable
        self._register_tasks()
        self.logger = DummyLogger()

    # --------------------------------------------------------------------------
    def _register_tasks(self):
        """Register simulation as a plain asyncio.Task factory."""

        def simulation(sim_inputs=None, **kwargs):
            return asyncio.create_task(asyncio.sleep(10.0))

        self.simulation = simulation

    # --------------------------------------------------------------------------
    def stop_simulation(self, prediction=None, **kwargs):
        """Cancel simulation if prediction is below threshold."""
        return (prediction or 0) < self.prediction_threshold

    # --------------------------------------------------------------------------
    async def collect_sim_inputs(self, n=5):
        """Helper to populate the queue with n dummy simulation inputs."""
        for i in range(n):
            await self.sim_task_queue.put({"sim_idx": f"sim_{i}"})

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        pass

    # --------------------------------------------------------------------------
    async def check_train_status(self):
        return True

    # --------------------------------------------------------------------------
    async def train_model(self):
        pass

    # --------------------------------------------------------------------------
    async def run_inference(self):
        pass

    # --------------------------------------------------------------------------
    async def add_sims_to_queue(self, sim_ids):
        pass

    # --------------------------------------------------------------------------
    async def del_files(self, sim_idx):
        """Alias kept for backwards compatibility with existing tests."""
        pass

    # --------------------------------------------------------------------------
    async def post_process_sim(self, sim_idx):
        pass

    # --------------------------------------------------------------------------
    async def finalize_results(self):
        """
        Stop the workflow after one iteration (keeps start()
        from looping forever).
        """
        self.run_workflow = False
        self.shutting_down.set()

    # --------------------------------------------------------------------------
    async def close(self):
        self.shutting_down.set()
