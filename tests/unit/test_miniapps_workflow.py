"""
Unit tests for MiniAppsWorkflow._*_cmd() builder methods and check_train_status.

These tests verify:
  - Command strings are built from current self.iteration / self.phase at call
    time, not captured at init time.
  - Each builder picks up mutations to self.iteration and self.phase.
  - check_train_status polls until the expected HDF5 file appears.
"""

import shutil
from unittest.mock import MagicMock

import pytest

pytest.importorskip("radical.asyncflow", reason="radical.asyncflow not installed")


# ---------------------------------------------------------------------------
# Fixture: a MiniAppsWorkflow with all external deps mocked out
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_home(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture()
def workflow(tmp_home):
    """
    MiniAppsWorkflow with radical.asyncflow mocked.
    No real filesystem side-effects beyond tmp_home.
    """
    from workflows.miniapps_workflow.miniapps_workflow import MiniAppsWorkflow

    mock_asyncflow = MagicMock()

    cfg = {
        "debug": False,
        "total_num_sim": 10,
        "max_sim_batch": 4,
        "training_cores": 1,
        "free_resources_for_train": False,
        "phase": 0,
        "num_step": 500,
        "num_epochs": 5,
        "num_mult": 1,
        "miniapps_data_ready": 3,
    }

    return MiniAppsWorkflow(
        config=cfg,
        asyncflow=mock_asyncflow,
        name="test_wf",
        home_dir=str(tmp_home),
    )


# ---------------------------------------------------------------------------
# Group 1: _sim_cmd
# ---------------------------------------------------------------------------


class TestSimCmd:
    def test_contains_sim_idx(self, workflow):
        cmd = workflow._sim_cmd(sim_idx=3)
        assert "--instance_index 3" in cmd

    def test_contains_phase(self, workflow):
        cmd = workflow._sim_cmd(sim_idx=0)
        assert f"--phase {workflow.phase}" in cmd

    def test_contains_num_step(self, workflow):
        cmd = workflow._sim_cmd(sim_idx=0)
        assert f"--num_step {workflow.num_step}" in cmd

    def test_contains_simulation_script(self, workflow):
        cmd = workflow._sim_cmd(sim_idx=0)
        assert "simulation.py" in cmd

    def test_phase_updates_live(self, workflow):
        workflow.phase = 2
        cmd = workflow._sim_cmd(sim_idx=0)
        assert "--phase 2" in cmd


# ---------------------------------------------------------------------------
# Group 2: _training_cmd — iteration must be read at call time
# ---------------------------------------------------------------------------


class TestTrainingCmd:
    def test_initial_iteration(self, workflow):
        workflow.iteration = 0
        cmd = workflow._training_cmd()
        assert "--instance_index 0" in cmd

    def test_iteration_updates_live(self, workflow):
        """Key test: after self.iteration changes, _training_cmd reflects it."""
        workflow.iteration = 0
        cmd0 = workflow._training_cmd()
        assert "--instance_index 0" in cmd0

        workflow.iteration = 3
        cmd3 = workflow._training_cmd()
        assert "--instance_index 3" in cmd3
        # Old command must NOT have contained 3
        assert "--instance_index 3" not in cmd0

    def test_contains_phase(self, workflow):
        cmd = workflow._training_cmd()
        assert f"--phase {workflow.phase}" in cmd

    def test_contains_num_epochs(self, workflow):
        cmd = workflow._training_cmd()
        assert f"--num_epochs {workflow.num_epochs}" in cmd

    def test_contains_training_script(self, workflow):
        cmd = workflow._training_cmd()
        assert "training.py" in cmd


# ---------------------------------------------------------------------------
# Group 3: _prediction_cmd
# ---------------------------------------------------------------------------


class TestPredictionCmd:
    def test_iteration_updates_live(self, workflow):
        workflow.iteration = 1
        cmd1 = workflow._prediction_cmd()
        assert "--instance_index 1" in cmd1

        workflow.iteration = 5
        cmd5 = workflow._prediction_cmd()
        assert "--instance_index 5" in cmd5

    def test_contains_output_file(self, workflow):
        cmd = workflow._prediction_cmd()
        assert "--output_file" in cmd
        assert "predictions.yaml" in cmd

    def test_contains_agent_script(self, workflow):
        assert "agent.py" in workflow._prediction_cmd()

    def test_num_mult_uses_num_mult_not_num_epochs(self, workflow):
        """Regression: num_mult was read from num_epochs due to copy-paste error."""
        assert f"--num_mult {workflow.num_mult}" in workflow._prediction_cmd()
        assert f"--num_mult {workflow.num_epochs}" not in workflow._prediction_cmd()

    def test_num_mult_default_is_one_when_not_in_config(self, tmp_path):
        """When num_mult is absent from config, it defaults to 1, not num_epochs."""
        from unittest.mock import MagicMock

        from workflows.miniapps_workflow.miniapps_workflow import MiniAppsWorkflow

        cfg = {"num_epochs": 100}  # num_mult intentionally absent
        w = MiniAppsWorkflow(config=cfg, asyncflow=MagicMock(), home_dir=str(tmp_path))
        assert w.num_mult == 1
        assert "--num_mult 1" in w._prediction_cmd()


# ---------------------------------------------------------------------------
# Group 4: _selection_cmd
# ---------------------------------------------------------------------------


class TestSelectionCmd:
    def test_iteration_updates_live(self, workflow):
        workflow.iteration = 2
        cmd = workflow._selection_cmd()
        assert "--instance_index 2" in cmd

        workflow.iteration = 7
        assert "--instance_index 7" in workflow._selection_cmd()

    def test_contains_phase(self, workflow):
        assert f"--phase {workflow.phase}" in workflow._selection_cmd()

    def test_contains_selection_script(self, workflow):
        assert "selection.py" in workflow._selection_cmd()


# ---------------------------------------------------------------------------
# Group 5: check_train_status — polls until HDF5 file appears
# ---------------------------------------------------------------------------


class TestCheckTrainStatus:
    @pytest.mark.asyncio
    async def test_returns_false_below_threshold(self, workflow):
        """Below train_every_n_sims new completions → False."""
        workflow.completed_sims = [0]  # 1 new, threshold is 3
        # Keep a fake running sim so the "nothing left" safety exit doesn't fire.
        workflow.registered_sims = {"sim_0": MagicMock()}
        result = await workflow.check_train_status()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_at_threshold(self, workflow):
        """Exactly train_every_n_sims new completions → True."""
        workflow.completed_sims = list(range(workflow.train_every_n_sims))
        result = await workflow.check_train_status()
        assert result is True

    @pytest.mark.asyncio
    async def test_false_again_after_training_run(self, workflow):
        """After training snapshots the count, check_train_status returns False
        until another batch of train_every_n_sims sims complete."""
        workflow.completed_sims = list(range(workflow.train_every_n_sims))
        workflow.registered_sims = {"sim_0": MagicMock()}
        assert await workflow.check_train_status() is True
        # Snapshot what train_model() would do (avoids awaiting real tasks).
        workflow._sims_at_last_train = len(workflow.completed_sims)
        # No new sims yet → False
        assert await workflow.check_train_status() is False
        # Add enough new sims → True again
        workflow.completed_sims = list(range(workflow.train_every_n_sims * 2))
        assert await workflow.check_train_status() is True

    @pytest.mark.asyncio
    async def test_returns_true_when_nothing_running_or_queued(self, workflow):
        """Safety: if no sims are running and the queue is empty, return True
        so the workflow does not wait forever when sims have all failed."""
        workflow.completed_sims = []
        workflow.registered_sims = {}
        # Queue is empty by default in fixture
        result = await workflow.check_train_status()
        assert result is True


# ---------------------------------------------------------------------------
# Group 6: init_sim_queue
# ---------------------------------------------------------------------------


class TestInitSimQueue:
    @pytest.mark.asyncio
    async def test_queue_populated_with_total_num_sim(self, workflow):
        await workflow.init_sim_queue()
        assert workflow.sim_task_queue.qsize() == workflow.total_num_sim

    @pytest.mark.asyncio
    async def test_queue_items_have_sim_idx(self, workflow):
        await workflow.init_sim_queue()
        items = []
        while not workflow.sim_task_queue.empty():
            items.append(workflow.sim_task_queue.get_nowait())
        indices = [item["sim_idx"] for item in items]
        assert indices == list(range(workflow.total_num_sim))


# ---------------------------------------------------------------------------
# Group 7: finalize_results exit condition
# ---------------------------------------------------------------------------


class TestFinalizeResults:
    @pytest.mark.asyncio
    async def test_stops_workflow_when_all_sims_done(self, workflow):
        workflow.completed_sims = list(range(workflow.total_num_sim))
        await workflow.finalize_results()
        assert workflow.run_workflow is False
        assert workflow.shutting_down.is_set()

    @pytest.mark.asyncio
    async def test_does_not_stop_when_sims_incomplete(self, workflow):
        workflow.completed_sims = [0, 1]
        workflow.run_workflow = True
        await workflow.finalize_results()
        assert workflow.run_workflow is True
