"""
Unit tests for MiniAppsWorkflow._*_cmd() builder methods and check_train_status.

These tests verify:
  - Command strings are built from current self.iteration / self.phase at call
    time, not captured at init time.
  - Each builder picks up mutations to self.iteration and self.phase.
  - check_train_status polls until the expected HDF5 file appears.
"""

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_contains_hdf5_env(self, workflow):
        env = workflow.task_description["process_template"]["env"]
        assert env.get("HDF5_USE_FILE_LOCKING") == "FALSE"

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

    def test_contains_hdf5_env(self, workflow):
        env = workflow.task_description["process_template"]["env"]
        assert env.get("HDF5_USE_FILE_LOCKING") == "FALSE"

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

    def test_contains_hdf5_env(self, workflow):
        env = workflow.task_description["process_template"]["env"]
        assert env.get("HDF5_USE_FILE_LOCKING") == "FALSE"

    def test_contains_agent_script(self, workflow):
        assert "agent.py" in workflow._prediction_cmd()


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
    async def test_returns_true_when_file_present(self, workflow):
        """check_train_status should return True once the expected file exists."""
        # iteration=0, next_iter=1 → file: phase0/data_0_1.h5
        phase_dir = Path(workflow.sim_output_dir) / "phase0"
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "data_0_1.h5").touch()

        result = await workflow.check_train_status()
        assert result is True

    @pytest.mark.asyncio
    async def test_waits_until_file_appears(self, workflow):
        """check_train_status should block until the file appears."""
        phase_dir = Path(workflow.sim_output_dir) / "phase0"
        phase_dir.mkdir(parents=True, exist_ok=True)
        target = phase_dir / "data_0_1.h5"

        async def _create_file_after_delay():
            await asyncio.sleep(0.05)
            target.touch()

        creator = asyncio.create_task(_create_file_after_delay())
        with patch(
            "workflows.miniapps_workflow.miniapps_workflow.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await asyncio.wait_for(
                asyncio.gather(workflow.check_train_status(), creator),
                timeout=2.0,
            )
        assert result[0] is True

    @pytest.mark.asyncio
    async def test_checks_correct_file_for_iteration(self, workflow):
        """File checked must match self.iteration + 1."""
        workflow.iteration = 2  # next_iter = 3 → data_0_3.h5
        phase_dir = Path(workflow.sim_output_dir) / "phase0"
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Wrong file (iter 1) — should NOT satisfy
        wrong = phase_dir / "data_0_1.h5"
        wrong.touch()

        # Correct file
        correct = phase_dir / "data_0_3.h5"

        async def _create_correct():
            await asyncio.sleep(0.02)
            correct.touch()

        creator = asyncio.create_task(_create_correct())
        with patch(
            "workflows.miniapps_workflow.miniapps_workflow.asyncio.sleep",
            side_effect=lambda _: asyncio.sleep(0),
        ):
            result = await asyncio.wait_for(
                asyncio.gather(workflow.check_train_status(), creator),
                timeout=2.0,
            )
        assert result[0] is True
        assert correct.exists()


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
