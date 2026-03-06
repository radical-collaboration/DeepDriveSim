"""Unit tests for workflows.ddmd_workflow.data.api module."""

import json
from pathlib import Path

import pytest

MDAnalysis = pytest.importorskip("MDAnalysis", reason="MDAnalysis not installed")

from workflows.ddmd_workflow.data.api import DeepDriveMD_API, Stage_API  # noqa: E402


# ---------------------------------------------------------------------------
# Stage_API naming helpers
# ---------------------------------------------------------------------------
class TestStageAPINaming:
    def test_task_name_format(self):
        assert Stage_API.task_name(0) == "task0000"
        assert Stage_API.task_name(5) == "task0005"
        assert Stage_API.task_name(9999) == "task9999"

    def test_stage_name_format(self):
        assert Stage_API.stage_name(0) == "stage0000"
        assert Stage_API.stage_name(3) == "stage0003"

    def test_unique_name(self, tmp_path):
        task_path = tmp_path / "stage0001" / "task0003"
        task_path.mkdir(parents=True)
        assert Stage_API.unique_name(task_path) == "stage0001_task0003"


# ---------------------------------------------------------------------------
# Stage_API directory operations
# ---------------------------------------------------------------------------
class TestStageAPIDirs:
    @pytest.fixture
    def stage_api(self, tmp_path):
        """Create a Stage_API with a populated directory structure."""
        api = Stage_API(tmp_path, "md_runs")
        runs = tmp_path / "md_runs"
        runs.mkdir()
        for s in range(3):
            stage = runs / f"stage{s:04d}"
            stage.mkdir()
            for t in range(2):
                (stage / f"task{t:04d}").mkdir()
        return api

    def test_runs_dir(self, stage_api, tmp_path):
        assert stage_api.runs_dir == tmp_path / "md_runs"

    def test_stage_dir_by_index(self, stage_api):
        assert stage_api.stage_dir(0).name == "stage0000"
        assert stage_api.stage_dir(2).name == "stage0002"

    def test_stage_dir_latest(self, stage_api):
        latest = stage_api.stage_dir(-1)
        assert latest is not None
        assert latest.name == "stage0002"

    def test_stage_dir_count(self, stage_api):
        assert stage_api.stage_dir_count() == 3

    def test_task_dir_by_index(self, stage_api):
        task = stage_api.task_dir(stage_idx=1, task_idx=0)
        assert task is not None
        assert task.name == "task0000"
        assert task.parent.name == "stage0001"

    def test_task_dir_mkdir(self, stage_api):
        task = stage_api.task_dir(stage_idx=0, task_idx=5, mkdir=True)
        assert task is not None
        assert task.exists()
        assert task.name == "task0005"

    def test_config_path(self, stage_api):
        path = stage_api.config_path(stage_idx=0, task_idx=0)
        assert path is not None
        assert path.suffix == ".yaml"
        assert "stage0000_task0000" in path.name

    def test_json_path(self, stage_api):
        path = stage_api.json_path(stage_idx=0, task_idx=0)
        assert path is not None
        assert path.suffix == ".json"


# ---------------------------------------------------------------------------
# Stage_API JSON read/write
# ---------------------------------------------------------------------------
class TestStageAPIJson:
    @pytest.fixture
    def stage_api(self, tmp_path):
        api = Stage_API(tmp_path, "runs")
        runs = tmp_path / "runs" / "stage0000" / "task0000"
        runs.mkdir(parents=True)
        return api

    def test_write_and_read_task_json(self, stage_api):
        data = [{"key": "value", "num": 42}]
        stage_api.write_task_json(data, stage_idx=0, task_idx=0)
        result = stage_api.read_task_json(stage_idx=0, task_idx=0)
        assert result == data

    def test_read_task_json_nonexistent_returns_none(self, tmp_path):
        api = Stage_API(tmp_path, "empty_runs")
        result = api.read_task_json(stage_idx=0, task_idx=0)
        assert result is None


# ---------------------------------------------------------------------------
# Stage_API get_latest / get_count
# ---------------------------------------------------------------------------
class TestStageAPIGetLatest:
    def test_get_latest_returns_max(self, tmp_path):
        for name in ["a", "b", "c"]:
            (tmp_path / name).mkdir()
        result = Stage_API.get_latest(tmp_path, "*", is_dir=True)
        assert result == tmp_path / "c"

    def test_get_latest_empty_returns_none(self, tmp_path):
        result = Stage_API.get_latest(tmp_path, "stage*", is_dir=True)
        assert result is None

    def test_get_count(self, tmp_path):
        for i in range(4):
            (tmp_path / f"stage{i:04d}").mkdir()
        (tmp_path / "not_a_stage.txt").write_text("hi")
        count = Stage_API.get_count(tmp_path, "stage*", is_dir=True)
        assert count == 4


# ---------------------------------------------------------------------------
# DeepDriveMD_API initialization
# ---------------------------------------------------------------------------
class TestDeepDriveMDAPIInit:
    def test_creates_stage_apis(self, tmp_path):
        api = DeepDriveMD_API(tmp_path)
        assert isinstance(api.molecular_dynamics_stage, Stage_API)
        assert isinstance(api.aggregation_stage, Stage_API)
        assert isinstance(api.machine_learning_stage, Stage_API)
        assert isinstance(api.model_selection_stage, Stage_API)
        assert isinstance(api.agent_stage, Stage_API)

    def test_get_total_iterations_empty(self, tmp_path):
        api = DeepDriveMD_API(tmp_path)
        (tmp_path / "molecular_dynamics_runs").mkdir()
        assert api.get_total_iterations() == 0


# ---------------------------------------------------------------------------
# DeepDriveMD_API PDB helpers
# ---------------------------------------------------------------------------
class TestDeepDriveMDAPIPDB:
    def test_get_initial_pdbs(self, tmp_path):
        """PDB files must be inside subdirectories."""
        system_dir = tmp_path / "system1"
        system_dir.mkdir()
        (system_dir / "protein.pdb").write_text("ATOM")
        (system_dir / "protein2.pdb").write_text("ATOM")

        pdbs = DeepDriveMD_API.get_initial_pdbs(tmp_path)
        assert len(pdbs) == 2
        assert all(p.suffix == ".pdb" for p in pdbs)

    def test_get_initial_pdbs_empty_dir(self, tmp_path):
        """Returns empty list when no PDBs found."""
        pdbs = DeepDriveMD_API.get_initial_pdbs(tmp_path)
        assert pdbs == []

    def test_get_initial_pdbs_rejects_double_underscore(self, tmp_path):
        system_dir = tmp_path / "sys"
        system_dir.mkdir()
        (system_dir / "bad__name.pdb").write_text("ATOM")

        with pytest.raises(ValueError, match="double underscore"):
            DeepDriveMD_API.get_initial_pdbs(tmp_path)

    def test_get_system_name_from_subdirectory(self, tmp_path):
        pdb = tmp_path / "bba" / "1FME.pdb"
        name = DeepDriveMD_API.get_system_name(pdb)
        assert name == "bba"

    def test_get_system_name_from_double_underscore(self):
        pdb = Path("/path/to/bba__restart.pdb")
        name = DeepDriveMD_API.get_system_name(pdb)
        assert name == "bba"

    def test_get_system_pdb_name_adds_system_prefix(self):
        pdb = Path("/path/to/bba/protein.pdb")
        result = DeepDriveMD_API.get_system_pdb_name(pdb)
        assert result == "bba__protein.pdb"

    def test_get_system_pdb_name_preserves_existing(self):
        pdb = Path("/path/to/bba__protein.pdb")
        result = DeepDriveMD_API.get_system_pdb_name(pdb)
        assert result == "bba__protein.pdb"

    def test_get_system_pdb_name_rejects_multiple_double_underscores(self):
        pdb = Path("/path/to/a__b__c.pdb")
        with pytest.raises(ValueError, match="one occurence"):
            DeepDriveMD_API.get_system_pdb_name(pdb)

    def test_get_topology_found(self, tmp_path):
        system_dir = tmp_path / "bba"
        system_dir.mkdir()
        top_file = system_dir / "system.top"
        top_file.write_text("topology")

        pdb = system_dir / "protein.pdb"
        result = DeepDriveMD_API.get_topology(tmp_path, pdb, suffix=".top")
        assert result is not None
        assert result.name == "system.top"

    def test_get_topology_not_found(self, tmp_path):
        system_dir = tmp_path / "bba"
        system_dir.mkdir()
        pdb = system_dir / "protein.pdb"
        result = DeepDriveMD_API.get_topology(tmp_path, pdb, suffix=".top")
        assert result is None


# ---------------------------------------------------------------------------
# DeepDriveMD_API restart PDB
# ---------------------------------------------------------------------------
class TestDeepDriveMDAPIRestart:
    def test_get_restart_pdb(self, tmp_path):
        """get_restart_pdb retrieves data from agent JSON."""
        api = DeepDriveMD_API(tmp_path)
        # Create agent stage directory structure
        agent_dir = tmp_path / "agent_runs" / "stage0000" / "task0000"
        agent_dir.mkdir(parents=True)
        json_path = agent_dir / "stage0000_task0000.json"
        data = [{"pdb": "restart_0.pdb"}, {"pdb": "restart_1.pdb"}]
        json_path.write_text(json.dumps(data))

        result = api.get_restart_pdb(index=1, stage_idx=0, task_idx=0)
        assert result == {"pdb": "restart_1.pdb"}
