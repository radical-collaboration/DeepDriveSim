"""Unit tests for workflows.ddmd_workflow.config module."""

from pathlib import Path

import pytest
import yaml

from workflows.ddmd_workflow.config import (
    AgentStageConfig,
    AgentTaskConfig,
    AggregationStageConfig,
    AggregationTaskConfig,
    BaseStageConfig,
    BaseTaskConfig,
    CPUReqs,
    ExperimentConfig,
    GPUReqs,
    MachineLearningStageConfig,
    MachineLearningTaskConfig,
    ModelSelectionStageConfig,
    ModelSelectionTaskConfig,
    MolecularDynamicsStageConfig,
    MolecularDynamicsTaskConfig,
    generate_sample_config,
)


# ---------------------------------------------------------------------------
# BaseSettings: YAML serialization
# ---------------------------------------------------------------------------
class TestBaseSettings:
    def test_dump_yaml_and_from_yaml_roundtrip(self, tmp_path):
        """Test that dump_yaml -> from_yaml preserves data."""
        cfg = BaseTaskConfig(stage_idx=3, task_idx=7)
        yaml_path = tmp_path / "test.yaml"
        cfg.dump_yaml(yaml_path)

        loaded = BaseTaskConfig.from_yaml(yaml_path)
        assert loaded.stage_idx == 3
        assert loaded.task_idx == 7

    def test_from_yaml_expands_env_vars(self, tmp_path, monkeypatch):
        """Test that from_yaml expands environment variables in string values."""
        monkeypatch.setenv("TEST_NODE_PATH", "/scratch/local")
        yaml_path = tmp_path / "env_test.yaml"
        yaml_path.write_text(
            "stage_idx: 0\n"
            "task_idx: 0\n"
            "experiment_directory: /some/path\n"
            "output_path: /some/output\n"
            "node_local_path: ${TEST_NODE_PATH}\n"
        )
        loaded = BaseTaskConfig.from_yaml(yaml_path)
        assert str(loaded.node_local_path) == "/scratch/local"


# ---------------------------------------------------------------------------
# CPUReqs / GPUReqs validation
# ---------------------------------------------------------------------------
class TestCPUReqs:
    def test_defaults(self):
        reqs = CPUReqs()
        assert reqs.processes == 1
        assert reqs.threads_per_process == 1
        assert reqs.process_type is None
        assert reqs.thread_type is None

    def test_valid_process_type_mpi(self):
        reqs = CPUReqs(process_type="MPI")
        assert reqs.process_type == "MPI"

    def test_invalid_process_type_raises(self):
        with pytest.raises(ValueError, match="process_type"):
            CPUReqs(process_type="INVALID")

    def test_valid_thread_type_openmp(self):
        reqs = CPUReqs(thread_type="OpenMP")
        assert reqs.thread_type == "OpenMP"

    def test_invalid_thread_type_raises(self):
        with pytest.raises(ValueError, match="thread_type"):
            CPUReqs(thread_type="CUDA")


class TestGPUReqs:
    def test_defaults(self):
        reqs = GPUReqs()
        assert reqs.processes == 0
        assert reqs.threads_per_process == 0

    def test_valid_thread_type_cuda(self):
        reqs = GPUReqs(thread_type="CUDA")
        assert reqs.thread_type == "CUDA"

    def test_invalid_process_type_raises(self):
        with pytest.raises(ValueError, match="process_type"):
            GPUReqs(process_type="SHMEM")

    def test_invalid_thread_type_raises(self):
        with pytest.raises(ValueError, match="thread_type"):
            GPUReqs(thread_type="INVALID")


# ---------------------------------------------------------------------------
# BaseTaskConfig
# ---------------------------------------------------------------------------
class TestBaseTaskConfig:
    def test_extra_fields_allowed(self):
        """BaseTaskConfig has extra='allow' for arbitrary task params."""
        cfg = BaseTaskConfig(custom_field="hello")
        assert cfg.custom_field == "hello"

    def test_defaults(self):
        cfg = BaseTaskConfig()
        assert cfg.stage_idx == 0
        assert cfg.task_idx == 0


# ---------------------------------------------------------------------------
# BaseStageConfig
# ---------------------------------------------------------------------------
class TestBaseStageConfig:
    def test_defaults(self):
        cfg = BaseStageConfig()
        assert cfg.pre_exec == []
        assert cfg.executable == ""
        assert cfg.arguments == []
        assert isinstance(cfg.cpu_reqs, CPUReqs)
        assert isinstance(cfg.gpu_reqs, GPUReqs)


# ---------------------------------------------------------------------------
# MolecularDynamicsTaskConfig
# ---------------------------------------------------------------------------
class TestMolecularDynamicsTaskConfig:
    def test_requires_initial_pdb_dir(self):
        with pytest.raises(ValueError):
            MolecularDynamicsTaskConfig()

    def test_with_initial_pdb_dir(self, tmp_path):
        cfg = MolecularDynamicsTaskConfig(initial_pdb_dir=tmp_path)
        assert cfg.initial_pdb_dir == tmp_path
        assert cfg.pdb_file == Path("set_by_deepdrivemd")


# ---------------------------------------------------------------------------
# AggregationStageConfig
# ---------------------------------------------------------------------------
class TestAggregationStageConfig:
    def test_skip_aggregation_default(self):
        cfg = AggregationStageConfig(task_config=AggregationTaskConfig())
        assert cfg.skip_aggregation is False

    def test_skip_aggregation_true(self):
        cfg = AggregationStageConfig(
            skip_aggregation=True, task_config=AggregationTaskConfig()
        )
        assert cfg.skip_aggregation is True


# ---------------------------------------------------------------------------
# MachineLearningTaskConfig
# ---------------------------------------------------------------------------
class TestMachineLearningTaskConfig:
    def test_defaults(self):
        cfg = MachineLearningTaskConfig()
        assert cfg.model_tag == "set_by_deepdrivemd"
        assert cfg.init_weights_path is None

    def test_with_weights_path(self, tmp_path):
        weights = tmp_path / "model.h5"
        cfg = MachineLearningTaskConfig(init_weights_path=weights)
        assert cfg.init_weights_path == weights


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------
class TestExperimentConfig:
    def test_experiment_directory_must_not_exist(self, tmp_path):
        """Validator rejects existing directories."""
        existing_dir = tmp_path / "exists"
        existing_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="already exists"):
            ExperimentConfig(
                title="test",
                resource="local",
                queue="default",
                schema_="local",
                project="test",
                walltime_min=10,
                max_iteration=1,
                cpus_per_node=4,
                gpus_per_node=0,
                hardware_threads_per_cpu=1,
                experiment_directory=existing_dir,
                node_local_path=None,
                molecular_dynamics_stage=MolecularDynamicsStageConfig(
                    task_config=MolecularDynamicsTaskConfig(initial_pdb_dir=tmp_path)
                ),
                aggregation_stage=AggregationStageConfig(
                    task_config=AggregationTaskConfig()
                ),
                machine_learning_stage=MachineLearningStageConfig(
                    task_config=MachineLearningTaskConfig()
                ),
                model_selection_stage=ModelSelectionStageConfig(
                    task_config=ModelSelectionTaskConfig()
                ),
                agent_stage=AgentStageConfig(task_config=AgentTaskConfig()),
            )

    def test_experiment_directory_must_be_absolute(self, tmp_path):
        """Validator rejects relative paths."""
        with pytest.raises(ValueError, match="absolute path"):
            ExperimentConfig(
                title="test",
                resource="local",
                queue="default",
                schema_="local",
                project="test",
                walltime_min=10,
                max_iteration=1,
                cpus_per_node=4,
                gpus_per_node=0,
                hardware_threads_per_cpu=1,
                experiment_directory=Path("relative/path/nonexistent"),
                node_local_path=None,
                molecular_dynamics_stage=MolecularDynamicsStageConfig(
                    task_config=MolecularDynamicsTaskConfig(initial_pdb_dir=tmp_path)
                ),
                aggregation_stage=AggregationStageConfig(
                    task_config=AggregationTaskConfig()
                ),
                machine_learning_stage=MachineLearningStageConfig(
                    task_config=MachineLearningTaskConfig()
                ),
                model_selection_stage=ModelSelectionStageConfig(
                    task_config=ModelSelectionTaskConfig()
                ),
                agent_stage=AgentStageConfig(task_config=AgentTaskConfig()),
            )


# ---------------------------------------------------------------------------
# generate_sample_config
# ---------------------------------------------------------------------------
class TestGenerateSampleConfig:
    def test_generates_valid_config(self):
        """generate_sample_config returns an ExperimentConfig instance."""
        config = generate_sample_config()
        assert isinstance(config, ExperimentConfig)
        assert config.max_iteration == 4

    def test_dump_to_yaml(self, tmp_path):
        """Sample config can be serialized to YAML."""
        config = generate_sample_config()
        yaml_path = tmp_path / "test_config.yaml"
        config.dump_yaml(yaml_path)
        assert yaml_path.exists()
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["title"] == "COVID-19 - Workflow2"
        assert data["max_iteration"] == 4
