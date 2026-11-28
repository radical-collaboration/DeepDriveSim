import os
import subprocess
import sys
import textwrap

import pytest

from tests.unit.mock_manager import MockLearner


@pytest.mark.asyncio
async def test_integration():
    manager = MockLearner()

    await manager.start()
    assert manager.registered_sims == {}
    assert manager.sim_task_queue.empty()
    await manager.close()


class TestRunWorkflowErrorHandling:
    """run_workflow.py must propagate errors, show a traceback, suppress the banner."""

    def _run(self, script, config_text, tmp_path, env=None):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(config_text))
        e = {**os.environ, **(env or {})}
        return subprocess.run(
            [sys.executable, script, "--config_file", str(cfg)],
            capture_output=True,
            text=True,
            env=e,
        )

    def test_bad_config_path_exits_nonzero(self, tmp_path):
        """Missing config → FileNotFoundError, non-zero exit, traceback shown."""
        result = subprocess.run(
            [
                sys.executable,
                "workflows/miniapps_workflow/run_workflow.py",
                "--config_file",
                str(tmp_path / "nonexistent.yaml"),
            ],
            capture_output=True,
            text=True,
            cwd="/work/nvme/bdyk/mgoliyad1/DeepDriveSim",
        )
        assert result.returncode != 0
        assert "Traceback" in result.stderr
        assert "complete" not in result.stdout

    def test_bad_config_path_no_banner_ddmd(self, tmp_path):
        """ddmd run_workflow.py also propagates errors and shows traceback."""
        result = subprocess.run(
            [
                sys.executable,
                "workflows/ddmd_workflow/run_workflow.py",
                "--config",
                str(tmp_path / "nonexistent.yaml"),
            ],
            capture_output=True,
            text=True,
            cwd="/work/nvme/bdyk/mgoliyad1/DeepDriveSim",
        )
        assert result.returncode != 0
        assert "Traceback" in result.stderr
        assert "complete" not in result.stdout
