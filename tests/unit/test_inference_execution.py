"""Real command cancellation, not merely cancellation of an asyncio future."""

import asyncio
import os
import sys

import pytest

pytest.importorskip("rhapsody")

from workflows.miniapps_workflow.inference_execution import InferenceExecutionBackend

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def command(uid, code):
    return {
        "uid": uid,
        "executable": sys.executable,
        "arguments": ["-u", "-c", code],
        "capture_stdio": True,
    }


async def ready(task, tmp_path):
    async def wait():
        path = tmp_path / f"{task['uid']}.stdout"
        while not path.exists() or "ready" not in path.read_text():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


async def test_cancellation_reaps_command_before_acknowledgement(tmp_path):
    backend = await InferenceExecutionBackend()
    backend._work_dir = str(tmp_path)
    states = []
    backend.register_callback(lambda task, state: states.append(state))
    task = command("cancel", "import time; print('ready'); time.sleep(30)")
    try:
        await backend.submit_tasks([task])
        await ready(task, tmp_path)
        pid = task["command_pid"]
        assert await backend.cancel_task(task["uid"])
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert states == ["RUNNING", "CANCELED"]
    finally:
        await backend.shutdown()


async def test_shutdown_reaps_active_commands(tmp_path):
    backend = await InferenceExecutionBackend()
    backend._work_dir = str(tmp_path)
    tasks = [
        command(str(k), "import time; print('ready'); time.sleep(30)") for k in range(2)
    ]
    await backend.submit_tasks(tasks)
    await asyncio.gather(*(ready(task, tmp_path) for task in tasks))
    pids = [task["command_pid"] for task in tasks]
    await backend.shutdown()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert await backend.state() == "SHUTDOWN"


async def test_command_output_and_failure_status(tmp_path):
    backend = await InferenceExecutionBackend()
    backend._work_dir = str(tmp_path)
    states = []
    backend.register_callback(lambda task, state: states.append(state))
    task = command("failure", "print('real output'); raise SystemExit(3)")
    try:
        futures = await backend.submit_tasks([task])
        await asyncio.gather(*futures)
        assert task["exit_code"] == 3
        assert (tmp_path / "failure.stdout").read_text().strip() == "real output"
        assert states == ["RUNNING", "FAILED"]
    finally:
        await backend.shutdown()
