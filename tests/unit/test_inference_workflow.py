"""Inference-only progress/cleanup using toy simulation tasks, no research runs."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from workflows.miniapps_workflow.inference_workflow import InferenceOnlyMiniAppsWorkflow


class ToyFlow:
    def block(self, function):
        return function

    def executable_task(self, **kwargs):
        return lambda function: function

    def prompt_task(self, **kwargs):
        return lambda function: function


@pytest.fixture
def workflow(tmp_path):
    cfg = {
        "home_dir": str(tmp_path),
        "total_num_sim": 3,
        "max_sim_batch": 3,
        "miniapps_data_ready": 1,
        "agentic": {"prompt_tokens": 32, "steering_interval_s": 0.001},
        "llm": {"backend": "compute"},
    }
    result = InferenceOnlyMiniAppsWorkflow(config=cfg, name="tiny", asyncflow=ToyFlow())
    result.training = AsyncMock(side_effect=AssertionError("no training"))
    result.train_model = AsyncMock(side_effect=AssertionError("no training"))
    result.monitor_training_data = AsyncMock(side_effect=AssertionError("no training"))
    result._free_resources_for_training = AsyncMock(
        side_effect=AssertionError("do not reserve/cancel for training")
    )
    return result


def install_sims(workflow, delays, fail=None):
    async def simulate(index):
        await asyncio.sleep(delays[index])
        if index == fail:
            raise RuntimeError("toy sim failed")
        root = Path(workflow.sim_output_dir, "phase0")
        root.mkdir(exist_ok=True)
        (root / f"data_0_{index}.h5").write_bytes(b"observation")

    workflow.simulation = lambda sim_inputs: asyncio.create_task(
        simulate(sim_inputs["sim_idx"])
    )


async def test_progress_cancel_and_completion_without_training(workflow):
    install_sims(workflow, [0.01, 2, 2])
    workflow.llm_steering = AsyncMock(return_value="sim_0: 0.0\nsim_1: 0.0\nsim_2: 0.0")
    await asyncio.wait_for(workflow.start(), 2)
    assert sorted(workflow.completed_sims) == [0, 1, 2]
    assert workflow.finished_sim_ids == [0]
    assert sorted(workflow.stopped_sim_ids) == [1, 2]
    assert workflow.iteration >= 1
    assert workflow.shutting_down.is_set()
    assert not workflow.registered_sims
    assert workflow.sim_task_queue.empty()  # cancelled decisions were not requeued
    assert workflow.training.await_count == 0
    assert workflow.train_model.await_count == 0
    assert workflow.monitor_training_data.await_count == 0
    assert workflow._free_resources_for_training.await_count == 0
    assert not workflow.home_dir.exists()


async def test_completed_task_is_not_double_counted_as_a_cancellation(workflow):
    task = asyncio.get_running_loop().create_future()
    task.set_result(None)
    workflow.registered_sims[0] = task
    workflow.sim_predictions = {0: 0.0}
    await workflow.cancel_sims()
    assert workflow.completed_sims == []
    workflow._on_sim_done(task, 0)
    assert workflow.completed_sims == [0]


async def test_simulations_can_finish_without_an_inference_request(workflow):
    install_sims(workflow, [0, 0, 0])
    workflow.llm_steering = AsyncMock(side_effect=AssertionError("all done"))
    await asyncio.wait_for(workflow.start(), 2)
    assert sorted(workflow.completed_sims) == [0, 1, 2]
    assert workflow.llm_steering.await_count == 0


async def test_simulation_failure_exits_and_cleans_up(workflow):
    install_sims(workflow, [0.01, 2, 2], fail=0)
    with pytest.raises(RuntimeError, match="simulation failed"):
        await asyncio.wait_for(workflow.start(), 2)
    assert not workflow.registered_sims
    assert workflow.sim_task_queue.empty()
    assert workflow.shutting_down.is_set()


async def test_inference_failure_exits_and_cleans_up(workflow):
    install_sims(workflow, [0.01, 2, 2])
    workflow.llm_steering = AsyncMock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        await asyncio.wait_for(workflow.start(), 2)
    assert not workflow.registered_sims
    assert workflow.sim_task_queue.empty()


async def test_external_cancellation_does_not_requeue_tasks(workflow):
    install_sims(workflow, [2, 2, 2])
    task = asyncio.create_task(workflow.start())
    while not workflow.registered_sims:
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not workflow.registered_sims
    assert workflow.sim_task_queue.empty()
    assert not workflow.home_dir.exists()


async def test_visible_output_waits_for_successful_producer(workflow):
    release = asyncio.Event()
    root = Path(workflow.sim_output_dir, "phase0")
    root.mkdir()

    async def simulate(index):
        if index == 0:
            (root / "data_0_0.h5").write_bytes(b"not closed by producer yet")
            await release.wait()
        else:
            await asyncio.sleep(5)

    workflow.simulation = lambda sim_inputs: asyncio.create_task(
        simulate(sim_inputs["sim_idx"])
    )
    workflow.llm_steering = AsyncMock(return_value="sim_1: 0.0\nsim_2: 0.0")
    task = asyncio.create_task(workflow.start())
    try:
        while not (root / "data_0_0.h5").exists():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.02)
        assert workflow.llm_steering.await_count == 0
        release.set()
        await asyncio.wait_for(task, 2)
        assert workflow.llm_steering.await_count > 0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_existing_file_does_not_mislabel_running_sim(workflow):
    root = Path(workflow.sim_output_dir, "phase0")
    root.mkdir()
    (root / "data_0_1.h5").touch()
    workflow.registered_sims[1] = object()
    assert workflow._collect_data()[0]["state"] == "running"


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), "1"])
def test_poll_interval_is_bounded(tmp_path, interval):
    with pytest.raises(ValueError, match="finite and positive"):
        InferenceOnlyMiniAppsWorkflow(
            config={"agentic": {"steering_interval_s": interval}},
            asyncflow=ToyFlow(),
            home_dir=str(tmp_path),
        )


async def test_real_cpu_compute_steers_toy_simulations(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("radical.asyncflow")
    from radical.asyncflow import NoopExecutionBackend, WorkflowEngine

    from workflows.miniapps_workflow.llm_backends import make_llm_backend

    backend, _ = await make_llm_backend(
        {
            "backend": "compute",
            "compute": {
                "device": "cpu",
                "new_tokens": 2,
                "spec": {
                    "hidden_size": 16,
                    "intermediate_size": 24,
                    "num_layers": 1,
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "vocab_size": 32,
                    "max_prompt_tokens": 256,
                },
            },
        }
    )
    flow = await WorkflowEngine.create(
        backend=[NoopExecutionBackend(name="compute"), backend]
    )
    workflow = InferenceOnlyMiniAppsWorkflow(
        config={
            "total_num_sim": 3,
            "max_sim_batch": 3,
            "agentic": {
                "prompt_tokens": 32,
                "stop_threshold": 1.1,
                "steering_interval_s": 0.01,
            },
            "llm": {"backend": "compute"},
        },
        asyncflow=flow,
        home_dir=str(tmp_path),
        name="compute-toy",
    )
    install_sims(workflow, [0.01, 5, 5])
    try:
        await asyncio.wait_for(workflow.start(), 8)
        assert workflow.requests
        assert all(r["status"] == "completed" for r in workflow.requests)
        assert workflow.requests[0]["llm_backend"] == "compute"
        assert sorted(workflow.completed_sims) == [0, 1, 2]
        assert not hasattr(workflow, "training")
        assert not workflow.registered_sims
    finally:
        await flow.shutdown()


def test_real_kernel_command_bounds_and_context(tmp_path):
    import shlex

    workflow = InferenceOnlyMiniAppsWorkflow(
        config={
            "simulation": {
                "mat_size": 256,
                "read_size": 4096,
                "write_size": 4096,
                "steps_by_sim": {1: 50},
            }
        },
        asyncflow=ToyFlow(),
        home_dir=str(tmp_path),
    )
    arguments = shlex.split(workflow._sim_cmd(1))
    assert arguments[-8:] == [
        "--num_step",
        "50",
        "--mat_size",
        "256",
        "--read_size",
        "4096",
        "--write_size",
        "4096",
    ]
    assert "Steering context: replica=miniapps iteration=0" in workflow._build_prompt(
        []
    )


@pytest.mark.parametrize(
    "simulation",
    [
        {"mat_size": -1},
        {"read_size": "100"},
        {"steps_by_sim": {1: -1}},
        {"steps_by_sim": []},
        {"bogus": 1},
        ["mat_size"],
    ],
)
def test_invalid_simulation_settings(tmp_path, simulation):
    with pytest.raises(ValueError):
        InferenceOnlyMiniAppsWorkflow(
            config={"simulation": simulation},
            asyncflow=ToyFlow(),
            home_dir=str(tmp_path),
        )


def test_training_tasks_are_not_registered(tmp_path):
    workflow = InferenceOnlyMiniAppsWorkflow(
        config={}, asyncflow=ToyFlow(), home_dir=str(tmp_path)
    )
    assert not hasattr(workflow, "training")
    assert not hasattr(workflow, "prediction")
    assert workflow.sim_batch_size == workflow.max_sim_batch
    assert workflow.training_cores == 0
    assert workflow.retrain_model is False
