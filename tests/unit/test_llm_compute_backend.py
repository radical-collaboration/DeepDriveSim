"""Backend contracts with a tiny fake workload; no torch/GPU required."""

import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("radical.asyncflow")

from workflows.miniapps_workflow.inference_compute import InferenceSpec
from workflows.miniapps_workflow.llm_compute_backend import (
    ComputeLLMBackend,
    seeded_steering_scores,
)


def expected_answer(prompt, seed=0):
    return "\n".join(
        f"sim_{k}: {score:.8f}"
        for k, score in seeded_steering_scores(prompt, seed).items()
    )


def test_seeded_scores_are_pinned_to_recorded_values():
    # Recorded from full mini-app runs on CPU and GPU; pins the policy against
    # accidental change.
    prompt = "Steering context: replica=min1 iteration=1\nsim_0 sim_1 sim_2 sim_3"
    assert seeded_steering_scores(prompt, 17) == {
        0: 0.15975377612343167,
        1: 0.7522643103032313,
        2: 0.6224790290085702,
        3: 0.5521004388072782,
    }


def test_seeded_scores_ignore_order_and_device_content():
    context = "Steering context: replica=min1 iteration=3\n"
    a = seeded_steering_scores(context + "sim_1 sim_2", 17)
    b = seeded_steering_scores(context + "sim_2 sim_1 sim_1", 17)
    assert a == b
    assert all(0 <= value < 1 for value in a.values())
    assert a != seeded_steering_scores(context + "sim_1 sim_2", 18)
    assert a != seeded_steering_scores(context.replace("=3", "=4") + "sim_1 sim_2", 17)


class FakeWorkload:
    def __init__(self, spec=None, device="cpu"):
        self.spec = spec or InferenceSpec()
        self.device = SimpleNamespace(type=device)
        self.calls = []
        self.error = None

    def run(self, batch, prompt, new):
        self.calls.append((batch, prompt, new))
        time.sleep(0.01)  # stand-in to check the event loop remains responsive
        if self.error:
            raise self.error
        return {
            "token_ids": [[0] * new for _ in range(batch)],
            "batch_size": batch,
            "prompt_tokens": prompt,
            "new_tokens": new,
        }


def start_backend(workload=None, **kwargs):
    backend = ComputeLLMBackend(workload or FakeWorkload(), **kwargs)
    states = []
    backend.register_callback(lambda task, state: states.append((task, state)))
    backend.start()
    return backend, states


async def wait_terminal(states, count):
    async def wait():
        while sum(s in ("DONE", "FAILED") for _, s in states) < count:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 3)


async def test_batch_padding_actual_compute_then_scores():
    workload = FakeWorkload()
    backend, states = start_backend(workload, new_tokens=2)
    tasks = [{"uid": "a", "prompt": "sim_1 sim_3"}, {"uid": "b", "prompt": "sim_4"}]
    try:
        await backend.submit_tasks(tasks)
        await wait_terminal(states, 2)
        assert workload.calls == [(2, 3, 2)]  # ceil(11 chars / 4)
        assert tasks[0]["stdout"] == expected_answer(tasks[0]["prompt"])
        assert tasks[1]["inference_compute"]["original_prompt_tokens"] == 2
        assert tasks[1]["inference_compute"]["prompt_tokens"] == 3
    finally:
        await backend.shutdown()


async def test_seeded_decisions_follow_successful_compute_and_ignore_tokens():
    workload = FakeWorkload()
    backend, states = start_backend(workload, steering_seed=17)
    tasks = [{"uid": str(k), "prompt": "sim_1 sim_2"} for k in range(2)]
    try:
        await backend.submit_tasks(tasks[:1])
        await wait_terminal(states, 1)
        workload.run = lambda batch, prompt, new: {
            "token_ids": [[99] * new for _ in range(batch)]
        }
        await backend.submit_tasks(tasks[1:])
        await wait_terminal(states, 2)
        assert (
            tasks[0]["stdout"]
            == tasks[1]["stdout"]
            == expected_answer("sim_1 sim_2", 17)
        )
        assert workload.calls  # real-workload interface was invoked before replying
        assert backend.state() == "RUNNING"
    finally:
        await backend.shutdown()
    assert backend.state() == "SHUTDOWN"


async def test_prompt_limit_and_queue_limit_fail_instead_of_truncating():
    backend, states = start_backend(max_pending=1)
    tasks = [
        {"uid": "long", "prompt": "x" * 2049},
        {"uid": "ok", "prompt": "sim_0"},
        {"uid": "full", "prompt": "sim_1"},
    ]
    try:
        await backend.submit_tasks(tasks)
        await wait_terminal(states, 3)
        assert [(t["uid"], s) for t, s in states if s == "FAILED"] == [
            ("long", "FAILED"),
            ("full", "FAILED"),
        ]
        assert "max_prompt_tokens" in tasks[0]["stderr"]
        assert "queue is full" in tasks[2]["stderr"]
    finally:
        await backend.shutdown()


async def test_execution_failure_resolves_request_and_backend_remains_usable():
    workload = FakeWorkload()
    workload.error = RuntimeError("operator unavailable")
    backend, states = start_backend(workload)
    try:
        await backend.submit_tasks([{"uid": "bad", "prompt": "sim_0"}])
        await wait_terminal(states, 1)
        assert states[-1][1] == "FAILED"
        workload.error = None
        await backend.submit_tasks([{"uid": "good", "prompt": "sim_0"}])
        await wait_terminal(states, 2)
        assert states[-1][1] == "DONE"
    finally:
        await backend.shutdown()


async def test_shutdown_finishes_active_work_fails_queued_and_rejects_new():
    workload = FakeWorkload(InferenceSpec(max_batch_size=1))
    backend, states = start_backend(workload)
    tasks = [{"uid": "a", "prompt": "sim_0"}, {"uid": "b", "prompt": "sim_1"}]
    await backend.submit_tasks(tasks)
    while not workload.calls:
        await asyncio.sleep(0.001)
    await asyncio.wait_for(backend.shutdown(), 3)
    assert [(t["uid"], s) for t, s in states if s in ("DONE", "FAILED")] == [
        ("a", "DONE"),
        ("b", "FAILED"),
    ]
    await backend.shutdown()  # idempotent
    await backend.submit_tasks([{"uid": "late", "prompt": "sim_0"}])
    assert states[-1][1] == "FAILED"


async def test_factory_integration_with_asyncflow(monkeypatch):
    from radical.asyncflow import NoopExecutionBackend, WorkflowEngine

    from workflows.miniapps_workflow import llm_compute_backend as module
    from workflows.miniapps_workflow.llm_backends import make_llm_backend

    monkeypatch.setattr(module, "InferenceWorkload", FakeWorkload)
    backend, info = await make_llm_backend({"backend": "compute"})
    assert info["num_gpus"] == 0
    assert len(info["compute_spec_sha256"]) == 64
    assert (
        '"fidelity": "synthetic_decoder_not_measured_trace"'
        in (info["compute_spec_json"])
    )
    flow = await WorkflowEngine.create(
        backend=[NoopExecutionBackend(name="compute"), backend]
    )

    @flow.prompt_task(backend="llm")
    async def ask():
        return "sim_2 sim_7"

    try:
        assert await asyncio.wait_for(ask(), 5) == expected_answer("sim_2 sim_7")
    finally:
        await flow.shutdown()


@pytest.mark.parametrize(
    "config",
    [
        {"new_tokens": 0},
        {"new_tokens": 9999},
        {"max_pending": 0},
        {"steering_seed": -1},
        {"spec": {"hidden_size": 15}},
    ],
)
async def test_factory_validates_before_allocating(config, monkeypatch):
    from workflows.miniapps_workflow import llm_compute_backend as module

    def forbidden(*args):
        raise AssertionError("must not allocate")

    monkeypatch.setattr(module, "InferenceWorkload", forbidden)
    with pytest.raises(ValueError):
        await ComputeLLMBackend.create(config)


def test_service_event_accepts_compute_provenance():
    pytest.importorskip("rhapsody.telemetry")
    from workflows.miniapps_workflow.agentic_workflow import InferenceService
    from workflows.miniapps_workflow.miniapps_workflow import make_event

    event = make_event(
        InferenceService,
        session_id="unit",
        backend="concurrent",
        compute_spec_json='{"device": "cpu"}',
        compute_spec_sha256="a" * 64,
    )
    assert event.compute_spec_sha256 == "a" * 64
    assert event.compute_spec_json == '{"device": "cpu"}'
    assert event.emulator_profile == ""


async def test_real_cpu_backend_through_prompt_task():
    pytest.importorskip("torch")
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
                    "max_prompt_tokens": 16,
                },
            },
        }
    )
    flow = await WorkflowEngine.create(
        backend=[NoopExecutionBackend(name="compute"), backend]
    )

    @flow.prompt_task(backend="llm")
    async def ask():
        return "sim_1 sim_2"

    try:
        text = await asyncio.wait_for(ask(), 5)
        assert "sim_1:" in text and "sim_2:" in text
    finally:
        await flow.shutdown()
