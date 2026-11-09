"""
Unit tests for the agentic MiniApps workflow (LLM steering).

These tests verify:
  - the steering prompt has the configured size and lists every simulation;
  - LLM answers are parsed into per-simulation scores (clamped, unknown sims ignored);
  - the stop decision uses the LLM score (or the stock coin flip when configured);
  - PromptTaskBridge makes an AsyncFlow prompt_task return the answer of a backend that,
    like RHAPSODY's DragonVllmInferenceBackend, only sets task["response"];
  - the CPU stand-in backend answers in the expected format.
"""

import asyncio
import shutil
from unittest.mock import MagicMock

import pytest

pytest.importorskip("radical.asyncflow", reason="radical.asyncflow not installed")


@pytest.fixture()
def tmp_home(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture()
def workflow(tmp_home):
    from workflows.miniapps_workflow.agentic_workflow import AgenticMiniAppsWorkflow

    cfg = {
        "home_dir": str(tmp_home),
        "agentic": {"prompt_tokens": 512, "stop_threshold": 0.5, "seed": 1},
        "llm": {"backend": "standin"},
    }
    return AgenticMiniAppsWorkflow(
        config=cfg, name="test", asyncflow=MagicMock(), home_dir=str(tmp_home)
    )


def test_prompt_size_and_content(workflow):
    from workflows.miniapps_workflow.agentic_workflow import CHARS_PER_TOKEN

    data = [{"sim": k, "state": "finished", "bytes": 10, "age_s": 1.0} for k in (0, 3)]
    prompt = workflow._build_prompt(data)
    target = workflow.prompt_tokens * CHARS_PER_TOKEN
    assert target <= len(prompt) < target + 200
    assert "sim_0:" in prompt and "sim_3:" in prompt
    assert "reasoning" in prompt  # default answer_style
    workflow.answer_style = "scores"
    assert "reasoning" not in workflow._build_prompt(data)


def test_parse_scores(workflow):
    text = "sim_0: 0.9\nsim_3 = 1.7\nsim_9: 0.1\nsim_0: 0.2\ngarbage"
    scores = workflow._parse_scores(text, [0, 3])
    assert scores == {0: 0.9, 3: 1.0}  # clamped, unknown sim ignored, first value kept


def test_stop_decision(workflow):
    assert workflow.stop_simulation(prediction=0.1) is True
    assert workflow.stop_simulation(prediction=0.9) is False
    assert workflow.stop_simulation(prediction=None) is False
    workflow.decision = "coinflip"
    assert workflow.stop_simulation(prediction=0.1) in (True, False)


async def test_prompt_task_bridge():
    from radical.asyncflow import NoopExecutionBackend, WorkflowEngine

    from workflows.miniapps_workflow.llm_backends import PromptTaskBridge

    class VllmLike(NoopExecutionBackend):  # completes like DragonVllmInferenceBackend
        async def submit_tasks(self, tasks):
            for task in tasks:
                self._callback_func(task, "RUNNING")
                task["response"] = [f"answer:{task['prompt']}"]
                self._callback_func(task, "DONE")

    class Bridged(PromptTaskBridge, VllmLike):
        pass

    flow = await WorkflowEngine.create(
        backend=[NoopExecutionBackend(name="compute"), Bridged(name="llm")]
    )

    @flow.prompt_task(backend="llm")
    async def ask(x):
        return f"q{x}"

    try:
        assert await asyncio.wait_for(ask(1), 10) == "answer:q1"
    finally:
        await flow.shutdown()


async def test_standin_backend_answers_every_sim():
    from radical.asyncflow import NoopExecutionBackend, WorkflowEngine

    from workflows.miniapps_workflow.llm_backends import StandInLLMBackend

    flow = await WorkflowEngine.create(
        backend=[
            NoopExecutionBackend(name="compute"),
            StandInLLMBackend(name="llm", latency_s=0.0, per_token_s=0.0),
        ]
    )

    @flow.prompt_task(backend="llm")
    async def ask():
        return "sim_1: a\nsim_4: b"

    try:
        answer = await asyncio.wait_for(ask(), 10)
    finally:
        await flow.shutdown()
    assert [line.split(":")[0] for line in answer.splitlines()] == ["sim_1", "sim_4"]
