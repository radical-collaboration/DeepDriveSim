from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from tests.unit.mock_manager import MockLearner


@pytest.mark.asyncio
async def test_integration():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    manager = MockLearner(asyncflow=asyncflow)

    await manager.start()
    assert manager.registered_sims == {}
    assert manager.sim_task_queue.empty()
    await manager.close()
