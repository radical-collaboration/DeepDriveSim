import pytest

from tests.unit.mock_manager import MockLearner


@pytest.mark.asyncio
async def test_integration():
    manager = MockLearner()

    await manager.start()
    assert manager.registered_sims == {}
    assert manager.sim_task_queue.empty()
    await manager.close()
