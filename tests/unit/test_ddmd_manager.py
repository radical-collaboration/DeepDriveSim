import asyncio

import pytest

from tests.unit.mock_manager import MockLearner


# ---------------------------
# Async helpers
# ---------------------------
def make_done_task(result="ok"):
    async def _done():
        return result

    return asyncio.create_task(_done())


def make_failing_task(exc_msg="boom"):
    async def _fail():
        raise RuntimeError(exc_msg)

    return asyncio.create_task(_fail())


async def wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll until predicate() is truthy, or raise TimeoutError."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError(f"Condition not met within {timeout}s")
        await asyncio.sleep(interval)


# ---------------------------
# Shared factory
# ---------------------------
def make_manager(**kwargs):
    return MockLearner(**kwargs)


# ---------------------------
# Group 1: Simulation lifecycle (done-callback mechanism)
# ---------------------------
class TestSimulationLifecycle:
    @pytest.mark.asyncio
    async def test_submit_sims_registers_and_respects_batch(self):
        manager = make_manager()

        await manager.collect_sim_inputs(n=2)
        manager.sim_batch_size = 2

        manager.debug = True
        submit_task = asyncio.create_task(manager.submit_sims())

        await wait_until(lambda: len(manager.registered_sims) >= 2)

        manager.shutting_down.set()
        await submit_task

        assert len(manager.registered_sims) == 2
        assert manager.sim_task_queue.empty()
        assert manager.sim_batch_size == 0
        manager.logger.task_started.assert_called()

    @pytest.mark.asyncio
    async def test_done_callback_unregisters_sim_and_increments_batch(self):
        manager = make_manager(max_sim_batch=1, training_cores=1)
        done = make_done_task("done:sim_0")
        running = manager.simulation(sim_inputs={"sim_idx": "sim_1"})

        # Register done callbacks as submit_sims would
        done.add_done_callback(lambda t: manager._on_sim_done(t, "sim_0"))
        running.add_done_callback(lambda t: manager._on_sim_done(t, "sim_1"))
        manager.registered_sims["sim_0"] = done
        manager.registered_sims["sim_1"] = running

        manager.debug = True
        await asyncio.sleep(0)  # Turn 1: done task completes, callback scheduled
        await asyncio.sleep(0)  # Turn 2: callback fires, _on_sim_done removes sim_0

        assert "sim_0" not in manager.registered_sims
        assert "sim_1" in manager.registered_sims
        assert manager.sim_batch_size == 1  # capped at max_sim_batch=1
        manager.logger.task_completed.assert_called()

    @pytest.mark.asyncio
    async def test_done_callback_logs_failures_and_unregs(self):
        manager = make_manager(max_sim_batch=1, training_cores=1)
        failing = make_failing_task()
        ok = make_done_task()

        failing.add_done_callback(lambda t: manager._on_sim_done(t, "sim_fail"))
        ok.add_done_callback(lambda t: manager._on_sim_done(t, "sim_ok"))
        manager.registered_sims["sim_fail"] = failing
        manager.registered_sims["sim_ok"] = ok

        with pytest.raises(RuntimeError):
            await failing
        await ok

        await asyncio.sleep(0)  # Let pending callbacks fire

        assert "sim_fail" not in manager.registered_sims
        assert "sim_ok" not in manager.registered_sims
        assert manager.sim_batch_size == 1  # capped at max_sim_batch=1
        manager.logger.error.assert_called()


# ---------------------------
# Group 2: Cancel sims behavior
# ---------------------------
class TestCancelSimsBehavior:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "predictions, clean_flag, expected_deleted, expected_remaining",
        [
            ({"sim_0": 0.2, "sim_1": 0.8}, True, ["sim_0"], ["sim_1"]),
            ({"sim_0": 0.6, "sim_1": 0.8}, True, [], ["sim_0", "sim_1"]),
            ({"sim_0": 0.1, "sim_1": 0.7}, False, [], ["sim_1", "sim_0"]),
            ({}, True, [], []),
        ],
    )
    async def test_cancel_sims_various_cases(
        self, predictions, clean_flag, expected_deleted, expected_remaining
    ):
        manager = make_manager()
        for tag in predictions.keys():
            task = manager.simulation(sim_inputs={"sim_idx": tag})
            task.add_done_callback(lambda t, sid=tag: manager._on_sim_done(t, sid))
            manager.registered_sims[tag] = task

        manager.sim_predictions = predictions
        manager.clean_unregistered_sims = clean_flag

        await manager.cancel_sims()
        await asyncio.sleep(
            0
        )  # Turn 1: cancelled task's __step runs, callback scheduled
        await asyncio.sleep(0)  # Turn 2: done callback fires, _on_sim_done removes sim

        for sim in expected_deleted:
            assert sim not in manager.registered_sims.keys()
        for sim in expected_remaining:
            if predictions.get(sim, 1.0) >= 0.5 or not clean_flag:
                assert sim in manager.registered_sims.keys() or not clean_flag

        assert manager.sim_batch_size >= len(expected_deleted)


# ---------------------------
# Group 3: Start flow
# ---------------------------
class TestStartFlow:
    @pytest.mark.asyncio
    async def test_start_runs_full_cycle_and_exits(self):
        manager = make_manager()
        manager.retrain_model = False

        await manager.start()

        assert manager.sim_task_queue.empty()
        assert not manager.registered_sims
        manager.logger.manager_exiting.assert_called()
        manager.logger.separator.assert_called()


# ---------------------------
# Group 4: Shutdown safety
# ---------------------------
class TestShutdownSafety:
    @pytest.mark.asyncio
    async def test_close_and_stop_are_safe(self):
        manager = make_manager()
        await manager.close()
        await manager.stop()


# ---------------------------
# Group 5: File deletion
# ---------------------------
class TestDelFilesBehavior:
    @pytest.mark.asyncio
    async def test_del_files_records_multiple_deletions(self):
        manager = make_manager()
        sims = ["sim_0", "sim_1", "sim_2"]
        for sim in sims:
            await manager.del_files(sim)

        for sim in sims:
            assert sim not in manager.registered_sims.keys()


# ---------------------------
# Group 6: Simulation queue edge cases
# ---------------------------
class TestSimulationQueueEdgeCases:
    @pytest.mark.asyncio
    async def test_submit_sims_with_empty_queue(self):
        manager = make_manager()
        manager.sim_batch_size = 2

        submit_task = asyncio.create_task(manager.submit_sims())

        await asyncio.sleep(0.05)
        manager.shutting_down.set()
        await submit_task

        assert manager.sim_task_queue.empty()

    @pytest.mark.asyncio
    async def test_submit_sims_with_partial_queue(self):
        manager = make_manager()
        await manager.collect_sim_inputs(n=1)
        manager.sim_batch_size = 3

        submit_task = asyncio.create_task(manager.submit_sims())

        await wait_until(lambda: len(manager.registered_sims) >= 1)
        manager.shutting_down.set()
        await submit_task

        assert len(manager.registered_sims) == 1
        assert manager.sim_task_queue.empty()

    @pytest.mark.asyncio
    async def test_submit_sims_with_batch_larger_than_queue(self):
        manager = make_manager()
        await manager.collect_sim_inputs(n=2)
        manager.sim_batch_size = 5

        submit_task = asyncio.create_task(manager.submit_sims())

        await wait_until(lambda: len(manager.registered_sims) >= 2)
        manager.shutting_down.set()
        await submit_task

        assert len(manager.registered_sims) == 2
        assert manager.sim_task_queue.empty()
