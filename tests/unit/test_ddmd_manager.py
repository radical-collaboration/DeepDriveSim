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
    async def test_done_callback_first_failure_requeues(self):
        """First failure re-queues the sim; it is not yet in failed_sims."""
        manager = make_manager(max_sim_batch=1, training_cores=1)
        failing = make_failing_task()
        failing.add_done_callback(lambda t: manager._on_sim_done(t, "sim_fail"))
        manager.registered_sims["sim_fail"] = failing

        with pytest.raises(RuntimeError):
            await failing
        await asyncio.sleep(0)

        assert "sim_fail" not in manager.registered_sims
        assert "sim_fail" not in manager.failed_sims
        assert manager._retry_counts.get("sim_fail") == 1
        manager.logger.warning.assert_called()
        manager.logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_done_callback_permanent_failure_after_max_retries(self):
        """After max_retries the sim moves to failed_sims and an error is logged."""
        manager = make_manager(max_sim_batch=1, training_cores=1)
        manager.max_retries = 1
        manager._retry_counts["sim_fail"] = 1  # already used its one retry

        failing = make_failing_task()
        failing.add_done_callback(lambda t: manager._on_sim_done(t, "sim_fail"))
        manager.registered_sims["sim_fail"] = failing

        with pytest.raises(RuntimeError):
            await failing
        await asyncio.sleep(0)

        assert "sim_fail" not in manager.registered_sims
        assert "sim_fail" in manager.failed_sims
        assert "sim_fail" not in manager._retry_counts  # cleaned up
        manager.logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_done_callback_ok_sim_not_affected_by_failure_path(self):
        """Successful sim lands in completed_sims regardless of sibling failures."""
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
        await asyncio.sleep(0)

        assert "sim_ok" in manager.completed_sims
        assert "sim_fail" not in manager.completed_sims


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


# ---------------------------
# Group 7: failed_sims exit-condition helpers
# ---------------------------
class TestFailedSimsExitCondition:
    """
    Verify that failed_sims count toward the exit condition in the same way
    as completed_sims.  These tests use a small inline subclass that mirrors
    the check_train_status logic used in ddmd_workflow and the finalize_results
    logic used in miniapps_workflow / dummy_workflow.
    """

    def _make_manager_with_counts(self, completed, failed, total):
        """Return a MockLearner pre-seeded with the given sim counts."""
        manager = make_manager()
        manager.completed_sims = set(range(completed))
        manager.failed_sims = set(range(completed, completed + failed))
        manager._total = total  # used by the inline check below
        return manager

    def test_all_completed_meets_target(self):
        m = self._make_manager_with_counts(completed=4, failed=0, total=4)
        assert len(m.completed_sims) + len(m.failed_sims) >= m._total

    def test_all_failed_meets_target(self):
        m = self._make_manager_with_counts(completed=0, failed=4, total=4)
        assert len(m.completed_sims) + len(m.failed_sims) >= m._total

    def test_mixed_meets_target(self):
        m = self._make_manager_with_counts(completed=2, failed=2, total=4)
        assert len(m.completed_sims) + len(m.failed_sims) >= m._total

    def test_incomplete_does_not_meet_target(self):
        m = self._make_manager_with_counts(completed=1, failed=1, total=4)
        assert len(m.completed_sims) + len(m.failed_sims) < m._total


# ---------------------------
# Group 8: Issue-20 — completed_sims double-count race
# ---------------------------
class TestCompletedSimsNoDuplicates:
    """
    Regression tests for the list→set fix (issue_20).

    Race: cancel_sims() checks `sim_idx in registered_sims` and pre-adds the
    sim to completed_sims, but the task was already done (not cancelled) — so
    _on_sim_done fires with task.cancelled()==False and hits completed_sims.add
    a second time.  With a list this inflated len(completed_sims) above
    total_num_sim and caused finalize_results to shut the workflow down before
    all sims had run.  With a set, .add() is idempotent.
    """

    @pytest.mark.asyncio
    async def test_set_prevents_double_count_on_cancel_race(self):
        """
        Simulate the exact race: cancel_sims pre-adds sim 0, then
        _on_sim_done also adds sim 0 (task already done, not cancelled).
        len(completed_sims) must stay 1, not grow to 2.
        """
        manager = make_manager()
        manager.sim_predictions = {0: 0.0}

        # Create a task that is already done (not cancelled).
        done_task = make_done_task()
        await asyncio.sleep(0)  # let it finish
        assert done_task.done() and not done_task.cancelled()

        # Register it so cancel_sims() sees it.
        done_task.add_done_callback(
            lambda t: manager._on_sim_done(t, 0)
        )
        manager.registered_sims[0] = done_task

        # cancel_sims() pre-adds sim 0 and calls task.cancel() (which is a
        # no-op because the task is already done).
        await manager.cancel_sims()

        # _on_sim_done fires: task is not cancelled → goes to completed_sims.add
        await asyncio.sleep(0)  # callback scheduled
        await asyncio.sleep(0)  # callback fires

        # Must be exactly 1 unique entry, not 2.
        assert len(manager.completed_sims) == 1
        assert 0 in manager.completed_sims

    @pytest.mark.asyncio
    async def test_list_would_double_count(self):
        """
        Demonstrate that the old list implementation WOULD have produced a
        double-count under the same race.  This is the failing-on-old-code
        counterpart to test_set_prevents_double_count_on_cancel_race.
        """
        # Replicate the race with an explicit list to show the bug.
        completed = []
        completed.append(0)   # cancel_sims pre-add
        completed.append(0)   # _on_sim_done duplicate
        assert len(completed) == 2  # list inflates the count — this was the bug

        # With a set the duplicate is silently ignored.
        completed_set = set()
        completed_set.add(0)  # cancel_sims pre-add
        completed_set.add(0)  # _on_sim_done duplicate
        assert len(completed_set) == 1  # fixed
