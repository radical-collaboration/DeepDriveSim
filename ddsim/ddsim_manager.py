#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Async-friendly DDSim manager for orchestrating simulations
# ------------------------------------------------------------------------------

import asyncio
from collections import OrderedDict
from typing import Any, Optional

from ddsim.logger import Logger


class DDSimManager:
    """
    Orchestrates the scheduling, monitoring, and cancellation of simulations
    in an AI-steered ensemble simulation workflow.

    Subclasses must define:
        - sim_batch_size, max_sim_batch, training_cores
        - retrain_model (flag to control training loop)
        - init_sim_queue, check_train_status, train_model
        - stop_simulation (returns True if sim should be canceled)
        - add_sims_to_queue (re-queue paused sims)
        - post_process_sim (per-simulation cleanup)
        - evaluate_simulations (collect prediction scores)
        - close
    Optional:
        - free_resources_for_train (default False)
    """

    def __init__(self, resource_manager: Optional[Any] = None):
        self.logger = Logger(use_colors=True)
        self.registered_sims = OrderedDict()  # Active simulations: {tag: asyncio.Task}
        self.sim_task_queue = asyncio.Queue()  # Queue of pending simulation inputs
        self.completed_sims = []  # Completed simulations
        self.sim_predictions = {}
        self.train_models = []  # Additional training coroutines to run in parallel

        self.sleep_time = 20  # Delay between prediction/start train checks
        self.debug = False

        # Workflow flags — set these in your workflow subclass to control behavior:
        self.call_finalize_results = False
        self.call_evaluate_simulations = False
        self.run_workflow = True
        self.free_resources_for_train = False
        self.call_cancel_simulations = False

        # ---- resource manager ----------------------------------------
        self._rm = resource_manager

        # Sims permanently killed by cancel_sims (prediction-based).
        # _on_sim_done uses this to distinguish "re-queue" vs "count as done".
        self._perm_cancelled: set = set()

        # Sims whose RM slot was preempted while running.
        # The RM already reclaimed their resources, so _on_sim_done must NOT
        # call rm.release() for these — that would log a spurious warning and
        # trigger an unnecessary _try_dispatch_locked.
        self._rm_preempted_ids: set = set()

        # Event should be set inside workflow code to stop simulation loop
        self.shutting_down = asyncio.Event()
        self.logger.info("DDSim Manager initialized...")

    async def _wait_for_resource(
        self, task_id: str, task_type: str, cancel_handle: list = None
    ) -> None:
        """
        Request a resource slot from the ResourceManager and suspend until
        it is granted.

        Bridges the thread-safe ResourceManager callbacks to the asyncio
        event loop via ``loop.call_soon_threadsafe``.

        Parameters
        ----------
        cancel_handle : list, optional
            Single-element mutable list.  After this method returns (slot
            granted) the caller should append the running asyncio.Task to it.
            If the RM later preempts the slot while the task is running,
            ``cancel_handle[0].cancel()`` is called automatically.

        Raises
        ------
        asyncio.CancelledError
            If the slot is preempted before being granted.
        """
        assert self._rm is not None, (
            "_wait_for_resource called without a ResourceManager"
        )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()

        def on_granted() -> None:
            loop.call_soon_threadsafe(fut.set_result, None)

        def on_preempted() -> None:
            # IMPORTANT: the fut.done() check MUST run in the asyncio thread.
            # on_granted schedules fut.set_result via call_soon_threadsafe; if
            # on_preempted checked fut.done() here (in the RM thread) it would
            # see False, schedule set_exception, and then both callbacks fire in
            # the event loop — set_result first, set_exception second → raises
            # InvalidStateError.  Running the check inside the callback makes it
            # atomic with the action.
            def _handle() -> None:
                if not fut.done():
                    # Pending phase — cancel the waiting future.
                    fut.set_exception(
                        asyncio.CancelledError(
                            f"[Sim {task_id}] resource slot was preempted"
                        )
                    )
                elif cancel_handle:
                    # Running phase — RM already reclaimed the slot.
                    # Mark before cancelling so _on_sim_done skips release().
                    self._rm_preempted_ids.add(task_id)
                    cancel_handle[0].cancel()

            loop.call_soon_threadsafe(_handle)

        cfg = self.tasks_config[task_type]
        priority = int(cfg.get("priority", 10))
        task_cpus = int(cfg.get("cores_per_rank", 1))
        task_gpus = float(cfg.get("gpus_per_rank", 0.0))

        if self.debug:
            self.logger.debug(
                f"[Sim {task_id}] Requesting resource slot, "
                f"workflow_id={self.workflow_id}, task_type={task_type}, "
                f"priority={priority}, "
                f"cpus={task_cpus}, gpus={task_gpus}"
            )

        self._rm.request(
            task_id=task_id,
            workflow_id=self.workflow_id,
            task_type=task_type,
            priority=priority,
            cpus=task_cpus,
            gpus=task_gpus,
            on_granted=on_granted,
            on_preempted=on_preempted,
        )
        await fut

    # --------------------------------------------------------------------------
    def add_sims_to_queue(self, *args, **kwargs):
        """
        Specify how to add sim back to queue after it was canceled to free resources.
        """
        raise NotImplementedError("add_sims_to_queue must be implemented")

    # --------------------------------------------------------------------------
    def post_process_sim(self, *args, **kwargs):
        """
        Specify any post process required after each sim completed
        """
        raise NotImplementedError("post_process_sim must be implemented")

    # --------------------------------------------------------------------------
    def stop_simulation(self, *args, **kwargs):
        """Decide whether to cancel a simulation based on prediction.
        Returns False by default (no early cancellation).
        Override in subclass to implement prediction-based stopping logic.
        """
        raise NotImplementedError("stop_simulation must be implemented")

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """
        Collect all simulation input files into task queue (sim_task_queue).
        Override this with actual logic in workflow subclass.
        """
        raise NotImplementedError("init_sim_queue must be implemented")

    # --------------------------------------------------------------------------
    async def check_train_status(self):
        """
        Check if enough training data is available to start training.
        Override this with actual logic in workflow subclass.
        """
        raise NotImplementedError("check_train_status must be implemented")

    # --------------------------------------------------------------------------
    async def train_model(self):
        """Run the primary model training step.

        Override in subclass. For parallel training of multiple models,
        append async callables to ``self.train_models``; they will be
        gathered concurrently alongside this method in ``start()``.

        If only ``self.train_models`` is used, this method can be left
        as the default no-op.
        """
        pass

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner.
        Override this with actual logic in workflow subclass.
        """
        raise NotImplementedError("close must be implemented")

    # --------------------------------------------------------------------------
    async def stop(self):
        """Alias for close(), can be used for external termination."""
        await self.close()

    # --------------------------------------------------------------------------
    def _on_sim_done(self, task: asyncio.Task, sim_idx) -> None:
        """
        Done-callback registered on every simulation task.

        Releases the RM slot, unregisters the sim, increments the batch
        counter, and logs the outcome.  Fires automatically when the task
        finishes for any reason (success, exception, or cancellation).

        Cancellation handling:
          - Permanently killed by cancel_sims (prediction-based): sim_idx is in
            _perm_cancelled — already counted in completed_sims, no re-queue.
          - Cancelled to free resources (monitor_training_data): re-queue via
            add_sims_to_queue so the sim runs again after training.
        """
        if self._rm is not None:
            if sim_idx in self._rm_preempted_ids:
                # RM preempted this slot while the task was running — resources
                # are already reclaimed; calling release() would log a spurious
                # warning and trigger an unnecessary dispatch cycle.
                self._rm_preempted_ids.discard(sim_idx)
            else:
                self._rm.release(sim_idx)

        self.registered_sims.pop(sim_idx, None)
        self.sim_batch_size += 1

        if task.cancelled():
            if sim_idx in self._perm_cancelled:
                # Prediction-based kill — already in completed_sims, just clean up.
                self._perm_cancelled.discard(sim_idx)
            else:
                # Resource-freeing cancel — re-queue for later re-execution.
                self.logger.info(
                    f"Sim {sim_idx} cancelled for resources — re-queuing",
                    component="simulation",
                )
                asyncio.ensure_future(self.add_sims_to_queue([sim_idx]))
        else:
            exc = task.exception()
            if exc:
                self.logger.error(
                    f"Sim {sim_idx} failed: {exc}", component="simulation"
                )
            else:
                self.completed_sims.append(sim_idx)
                self.logger.task_completed(f"Sim {sim_idx}", component="simulation")
                asyncio.ensure_future(self.post_process_sim(sim_idx))

    # --------------------------------------------------------------------------
    async def submit_sims(self):
        """Submit simulations from the queue and register them."""
        while not self.shutting_down.is_set():

            if self.sim_batch_size <= 0:
                await asyncio.sleep(1)
                continue

            num_to_submit = min(self.sim_batch_size, self.sim_task_queue.qsize())
            for _ in range(num_to_submit):
                try:
                    sim_inputs = self.sim_task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    self.logger.info("No more simulation inputs in queue.")
                    break

                if sim_inputs is None:
                    break

                sim_idx = sim_inputs["sim_idx"]
                cancel_handle = []
                if self._rm is not None:
                    await self._wait_for_resource(
                        sim_idx, "simulation", cancel_handle=cancel_handle
                    )

                simul = self.simulation(sim_inputs=sim_inputs)

                if self._rm is not None:
                    # Wire RM running-phase preemption → asyncio Task cancellation.
                    # on_preempted checks cancel_handle[0].cancel() once the
                    # slot is granted and the task is live.
                    cancel_handle.append(simul)

                # _on_sim_done handles RM release + unregistration when the
                # task finishes.
                simul.add_done_callback(
                    lambda t, sid=sim_idx: self._on_sim_done(t, sid)
                )

                self.logger.task_started(f"Sim {sim_idx}", component="simulation")
                self.registered_sims[sim_idx] = simul

            if num_to_submit > 0:
                self.logger.info(f"Submitted {num_to_submit} new simulation(s)")

            self.sim_batch_size -= num_to_submit
            await asyncio.sleep(0.1)

    # --------------------------------------------------------------------------
    async def monitor_training_data(self):
        """
        Poll until training data is ready.

        The ResourceManager handles preemption of running sims automatically
        when train_model requests a higher-priority slot: it selects the
        lowest-priority running tasks as victims, fires their on_preempted
        callbacks (which call simul.cancel() via cancel_handle), reclaims
        their resources, and grants the train slot.
        """
        while not await self.check_train_status():
            await asyncio.sleep(self.sleep_time)
        self.logger.info("Training data ready.")

    # --------------------------------------------------------------------------
    async def _free_resources_no_rm(self) -> None:
        """
        Cancel up to ``training_cores`` running simulations to free compute
        resources for training when no ResourceManager is available.

        The cancelled sims are NOT permanently killed — ``_on_sim_done`` will
        re-queue them via ``add_sims_to_queue`` so they resume after training.
        """
        n_to_cancel = getattr(self, "training_cores", 0)
        sims_to_cancel = list(self.registered_sims.keys())[:n_to_cancel]
        for sim_idx in sims_to_cancel:
            task = self.registered_sims.get(sim_idx)
            if task is not None and not task.done():
                self.logger.info(
                    f"Cancelling sim {sim_idx} to free resources for training",
                    component="simulation",
                )
                task.cancel()
        self.sim_batch_size -= n_to_cancel

    # --------------------------------------------------------------------------
    async def cancel_sims(self):
        """
        Cancel sims based on prediction score — permanent kill,
        counted as completed.
        """
        cancelled = False
        for sim_idx, pred in self.sim_predictions.items():
            if sim_idx not in self.registered_sims:
                continue

            if self.debug:
                self.logger.info(
                    f"Sim {sim_idx} prediction: {pred}", component="prediction"
                )

            if self.stop_simulation(prediction=pred):
                # Mark as permanently killed before cancelling so _on_sim_done
                # knows not to re-queue this sim.
                self._perm_cancelled.add(sim_idx)
                self.completed_sims.append(sim_idx)
                self.registered_sims[sim_idx].cancel()
                self.logger.task_killed(
                    f"Sim {sim_idx} permanently killed (prediction score {pred})",
                    component="simulation",
                )
                cancelled = True

    # --------------------------------------------------------------------------
    async def start(self):
        """
        Main event loop:
        - Collects input simulations
        - Submits and monitors tasks
        - Cancels based on predictions
        - Waits until all sims finish or queue empties
        """

        self.logger.separator("DDSim MANAGER STARTING")
        await self.init_sim_queue()
        submit_task = asyncio.create_task(self.submit_sims())

        while self.run_workflow:
            self.logger.info(f"{len(self.registered_sims)} simulation(s) running...")
            if self.debug:
                self.logger.info(f"{list(self.registered_sims.keys())}")

            # Train model if flag is set
            if self.retrain_model:
                # Skip waiting for training data if it is available at start
                if self.free_resources_for_train:
                    await self.monitor_training_data()  # blocks until training starts
                    if self._rm is None:
                        await self._free_resources_no_rm()
                else:
                    while True:
                        start_training = await self.check_train_status()
                        if start_training:
                            self.logger.info("Training can start now.")
                            break
                        else:
                            await asyncio.sleep(self.sleep_time)

                # Wait for a resource slot before submitting the request
                if self._rm is not None:
                    train_task = "train_task"
                    await self._wait_for_resource(train_task, "train_model")

                self.logger.task_started("Model Training", component="training")
                tasks = [self.train_model()]
                tasks.extend(t() for t in self.train_models)
                await asyncio.gather(*tasks)
                self.logger.task_completed("Model Training", component="training")

                if self._rm is not None:
                    self._rm.release(train_task)

            else:
                await asyncio.sleep(self.sleep_time)

            if self.call_evaluate_simulations:
                self.logger.task_started("Sim evaluation", component="evaluate")

                # Wait for a resource slot before submitting the request
                if self._rm is not None:
                    inf_task = "inf_task"
                    await self._wait_for_resource(inf_task, "inference")

                # Collect prediction scores for all simulations
                await self.evaluate_simulations()
                self.logger.task_completed("Sim evaluation", component="evaluate")
                if self._rm is not None:
                    self._rm.release(inf_task)

            if self.call_cancel_simulations:
                self.logger.task_started("Sim cancelation", component="cancelation")
                cancel_task = asyncio.create_task(self.cancel_sims())
                self.logger.task_completed("Sim cancelation", component="cancelation")
            else:
                cancel_task = None

            if self.call_finalize_results:
                # Wait for a resource slot before submitting the request
                if self._rm is not None:
                    fin_task = "finalize_task"
                    await self._wait_for_resource(fin_task, "finalize_results")

                self.logger.task_started("Finalize Results", component="finalization")
                finalize_results_task = asyncio.create_task(self.finalize_results())

                if finalize_results_task:
                    await finalize_results_task
                    if self._rm is not None:
                        self._rm.release(fin_task)
                self.logger.task_completed("Finalize Results", component="finalization")

            if cancel_task is not None:
                await cancel_task

            await asyncio.sleep(1)

        await submit_task
        self.logger.manager_exiting()
        self.logger.separator("DDSim MANAGER FINISHED")
