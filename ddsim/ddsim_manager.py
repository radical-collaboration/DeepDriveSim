#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Async-friendly DDSim manager for orchestrating simulations
# ------------------------------------------------------------------------------

import asyncio
from collections import OrderedDict

from ddsim.logger import Logger


class DDSimManager:
    """
    Orchestrates the scheduling, monitoring, and cancellation of simulations
    in an AI-steered ensemble simulation workflow.

    Sim lifecycle
    -------------
    submitted  → registered_sims  (task is running)
    completed  → completed_sims   (task finished successfully)
    cancelled for resources       → re-queued via add_sims_to_queue
    cancelled by prediction       → counted as completed (cancel_sims pre-adds
                                    to completed_sims before cancelling the task)

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

    def __init__(self, name: str = "ddsim"):
        self.name = name
        self.workflow_id = name
        self.logger = Logger(name=name, use_colors=True)
        self.registered_sims = OrderedDict()  # running:  {sim_idx: asyncio.Task}
        self.sim_task_queue = asyncio.Queue()  # pending sim inputs
        self.completed_sims = []  # finished sim indices
        self.sim_predictions = {}
        self.train_models = []

        self.sleep_time = 20
        self.debug = False

        # Workflow flags — set in subclass:
        self.call_finalize_results = False
        self.call_evaluate_simulations = False
        self.run_workflow = True
        self.free_resources_for_train = False
        self.call_cancel_simulations = False
        self.call_post_process_sim = False

        self.sim_batch_size = 0
        self.max_sim_batch = 0

        # Sims permanently killed by cancel_sims (prediction-based).
        # _on_sim_done uses this to distinguish a permanent kill (no re-queue)
        # from a resource-freeing cancel (re-queue).
        self._perm_cancelled: set = set()

        self.shutting_down = asyncio.Event()
        self.logger.separator(f"{self.name} MANAGER STARTING")

    # --------------------------------------------------------------------------
    def add_sims_to_queue(self, *args, **kwargs):
        raise NotImplementedError("add_sims_to_queue must be implemented")

    def post_process_sim(self, *args, **kwargs):
        raise NotImplementedError("post_process_sim must be implemented")

    def stop_simulation(self, *args, **kwargs):
        raise NotImplementedError("stop_simulation must be implemented")

    async def init_sim_queue(self):
        raise NotImplementedError("init_sim_queue must be implemented")

    async def check_train_status(self):
        raise NotImplementedError("check_train_status must be implemented")

    async def train_model(self):
        """Primary training step. Override in subclass (default: no-op)."""
        pass

    async def close(self):
        raise NotImplementedError("close must be implemented")

    async def stop(self):
        await self.close()

    # --------------------------------------------------------------------------
    def _on_sim_done(self, task: asyncio.Task, sim_idx) -> None:
        """
        Done-callback registered on every simulation task.

        Unregisters the sim and frees its batch slot, then:
          - task.cancelled() + in _perm_cancelled  → prediction kill; already
            counted in completed_sims, just clean up.
          - task.cancelled() + not in _perm_cancelled → resource-freeing cancel;
            re-queue via add_sims_to_queue so the sim runs again after training.
          - task completed normally → add to completed_sims, call post_process_sim.
        """
        self.registered_sims.pop(sim_idx, None)
        self.sim_batch_size += 1
        if self.max_sim_batch > 0:
            self.sim_batch_size = min(self.sim_batch_size, self.max_sim_batch)

        if task.cancelled():
            if sim_idx in self._perm_cancelled:
                self._perm_cancelled.discard(sim_idx)
            else:
                self.logger.info(
                    f"Sim {sim_idx} cancelled for resources — re-queuing",
                    component=f"{self.name}-sim",
                )
                asyncio.ensure_future(self.add_sims_to_queue([sim_idx]))
        else:
            exc = task.exception()
            if exc:
                self.logger.error(
                    f"Sim {sim_idx} failed: {exc}", component=f"{self.name}-sim"
                )
            else:
                self.completed_sims.append(sim_idx)
                if self.debug:
                    self.logger.task_completed(
                        f"Sim {sim_idx}", component=f"{self.name}-sim"
                    )
                if self.call_post_process_sim:
                    asyncio.ensure_future(self.post_process_sim(sim_idx))

    # --------------------------------------------------------------------------
    async def submit_sims(self):
        """Pull inputs from sim_task_queue and register running tasks."""
        while not self.shutting_down.is_set():
            if self.sim_batch_size <= 0:
                await asyncio.sleep(1)
                continue

            num_to_submit = min(self.sim_batch_size, self.sim_task_queue.qsize())
            submitted = 0
            for _ in range(num_to_submit):
                try:
                    sim_inputs = self.sim_task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if sim_inputs is None:
                    break

                sim_idx = sim_inputs["sim_idx"]
                simul = self.simulation(sim_inputs=sim_inputs)
                simul.add_done_callback(
                    lambda t, sid=sim_idx: self._on_sim_done(t, sid)
                )

                if self.debug:
                    self.logger.task_started(
                        f"Sim {sim_idx}", component=f"{self.name}-sim"
                    )
                self.registered_sims[sim_idx] = simul
                submitted += 1

            if submitted > 0:
                if self.debug:
                    self.logger.info(
                        f"Submitted {submitted} new simulation(s)", component=self.name
                    )
                self.sim_batch_size -= submitted

            await asyncio.sleep(0.1)

    # --------------------------------------------------------------------------
    async def _free_resources_for_training(self) -> None:
        """
        Cancel up to training_cores running sims to free resources for training.
        Cancelled sims are re-queued by _on_sim_done and run again afterwards.
        """
        n_to_cancel = getattr(self, "training_cores", 0)
        sims_to_cancel = list(self.registered_sims.keys())[:n_to_cancel]
        for sim_idx in sims_to_cancel:
            task = self.registered_sims.get(sim_idx)
            if task is not None and not task.done():
                self.logger.info(
                    f"Cancelling sim {sim_idx} to free resources for training",
                    component=f"{self.name}-sim",
                )
                task.cancel()
        self.sim_batch_size -= n_to_cancel
        self.free_resources_for_train = False

    # --------------------------------------------------------------------------
    async def monitor_training_data(self):
        """Poll until training data is ready."""
        while not await self.check_train_status():
            await asyncio.sleep(self.sleep_time)
        self.logger.info("Training data ready.", component=self.name)

    # --------------------------------------------------------------------------
    async def cancel_sims(self):
        """
        Permanently cancel sims whose prediction score is below threshold.
        Pre-adds to completed_sims so _on_sim_done skips re-queuing.
        """
        to_await = []
        for sim_idx, pred in self.sim_predictions.items():
            if self.debug:
                self.logger.info(
                    f"Sim {sim_idx} prediction: {pred}",
                    component=f"{self.name}-predict",
                )
            if sim_idx in self.registered_sims and self.stop_simulation(
                prediction=pred
            ):
                self._perm_cancelled.add(sim_idx)
                self.completed_sims.append(sim_idx)
                task = self.registered_sims[sim_idx]
                task.cancel()
                to_await.append(task)
                self.logger.task_killed(
                    f"Sim {sim_idx} permanently killed (prediction score {pred})",
                    component=f"{self.name}-sim",
                )

        # Await all cancelled tasks so their done-callbacks (_on_sim_done pops
        # from registered_sims) fire before the caller proceeds to
        # finalize_results.  Without this yield, finalize_results sees
        # registered_sims non-empty even though all sims are already counted
        # in completed_sims, causing a spurious warning and a missed shutdown.
        if to_await:
            await asyncio.gather(*to_await, return_exceptions=True)

    # --------------------------------------------------------------------------
    async def start(self):
        """
        Main loop: submit sims, train, evaluate, cancel, finalize — until all done.
        """

        flow = self.flow
        workflow = self

        @flow.block
        async def run_workflow():
            await workflow.init_sim_queue()
            submit_task = asyncio.create_task(workflow.submit_sims())

            wm = None
            try:
                while workflow.run_workflow:
                    if workflow.debug:
                        workflow.logger.info(
                            f"{len(workflow.registered_sims)} simulation(s) running: "
                            f"{list(workflow.registered_sims.keys())}",
                            component=workflow.name,
                        )
                    if workflow.retrain_model:
                        wm = workflow.monitor_training_data()

                        if workflow.free_resources_for_train:
                            await workflow._free_resources_for_training()
                    if wm:
                        await wm
                        if workflow.debug:
                            workflow.logger.task_started(
                                "Model Training", component=f"{workflow.name}-train"
                            )
                        await asyncio.gather(
                            workflow.train_model(),
                            *(t() for t in workflow.train_models),
                        )
                        if workflow.debug:
                            workflow.logger.task_completed(
                                "Model Training", component=f"{workflow.name}-train"
                            )
                    else:
                        await asyncio.sleep(workflow.sleep_time)

                    if workflow.call_evaluate_simulations:
                        if workflow.debug:
                            workflow.logger.task_started(
                                "Sim evaluation", component=f"{workflow.name}-eval"
                            )
                        await workflow.evaluate_simulations()
                        if workflow.debug:
                            workflow.logger.task_completed(
                                "Sim evaluation", component=f"{workflow.name}-eval"
                            )

                    if workflow.call_cancel_simulations:
                        if workflow.debug:
                            workflow.logger.task_started(
                                "Sim cancelation", component=f"{workflow.name}-cancel"
                            )
                        await workflow.cancel_sims()
                        if workflow.debug:
                            workflow.logger.task_completed(
                                "Sim cancelation", component=f"{workflow.name}-cancel"
                            )

                    if workflow.call_finalize_results:
                        if workflow.debug:
                            workflow.logger.task_started(
                                "Finalize Results",
                                component=f"{workflow.name}-finalize",
                            )
                        await workflow.finalize_results()
                        if workflow.debug:
                            workflow.logger.task_completed(
                                "Finalize Results",
                                component=f"{workflow.name}-finalize",
                            )

                    await asyncio.sleep(1)

            finally:
                if not workflow.shutting_down.is_set():
                    workflow.shutting_down.set()

                workflow.logger.task_completed(
                    "Finalize Results - canceling sims",
                    component=f"{workflow.name}-finalize",
                )
                submit_task.cancel()
                await asyncio.gather(submit_task, return_exceptions=True)

                if workflow.registered_sims:
                    for task in list(workflow.registered_sims.values()):
                        task.cancel()
                    await asyncio.gather(
                        *workflow.registered_sims.values(), return_exceptions=True
                    )

            workflow.logger.manager_exiting()
            workflow.logger.separator(f"{workflow.name} MANAGER FINISHED")

        await run_workflow()
        await self.close()
