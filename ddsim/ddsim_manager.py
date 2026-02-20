#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Async-friendly DDMD manager for orchestrating simulations
# ------------------------------------------------------------------------------

import asyncio
import subprocess
from collections import OrderedDict

from ddsim.logger import Logger


def gpu_available():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


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
        - run_inference (collect prediction scores)
        - close
    Optional:
        - free_resources_for_train (default False)
        - run_post_process (default False) + post_process method
    """

    def __init__(self):
        self.logger = Logger(use_colors=True)
        self.registered_sims = OrderedDict()  # Active simulations: {tag: asyncio.Task}
        self.sim_task_queue = asyncio.Queue()  # Queue of pending simulation inputs
        self.completed_sims = []  # Completed simulations
        self.sim_predictions = {}
        self.train_models = []  # Additional training coroutines to run in parallel

        self.sleep_time = 20  # Delay between prediction/start train checks
        self.debug = False

        self.run_pipeline = True
        self.free_resources_for_train = False
        self.run_post_process = False

        # Event should be set inside pipeline code to stop simulation loop
        self.shutting_down = asyncio.Event()
        self.logger.info("DDSim Manager initialized...")

    # --------------------------------------------------------------------------
    def add_sims_to_queue(self, *args, **kwargs):
        """
        Specify how to add sim back to queue after it was canceled to free resources.
        """
        raise NotImplementedError("add_sims_to_queue must be implemented")

    # --------------------------------------------------------------------------
    def post_process_sim(self, *args, **kwargs):
        """
        Specify any post pprecess required after each sim completed
        """
        raise NotImplementedError("post_process_sim must be implemented")

    # --------------------------------------------------------------------------
    def post_process(self, *args, **kwargs):
        """
        Specify any post pprecess required after each Inference iteration
        """
        raise NotImplementedError("post_process must be implemented")

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
        Override this with actual logic in pipeline subclass.
        """
        raise NotImplementedError("init_sim_queue must be implemented")

    # --------------------------------------------------------------------------
    async def check_train_status(self):
        """
        Check if enough training data is available to start training.
        Override this with actual logic in pipeline subclass.
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
    async def run_inference(self):
        """Collect prediction scores for all running simulations.
        Override this with actual logic in pipeline subclass.
        """
        pass

    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner.
        Override this with actual logic in pipeline subclass.
        """
        raise NotImplementedError("close must be implemented")

    # --------------------------------------------------------------------------
    async def stop(self):
        """Alias for close(), can be used for external termination."""
        await self.close()

    # --------------------------------------------------------------------------
    async def _unregister_sims(self, unregistered_sims):
        """Remove completed or canceled simulations from the registry."""
        for sim_idx in unregistered_sims:
            self.registered_sims.pop(sim_idx, None)
            self.completed_sims.append(sim_idx)

        if unregistered_sims and not self.sim_task_queue.empty():
            # Adjust next batch size (ensure it does not exceed max_sim_batch)
            num_to_submit = min(self.sim_batch_size, self.sim_task_queue.qsize())
            if num_to_submit > 0:
                self.logger.info(
                    f"{num_to_submit} simulations will start at next iteration"
                )

    # --------------------------------------------------------------------------
    async def submit_sims(self):
        """Submit simulations from the queue and register them."""
        while not self.shutting_down.is_set():
            await self.monitor_sims()  # Clean up completed/failed tasks

            if self.sim_batch_size <= 0:
                await asyncio.sleep(1)
                continue

            # Don't submit more than sim_batch_size simulation
            # to have enough resources for training task
            num_to_submit = min(self.sim_batch_size, self.sim_task_queue.qsize())
            for _ in range(num_to_submit):
                try:
                    sim_inputs = self.sim_task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    self.logger.info("No more simulation inputs in queue.")
                    break

                if sim_inputs is None:
                    break
                simul = self.simulation(sim_inputs=sim_inputs)
                sim_idx = sim_inputs["sim_idx"]
                self.logger.task_started(f"Sim {sim_idx}", component="simulation")
                self.registered_sims[sim_idx] = simul

            if num_to_submit > 0:
                self.logger.info(f"Submitted {num_to_submit} new simulation(s)")

            # Update sim_batch_size (subtract submitted items)
            self.sim_batch_size -= num_to_submit
            await asyncio.sleep(0.1)

    # --------------------------------------------------------------------------
    async def monitor_sims(self):
        """Unregister completed/failed simulations and prepare next batch."""
        unregistered_sims = []
        for sim_idx, task in self.registered_sims.items():
            if task.done():
                unregistered_sims.append(sim_idx)
                self.sim_batch_size += 1

                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    self.logger.info(
                        f"Sim {sim_idx} was cancelled", component="simulation"
                    )
                    continue

                if exc:
                    self.logger.error(
                        f"Sim {sim_idx} failed: {exc}",
                        component="simulation",
                    )
                else:
                    self.logger.task_completed(
                        f"Sim {sim_idx}", component="simulation"
                    )
                    await self.post_process_sim(sim_idx)
        await self._unregister_sims(unregistered_sims)

    # --------------------------------------------------------------------------
    async def monitor_training_data(self):
        """
        Cancel sims when enough training data is available and
        free resources for model training.
        """
        while True:
            start_training = await self.check_train_status()

            if start_training:
                unregistered_sims = []
                resubmitted_sims = []
                count = 0

                # Suspend simulations to free up resources for training
                for sim_idx, task in list(self.registered_sims.items()):
                    if task.done():
                        unregistered_sims.append(sim_idx)
                    else:
                        try:
                            task.cancel()
                            unregistered_sims.append(sim_idx)
                            self.logger.task_killed(
                                f"Cancelling Sim {sim_idx} to free training resources",
                                #  f"(ROSE task ID {getattr(task, 'id', 'N/A')})"
                                component="simulation",
                            )
                            resubmitted_sims.append(sim_idx)
                        except Exception as e:
                            self.logger.error(f"Error cancelling Sim {sim_idx}: {e}")
                            continue

                    count += 1
                    if count >= self.training_cores:
                        break

                self.logger.info(
                    f"Cancelled {count} simulations; Training will now start."
                )

                # Remove canceled sims from registry
                await self._unregister_sims(unregistered_sims)

                # Re-add canceled sims back to task queue for later rescheduling
                await self.add_sims_to_queue(resubmitted_sims)
                # for sim_idx in resubmitted_sims:
                #     await self.sim_task_queue.put({"sim_idx": sim_idx})
                #     self.logger.info(f"Re-added Sim {sim_idx} back the queue")

                break  # Exit loop after canceling
            else:
                await asyncio.sleep(self.sleep_time)

    # --------------------------------------------------------------------------
    async def cancel_sims(self):
        """Cancel sims based on prediction score."""
        unregister_sims = []

        for sim_idx, pred in self.sim_predictions.items():
            if sim_idx not in self.registered_sims.keys():
                continue

            if self.debug:
                self.logger.info(
                    f"Sim {sim_idx} prediction: {pred}", component="prediction"
                )

            if self.stop_simulation(prediction=pred):
                task = self.registered_sims[sim_idx]
                task.cancel()
                unregister_sims.append(sim_idx)
                self.logger.task_killed(
                    f"Sim {sim_idx} canceled due to prediction score {pred} ",
                    component="simulation",
                    # f"(task ID {getattr(task, 'id', 'N/A')})"
                )
                self.sim_batch_size += 1

        await self._unregister_sims(unregister_sims)

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

        while self.run_pipeline:
            self.logger.info(f"{len(self.registered_sims)} simulation(s) running...")
            if self.debug:
                self.logger.info(f"{list(self.registered_sims.keys())}")

            # Train model if flag is set
            if self.retrain_model:
                # Skip waiting for training data if it is available at start
                if self.free_resources_for_train:
                    await self.monitor_training_data()  # blocks until training starts
                else:
                    while True:
                        start_training = await self.check_train_status()
                        if start_training:
                            self.logger.info("Training can start now.")
                            break
                        else:
                            await asyncio.sleep(self.sleep_time)
                tasks = [self.train_model()]
                tasks.extend(t() for t in self.train_models)
                await asyncio.gather(*tasks)
            else:
                await asyncio.sleep(self.sleep_time)

            self.logger.task_started("Model Inference", component="inference")
            # Collect prediction scores for all simulations
            await self.run_inference()
            self.logger.task_completed("Model Inference", component="inference")

            cancel_task = asyncio.create_task(self.cancel_sims())

            post_process_task = None
            if self.run_post_process:
                post_process_task = asyncio.create_task(self.post_process())

            if self.sim_task_queue.empty():
                await self.monitor_sims()

            if post_process_task:
                await post_process_task
            await cancel_task

            await asyncio.sleep(1)

        await submit_task
        self.logger.manager_exiting()
        self.logger.separator("DDMD MANAGER FINISHED")
