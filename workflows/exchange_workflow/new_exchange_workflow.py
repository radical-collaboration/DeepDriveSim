#!/usr/bin/env python3
"""
exchange_workflow.py
────────────────────
Wires simulation.py and exchange.py into the DDSimManager framework
with true MD/exchange concurrency.

Concurrency model
──────────────────
                ┌──────────────────────────────────────────────────────┐
  Replica 0    │ [==equil==][==prod 0==]·[==prod 1==]·[==prod 2==] ... │
  Replica 1    │ [==equil==][==prod 0==]·[==prod 1==]·[==prod 2==] ... │
  Replica 2    │ [==equil==][==prod 0==]·[==prod 1==]·[==prod 2==] ... │
  Replica 3    │ [==equil==][==prod 0==]·[==prod 1==]·[==prod 2==] ... │
  Exchange              (idle)          [EX0]·     [EX1]·
                └──────────────────────────────────────────────────────┘
               · = brief pause while replica waits on resume_event

Event handshake (per cycle, per replica)
──────────────────────────────────────────
  Replica                          Exchange loop
  ───────                          ─────────────
  saveCheckpoint()
  ready_events[rid].set()   ──►   await all ready_events
                                   run_exchange()   (coord swap + resave)
                                   ready_events[r].clear()   # BEFORE release
                                   resume_events[r].set()    # release all
  await resume_events[rid]  ◄──
  resume_events[rid].clear()
  loadCheckpoint()
  [next production window]

Why ready_events are cleared BEFORE resume_events are set
───────────────────────────────────────────────────────────
If the order were reversed (set resume → clear ready), a fast replica
could finish its NEXT production window and call ready_events[rid].set()
before the exchange loop reaches ready_events[r].clear(). The exchange
loop's subsequent await ready_events[r].wait() would then return
immediately (the event is already set from the next cycle), causing the
exchange to fire one cycle early with stale checkpoints.

Clearing first guarantees the flag is always clean before any replica
can possibly set it again.

Why exchange is a plain async method, not a @flow.function_task
────────────────────────────────────────────────────────────────
Calling a @flow.function_task returns a Future, not a coroutine.
asyncio.create_task() requires a coroutine and raises:
    "a coroutine was expected, got <Future pending>"
The exchange loop runs entirely in the manager process (it only does
asyncio event coordination and offloads the CPU-bound OpenMM swap to
run_in_executor), so it does not need the workflow engine at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

from ddsim.ddsim_manager import DDSimManager
from simulation import run_simulation
from exchange   import run_exchange
from replica_signals import ReplicaSignalSet


class ExchangeWorkflow(DDSimManager):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.flow = kwargs.get("asyncflow")
        if self.flow is None:
            raise ValueError("asyncflow engine is required")

        self.config      = kwargs.get("config")
        self.workflow_id = "exchange_workflow"

        self.task_types        = list(self.config["tasks"].keys())
        self.task_descriptions = self._generate_task_description(self.config)

        # One long-running task per replica; no resource competition with exchange.
        self.sim_batch_size = float("inf")
        self.max_sim_batch  = float("inf")

        self.retrain_model             = False
        self.call_post_process_sim     = False  # replicas never complete mid-run
        self.call_evaluate_simulations = False
        self.call_finalize_results     = True

        # ── Paths ──────────────────────────────────────────────────────────────
        input_path    = Path(self.config["input"])
        self.top_file = input_path / self.config.get("top_file", "ala_dipep.top")
        self.work_dir = Path(self.config["work_dir"])
        self.work_dir.mkdir(parents=True, exist_ok=True)

        gro_files         = self.config["gro_files"]
        self.replica_gro  : Dict[int, Path]  = {
            i: input_path / f for i, f in enumerate(gro_files)
        }
        self.replica_temps: Dict[int, float] = dict(
            enumerate(self.config["temperatures"])
        )

        # ── Parameters ─────────────────────────────────────────────────────────
        self.start_cycle = self.config.get("start_cycle",0)
        self.max_equil_steps     = self.config.get("max_equil_steps",     1_000_000)
        self.production_steps    = self.config.get("production_steps",    2_000)
        self.max_exchange_cycles = self.config.get("max_exchange_cycles", 50)

        # ── Synchronisation events (one per replica) ───────────────────────────
        # ready_events[rid]  — set by replica when window done + checkpoint saved
        # resume_events[rid] — set by exchange loop to release replica
        num_replicas = len(gro_files)
        self.signals = ReplicaSignalSet(self.work_dir, num_replicas)
        self.ready_events = self.signals.ready 
        self.resume_events = self.signals.resume
        #self.ready_events  : List[asyncio.Event] = [
        #    asyncio.Event() for _ in range(num_replicas)
        #]
        #self.resume_events : List[asyncio.Event] = [
        #    asyncio.Event() for _ in range(num_replicas)
        #]

        # ── Internal state ──────────────────────────────────────────────────────
        self.all_sims       : Dict[str, int]  = {}
        self.sim_inputs     : Dict[str, dict] = {}
        self._exchange_task : asyncio.Task | None = None

        self.register_tasks()

    # ──────────────────────────────────────────────────────────────────────────
    # Task registration
    # ──────────────────────────────────────────────────────────────────────────

    def register_tasks(self):
        """Register md_simulation as a @flow.function_task (tracked by DDSimManager).
        The exchange loop is a plain async method — see _run_exchange_loop."""

        task_desc_md   = self.task_descriptions["MD"]
        replica_gro    = self.replica_gro
        replica_temps  = self.replica_temps
        top_file       = self.top_file
        work_dir       = self.work_dir
        start_cycle    = self.start_cycle
        max_equil      = self.max_equil_steps
        prod_steps     = self.production_steps
        max_cycles     = self.max_exchange_cycles
        ready_events   = self.ready_events
        resume_events  = self.resume_events

        @self.flow.function_task
        async def md_simulation(task_description=task_desc_md, **kwargs):
            """
            One long-running task per replica.
            Submitted ONCE; loops internally over all exchange cycles.
            Only returns when max_exchange_cycles are complete.
            """
            sim_inputs = kwargs.get("sim_inputs", {})
            sim_idx    = sim_inputs.get("sim_idx", "MD_0")
            rid        = sim_inputs.get("rid", int(sim_idx.split("_")[-1]))

            return await run_simulation(
                rid                 = rid,
                sim_idx             = sim_idx,
                gro_file            = replica_gro[rid],
                top_file            = top_file,
                work_dir            = work_dir,
                target_temp         = replica_temps[rid],
                start_cycle         = start_cycle,
                max_exchange_cycles = max_cycles,
                ready_events        = ready_events,
                resume_events       = resume_events,
                max_equil_steps     = max_equil,
                production_steps    = prod_steps,
            )

        self.simulation = md_simulation

    # ──────────────────────────────────────────────────────────────────────────
    # Exchange loop  —  plain async method, NOT a @flow.function_task
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_exchange_loop(self):
        """
        Runs concurrently with replica tasks via asyncio.create_task().

        For each cycle:
          1. Await all ready_events  — every replica has saved its checkpoint.
          2. Run run_exchange()      — CPU-bound, pushed to thread executor.
          3. Clear ready_events      — BEFORE setting resume_events (see docstring).
          4. Set resume_events       — release all replicas for next window.
        """
        ex_list = list(self.replica_temps.keys())
        gro_ref = self.replica_gro[ex_list[0]]
        loop    = asyncio.get_event_loop()

        for cycle in range(self.start_cycle,self.max_exchange_cycles):
            self.logger.info(
                f"Cycle {cycle} — waiting for all replicas",
                component="exchange",
            )

            # Wait until every replica has finished its production window
            await asyncio.gather(*[self.ready_events[r].wait(cycle) for r in ex_list])

            self.logger.info(
                f"Cycle {cycle} — all replicas ready, running swap",
                component="exchange",
            )
            
            # ADD DEBUG: print before executor call
            print(f"[DEBUG] Starting run_exchange for cycle {cycle}", flush=True)
            
            # Push CPU-bound OpenMM work to a thread so the event loop
            # stays responsive (resume_events.wait() must remain awaitable)
            try:
                await loop.run_in_executor(
                    None,
                    lambda c=cycle: run_exchange(
                        ex_list  = ex_list,
                        cycle    = c,
                        work_dir = self.work_dir,
                        top_file = self.top_file,
                        gro_file = gro_ref,
                        verbose  = self.debug,
                    )
                )
            except Exception as e:
                print(f"[Error] run_exchange failed: {e}", flush=True)
                raise
            
            # ADD DEBUG: print after executor call
            print(f"[DEBUG] run_exchange completed for cycle {cycle}", flush=True)            
            # ── Clear ready_events BEFORE setting resume_events ────────────────
            # A fast replica could finish its next window and call
            # ready_events[rid].set() before we clear it here.
            # If we set resume first, that race makes the NEXT cycle's
            # await ready_events[r].wait() return immediately with a stale flag.
            for r in ex_list:
                self.ready_events[r].clear(cycle)

            # Release all replicas — they will loadCheckpoint() and continue
            for r in ex_list:
                self.resume_events[r].set(cycle)

            self.logger.info(
                f"Cycle {cycle} complete — replicas released",
                component="exchange",
            )

        self.logger.info(
            f"All {self.max_exchange_cycles} exchange cycles complete",
            component="exchange",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Queue management
    # ──────────────────────────────────────────────────────────────────────────

    async def init_sim_queue(self):
        """Queue ONE long-running MD task per replica (submitted once, never re-queued)."""
        for rid in range(len(self.replica_gro)):
            n       = self.all_sims.get("MD", 0)
            sim_idx = f"MD_{n}"
            self.all_sims["MD"] = n + 1
            inputs  = {"sim_idx": sim_idx, "rid": rid}
            await self.sim_task_queue.put(inputs)
            self.sim_inputs[sim_idx] = inputs
            self.logger.info(
                f"Queued long-running task {sim_idx} for replica {rid}",
                component="queue",
            )

    async def add_sims_to_queue(self, sim_ids: List[str]):
        """Re-queue sims cancelled to free resources (should not occur normally)."""
        for sim_idx in sim_ids:
            inputs  = self.sim_inputs.get(sim_idx, {})
            rid     = inputs.get("rid", int(sim_idx.split("_")[-1]))
            n       = self.all_sims.get("MD", 0)
            new_idx = f"MD_{n}"
            self.all_sims["MD"] = n + 1
            new_inputs = {"sim_idx": new_idx, "rid": rid}
            await self.sim_task_queue.put(new_inputs)
            self.sim_inputs[new_idx] = new_inputs

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def stop_simulation(self, *args, **kwargs) -> bool:
        return False

    async def start(self):
        """
        Override DDSimManager.start():
          1. Seed the task queue.
          2. Launch submit_sims() to dispatch replica tasks via asyncflow.
          3. Launch _run_exchange_loop() as a plain asyncio.create_task().
          4. Poll finalize_results() every second until done.
        """
        self.logger.separator("DDSim MANAGER STARTING")
        await self.init_sim_queue()

        submit_task = asyncio.create_task(self.submit_sims())

        # _run_exchange_loop() returns a coroutine — safe for create_task.
        self._exchange_task = asyncio.create_task(self._run_exchange_loop())
        self.logger.info(
            "Exchange loop launched concurrently with MD replicas",
            component="exchange",
        )

        try:
            while self.run_workflow:
                if self.debug:
                    self.logger.info(
                        f"{len(self.registered_sims)} replica(s) running",
                        component=self.name,
                    )
                if self.call_finalize_results:
                    await self.finalize_results()
                await asyncio.sleep(1)

        finally:
            self.shutting_down.set()
            submit_task.cancel()
            await asyncio.gather(submit_task, return_exceptions=True)

            if self._exchange_task and not self._exchange_task.done():
                self._exchange_task.cancel()
                await asyncio.gather(self._exchange_task, return_exceptions=True)

            if self.registered_sims:
                for task in list(self.registered_sims.values()):
                    task.cancel()
                await asyncio.gather(
                    *self.registered_sims.values(), return_exceptions=True
                )

        self.logger.separator("DDSim MANAGER FINISHED")

    async def finalize_results(self):
        """
        Shut down once the exchange loop AND all replica tasks are complete.
        Both conditions are required to avoid shutting down while the last
        replica is still writing its final checkpoint.
        """
        exchange_done = (
            self._exchange_task is not None
            and self._exchange_task.done()
        )
        replicas_done = len(self.registered_sims) == 0

        if exchange_done and replicas_done:
            self.logger.info(
                "All replicas and exchange loop complete — shutting down.",
                component="finalize",
            )
            self.run_workflow = False

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_task_description(self, config: dict) -> dict:
        task_description = {}
        for t in self.task_types:
            cfg = config["tasks"][t]
            task_description[t] = {
                "ranks"         : 1,
                "cores_per_rank": cfg.get("cpu_reqs", 1),
                "gpus_per_rank" : cfg.get("gpu_reqs", 0),
                "pre_exec"      : cfg.get("pre_exec", []),
                "shell"         : True,
            }
        return task_description

    async def close(self):
        await self.flow.shutdown()
