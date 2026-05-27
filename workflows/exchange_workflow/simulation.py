#!/usr/bin/env python3
"""
simulation.py
─────────────
Long-running replica task for the Replica-Exchange workflow.

Each replica runs an internal loop for ALL exchange cycles without
ever exiting. At the end of every production window it:
  1. Saves a checkpoint to disk.
  2. Writes the state JSON (potential energy, temperature).
  3. Sets ready_events[rid]   — signals the exchange task.
  4. Waits on resume_events[rid] — blocks until exchange is done.
  5. Loads the (possibly swapped) checkpoint and continues.

Velocity policy
────────────────
setVelocitiesToTemperature is called ONCE at cycle 0 initialisation.
It is never called again — the Langevin thermostat handles kinetic
equilibration after every coordinate swap naturally.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import List

from replica_signals import _FileSignal

from openmm import CMMotionRemover, LangevinMiddleIntegrator
from openmm.app import (
    DCDReporter,
    GromacsGroFile,
    GromacsTopFile,
    HBonds,
    PME,
    Simulation,
    StateDataReporter,
)
from openmm.unit import kelvin, kilojoules_per_mole, nanometer, picosecond, picoseconds

KB_KJ = 0.008_314_462_175
GROMACS_TOP_INCLUDE = (
    "/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/"
    "gromacs-2025.2-64mhcw3/share/gromacs/top"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_dof(system) -> int:
    """Degrees of freedom: 3N − constraints − 3 (CMMotionRemover)."""
    n_c = system.getNumConstraints()
    n_p = system.getNumParticles()
    has_cmm = any(
        isinstance(system.getForce(i), CMMotionRemover)
        for i in range(system.getNumForces())
    )
    return 3 * n_p - n_c - (3 if has_cmm else 0)


def _instant_temp(simulation, n_dof: int) -> float:
    """Instantaneous temperature (K) from current kinetic energy."""
    state = simulation.context.getState(getEnergy=True)
    ke    = state.getKineticEnergy().value_in_unit(kilojoules_per_mole)
    return (2.0 * ke) / (n_dof * KB_KJ)


def _checkpoint_path(work_dir: Path, rid: int, cycle: int) -> Path:
    return work_dir / ("checkpoint_%04d.%04d" % (rid, cycle))


def _state_json_path(work_dir: Path, rid: int, cycle: int) -> Path:
    return work_dir / ("state_%04d.%04d.json" % (rid, cycle))


def _build_simulation(top_file, gro_file, target_temp, top_include_dir):
    gro = GromacsGroFile(str(gro_file))
    top = GromacsTopFile(
        str(top_file),
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=top_include_dir,
    )
    system = top.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * nanometer,
        constraints=HBonds,
    )
    integrator = LangevinMiddleIntegrator(
        target_temp * kelvin, 1 / picosecond, 0.002 * picoseconds
    )
    sim = Simulation(top.topology, system, integrator)
    return sim, system, gro


def _attach_reporters(simulation, work_dir, rid, cycle, dcd_interval, stat_interval):
    dcd_path  = work_dir / ("output_%04d.%04d.dcd" % (rid, cycle))
    stat_path = work_dir / ("state_%04d.%04d.txt"  % (rid, cycle))
    simulation.reporters.clear()
    simulation.reporters.append(DCDReporter(str(dcd_path), dcd_interval))
    simulation.reporters.append(
        StateDataReporter(
            str(stat_path), stat_interval,
            step=True, potentialEnergy=True, temperature=True,
        )
    )
    return stat_path


# ── Main long-running entry point ─────────────────────────────────────────────

async def run_simulation(
    rid: int,
    sim_idx: str,
    gro_file: Path,
    top_file: Path,
    work_dir: Path,
    target_temp: float,
    max_exchange_cycles: int,
    ready_events: List[_FileSignal],
    resume_events: List[_FileSignal],
    *,
    temp_tolerance: float    = 5.0,
    check_interval: int      = 500,
    window: int              = 50,
    max_equil_steps: int     = 1_000_000,
    production_steps: int    = 2_000,
    dcd_report_interval: int  = 100,
    stat_report_interval: int = 100,
    top_include_dir: str     = GROMACS_TOP_INCLUDE,
) -> dict:
    """
    Run one replica for all exchange cycles without exiting.

    Cycle 0  : initialise from .gro, minimise, draw velocities, equilibrate,
               then run first production window.
    Cycle N  : reload checkpoint (may carry swapped coords from exchange),
               run production window, signal, wait, repeat.

    The event handshake after every window:
        ready_events[rid].set()       — "my checkpoint is saved, exchange can proceed"
        await resume_events[rid].wait() — "waiting for exchange to finish"
        resume_events[rid].clear()    — reset for next cycle
        simulation.loadCheckpoint()   — pick up any coord swap exchange did
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()

    # Build OpenMM objects once — reused across all cycles
    simulation, system, gro = _build_simulation(
        top_file, gro_file, target_temp, top_include_dir
    )
    n_dof = _get_dof(system)

    # ── Cycle 0: initialise + equilibrate ─────────────────────────────────────
    simulation.context.setPositions(gro.positions)
    simulation.minimizeEnergy()
    # ONLY call to setVelocitiesToTemperature in the entire workflow.
    simulation.context.setVelocitiesToTemperature(target_temp * kelvin)
    print(f"[sim {rid}] initialised from {Path(gro_file).name}", flush=True)

    _attach_reporters(
        simulation, work_dir, rid, 0,
        dcd_report_interval, stat_report_interval
    )

    # Equilibration loop — run in executor so event loop stays live
    temp_history = deque(maxlen=window)
    total_equil  = 0

    while total_equil < max_equil_steps:
        await loop.run_in_executor(None, simulation.step, check_interval)
        total_equil += check_interval

        instant_t = _instant_temp(simulation, n_dof)
        temp_history.append(instant_t)

        if len(temp_history) < window:
            continue

        rolling_mean = sum(temp_history) / window
        if abs(rolling_mean - target_temp) <= temp_tolerance:
            print(
                f"[sim {rid}] equilibrated at step {total_equil} "
                f"(rolling mean {rolling_mean:.1f} K)",
                flush=True,
            )
            break
    else:
        print(f"[sim {rid}] WARNING: equilibration not achieved", flush=True)

    # ── Production loop across all exchange cycles ─────────────────────────────
    pot     = 0.0
    final_T = target_temp

    for cycle in range(max_exchange_cycles):

        # Fresh reporters for this cycle's output files
        _attach_reporters(
            simulation, work_dir, rid, cycle,
            dcd_report_interval, stat_report_interval
        )

        # Run production window in executor — keeps event loop responsive
        # so resume_events.wait() can resolve while other replicas are stepping
        await loop.run_in_executor(None, simulation.step, production_steps)

        # Collect state
        state = simulation.context.getState(
            getPositions=True, getVelocities=True,
            getEnergy=True, enforcePeriodicBox=True,
        )
        pot     = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        final_T = _instant_temp(simulation, n_dof)

        # Save checkpoint and JSON sidecar BEFORE signalling exchange
        ckpt_path = _checkpoint_path(work_dir, rid, cycle)
        simulation.saveCheckpoint(str(ckpt_path))

        with open(_state_json_path(work_dir, rid, cycle), "w") as fh:
            json.dump({
                "sim_idx"         : sim_idx,
                "rid"             : rid,
                "cycle"           : cycle,
                "steps_completed" : production_steps,
                "final_temp"      : final_T,
                "potential_energy": pot,
                "target_temp"     : target_temp,
                "checkpoint_path" : str(ckpt_path),
            }, fh, indent=4)

        print(
            f"[sim {rid}] cycle {cycle} done — "
            f"pot={pot:.2f} kJ/mol, T={final_T:.1f} K — signalling exchange",
            flush=True,
        )

        # ── Handshake with exchange task ───────────────────────────────────────
        #
        # Step 1 — signal: checkpoint is on disk, exchange can read/swap it.
        ready_events[rid].set(cycle)
        #
        # Step 2 — wait: block until exchange has finished the swap and
        #          re-saved all checkpoints (including ours).
        await resume_events[rid].wait(cycle)
        #
        # Step 3 — clear() is optional with file signals (each cycle has a
        #          unique file), but kept for filesystem hygiene.
        resume_events[rid].clear(cycle)
        #
        # Step 4 — reload: pick up whatever exchange wrote to our checkpoint.
        #   • If we were swapped   → checkpoint has foreign coordinates.
        #   • If we were not swapped → checkpoint is identical to what we saved.
        #   Either way the reload is safe and cheap.
        simulation.loadCheckpoint(str(ckpt_path))
        print(f"[sim {rid}] cycle {cycle} — resumed after exchange", flush=True)

    print(f"[sim {rid}] all {max_exchange_cycles} cycles complete.", flush=True)
    return {
        "sim_idx"          : sim_idx,
        "rid"              : rid,
        "cycles_completed" : max_exchange_cycles,
        "final_temp"       : final_T,
        "potential_energy" : pot,
    }
