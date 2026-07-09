#!/usr/bin/env python3
"""
exchange.py
───────────
Replica-exchange coordinate swap task for the ExchangeWorkflow.

Called by ExchangeWorkflow._run_exchange_loop() once every active
replica has set its ready_event (i.e. finished its production window
and saved its checkpoint).

Algorithm  (Gibbs / pairwise-independence sampling)
─────────────────────────────────────────────────────
For each replica r_i, consider swapping with its nearest-temperature
neighbour r_j. Compute the acceptance probability from the Metropolis
criterion in reduced potentials:

    Δu = u(r_i,T_j) + u(r_j,T_i) − u(r_i,T_i) − u(r_j,T_j)
    p  = min(1, exp(−Δu))    where u(r,T) = E(r) / (k_B·T)

Pairs are selected greedily so no replica is in two swaps per round.
Accepted swaps load both checkpoints, swap the coordinate sets, and
re-save ALL checkpoints (swapped and non-swapped alike) so that every
replica's next loadCheckpoint() call is always valid.

Velocity policy
────────────────
Velocities are NEVER re-drawn at exchange time. Each replica's momenta
stay with their thermostat; the Langevin integrator equilibrates kinetics
to the new coordinate set naturally over the next production window.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import (
    PME,
    GromacsGroFile,
    GromacsTopFile,
    HBonds,
    Simulation,
)
from openmm.unit import kelvin, nanometer, picosecond, picoseconds

KB_KJ = 0.008_314_462_175
GROMACS_TOP_INCLUDE = (
    "/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gromacs-2025.2-64mhcw3/share/gromacs/top"
)


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _checkpoint_path(work_dir: Path, rid: int) -> Path:
    return work_dir / f"checkpoint_{rid:04d}.chk"


def _state_json_path(work_dir: Path, rid: int) -> Path:
    return work_dir / f"state_{rid:04d}.jsonl"


def _load_replica_states(
    ex_list: list[int], cycle: int, work_dir: Path
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Read potential energies and thermostat temperatures from JSON sidecars
    written by simulation.py after each production window.
    Returns (poten, temp) dicts keyed by rid.
    """
    poten: dict[int, float] = {}
    temp: dict[int, float] = {}
    for rid in ex_list:
        path = _state_json_path(work_dir, rid)
        if not path.exists():
            raise FileNotFoundError(
                f"State JSON not found for replica {rid} cycle {cycle}: {path}"
            )
        # Read the last non-empty line -most recent cycle
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last_line = line
        if last_line is None:
            raise ValueError(f"Empty state file for replica {rid}: {path}")

        data = json.loads(last_line)
        poten[rid] = data["potential_energy"]
        temp[rid] = data["target_temp"]
    return poten, temp


# ── Reduced-potential matrix ──────────────────────────────────────────────────


def reduced_potential(temperature: float, potential: float) -> float:
    beta = 1.0 / (KB_KJ * temperature) if temperature != 0 else 1.0 / KB_KJ
    return beta * potential


def build_swap_matrix(
    ex_list: list[int],
    poten: dict[int, float],
    temp: dict[int, float],
) -> list[list[float]]:
    """matrix[i][j] = reduced_potential(T_j, E_i)."""
    n = len(ex_list)
    matrix = [[0.0] * n for _ in range(n)]
    for i, r_i in enumerate(ex_list):
        for j, r_j in enumerate(ex_list):
            matrix[i][j] = reduced_potential(temp[r_j], poten[r_i])
    return matrix


# ── Pairwise-independence sampling ────────────────────────────────────────────


def _weighted_choice(weights: list[float]) -> Optional[int]:
    total = sum(weights)
    if total <= 0:
        return None
    r = random.random() * total
    for idx, w in enumerate(weights):
        r -= w
        if r < 0:
            return idx
    return len(weights) - 1


def pairwise_independence_sampling(
    repl_i: int,
    candidates: list[int],
    u_matrix: list[list[float]],
    ex_list: list[int],
    verbose: bool = False,
) -> int:
    g2l = {r: k for k, r in enumerate(ex_list)}
    n = len(candidates)
    ps = np.zeros(n)
    du = np.zeros(n)
    i_i = -1
    i_pos = g2l[repl_i]

    for jj, repl_j in enumerate(candidates):
        j_pos = g2l[repl_j]
        du[jj] = (
            u_matrix[i_pos][j_pos]
            + u_matrix[j_pos][i_pos]
            - u_matrix[i_pos][i_pos]
            - u_matrix[j_pos][j_pos]
        )
        if repl_j == repl_i:
            i_i = jj

    eu = np.exp(-du)
    f = 1.0 / max(float(n - 1), 1.0)
    pii = 1.0

    for jj, repl_j in enumerate(candidates):
        if repl_j == repl_i:
            continue
        ps[jj] = f if eu[jj] > 1.0 else f * eu[jj]
        pii -= ps[jj]

    if i_i >= 0:
        ps[i_i] = max(0.0, pii)

    if verbose:
        print(f"  [exchange] rid={repl_i} | du={np.array2string(du, precision=4)}")
        print(f"  [exchange]           | eu={np.array2string(eu, precision=4)}")
        print(f"  [exchange]           | ps={np.array2string(ps, precision=4)}")

    chosen_idx = _weighted_choice(list(ps))
    return candidates[chosen_idx] if chosen_idx is not None else repl_i


def attempt_exchange(
    repl_i: int, ex_list: list[int], u_matrix: list[list[float]], verbose: bool = False
) -> int:
    """Only offer adjacent-temperature replicas as swap candidates."""
    pos = ex_list.index(repl_i)
    candidates = [ex_list[k] for k in range(len(ex_list)) if abs(k - pos) <= 1]
    return pairwise_independence_sampling(
        repl_i, candidates, u_matrix, ex_list, verbose
    )


def select_pairs(
    ex_list: list[int], u_matrix: list[list[float]], verbose: bool = False
) -> list[tuple[int, int]]:
    """Greedy non-overlapping swap pair selection."""
    exchange_pairs: list[tuple[int, int]] = []
    occupied: set = set()
    for r_i in ex_list:
        if r_i in occupied:
            continue
        r_j = attempt_exchange(r_i, ex_list, u_matrix, verbose=verbose)
        if r_i == r_j or r_j in occupied:
            continue
        pair = (min(r_i, r_j), max(r_i, r_j))
        if pair not in exchange_pairs:
            exchange_pairs.append(pair)
            occupied.update({r_i, r_j})
    return sorted(exchange_pairs)


# ── Coordinate swap via OpenMM ────────────────────────────────────────────────


def _do_coordinate_swap(
    exchange_pairs: list[tuple[int, int]],
    ex_list: list[int],
    cycle: int,
    temp: dict[int, float],
    work_dir: Path,
    top_file: Path,
    gro_file: Path,
    top_include_dir: str = GROMACS_TOP_INCLUDE,
) -> None:
    """
    Load all replica checkpoints, swap coordinates for accepted pairs,
    and re-save checkpoints for ALL replicas.

    Velocity policy: velocities are NEVER re-drawn here.
      - Swapped replicas    → foreign coordinates, own velocities.
                              Langevin thermostat re-equilibrates naturally.
      - Non-swapped replicas → unchanged context, checkpoint re-saved so
                               simulation.py always finds a valid file at
                               checkpoint_{rid}.{cycle}.

    Saving ALL checkpoints (not just swapped ones) is required because
    simulation.py calls loadCheckpoint(checkpoint_{rid}.{cycle}) after
    every resume_event — it must exist for every replica regardless of
    whether a swap occurred.
    """
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

    simulations: dict[int, Simulation] = {}
    for rid in ex_list:
        integrator = LangevinMiddleIntegrator(
            temp[rid] * kelvin, 1 / picosecond, 0.002 * picoseconds
        )
        platform = Platform.getPlatformByName("OpenCL")
        sim = Simulation(
            top.topology, system, integrator, platform, {"Precision": "mixed"}
        )
        sim.loadCheckpoint(str(_checkpoint_path(work_dir, rid)))
        simulations[rid] = sim
        print(f"[exchange] loaded checkpoint replica {rid}", flush=True)

    for r_i, r_j in exchange_pairs:
        state_i = simulations[r_i].context.getState(
            getPositions=True, getVelocities=True
        )
        state_j = simulations[r_j].context.getState(
            getPositions=True, getVelocities=True
        )
        # Swap coordinates only — velocities stay with their original replica
        simulations[r_i].context.setPositions(state_j.getPositions())
        simulations[r_j].context.setPositions(state_i.getPositions())
        print(
            f"[exchange] swapped positions: {r_i} ↔ {r_j} "
            f"(T={temp[r_i]}K / {temp[r_j]}K) — velocities preserved",
            flush=True,
        )

    # Re-save ALL checkpoints (swapped and non-swapped)
    swapped = {r for pair in exchange_pairs for r in pair}
    for rid in ex_list:
        simulations[rid].saveCheckpoint(str(_checkpoint_path(work_dir, rid)))
        label = "swapped" if rid in swapped else "unchanged"
        print(f"[exchange] saved checkpoint replica {rid} ({label})", flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────


def run_exchange(
    ex_list: list[int],
    cycle: int,
    work_dir: Path,
    top_file: Path,
    gro_file: Path,
    *,
    top_include_dir: str = GROMACS_TOP_INCLUDE,
    verbose: bool = False,
) -> dict:
    """
    One full exchange round:
      1. Load per-replica energies + temperatures from JSON sidecars.
      2. Build the reduced-potential swap matrix.
      3. Select non-overlapping swap pairs.
      4. Execute coordinate swaps and re-save all checkpoints.
      5. Return a summary dict logged by the workflow manager.
    """
    work_dir = Path(work_dir)
    poten, temp = _load_replica_states(ex_list, cycle, work_dir)
    print(f"[exchange] cycle {cycle} | pot={poten} | T={temp}", flush=True)

    swap_matrix = build_swap_matrix(ex_list, poten, temp)
    exchange_pairs = select_pairs(ex_list, swap_matrix, verbose=verbose)
    print(f"[exchange] selected pairs: {exchange_pairs}", flush=True)

    if exchange_pairs:
        _do_coordinate_swap(
            exchange_pairs,
            ex_list,
            cycle,
            temp,
            work_dir,
            top_file,
            gro_file,
            top_include_dir,
        )
    else:
        # No swaps accepted — still re-save all checkpoints so simulation.py
        # always finds checkpoint_{rid}.{cycle} after resume_event is set.
        print("[exchange] no swaps accepted — re-saving all checkpoints", flush=True)
        _do_coordinate_swap(
            [],
            ex_list,
            cycle,
            temp,
            work_dir,
            top_file,
            gro_file,
            top_include_dir,
        )

    log = {
        "cycle": cycle,
        "ex_list": ex_list,
        "potentials": {str(k): v for k, v in poten.items()},
        "temperatures": {str(k): v for k, v in temp.items()},
        "exchange_pairs": exchange_pairs,
        "n_swaps": len(exchange_pairs),
    }
    log_path = work_dir / "exchange.json"
    with open(log_path, "a") as fh:
        json.dump(log, fh)
        fh.write("\n")
    print(f"[exchange] log → {log_path}", flush=True)
    return log
