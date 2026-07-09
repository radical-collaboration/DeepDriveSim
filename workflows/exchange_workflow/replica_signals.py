#!/usr/bin/env python3
"""
replica_signals.py
──────────────────
File-based drop-in replacement for asyncio.Event that works across
Dragon worker processes (which cannot share asyncio primitives).

Each "event" is a sentinel file on a shared filesystem.  The simulation
worker writes (sets) the file; the exchange loop reads (waits for) it.

Usage — mirrors asyncio.Event exactly:
    signals = ReplicaSignalSet(work_dir, num_replicas)

    # In simulation worker (replaces ready_events[rid].set()):
    signals.ready[rid].set(cycle)

    # In exchange loop (replaces await ready_events[rid].wait()):
    await signals.ready[rid].wait(cycle)

    # In simulation worker (replaces await resume_events[rid].wait()):
    await signals.resume[rid].wait(cycle)

    # In exchange loop (replaces resume_events[rid].set()):
    signals.resume[rid].set(cycle)

    # clear() is no longer needed — each cycle uses a unique file name.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


class _FileSignal:
    """A single file-based signal for one replica, one direction."""

    def __init__(self, work_dir: Path, prefix: str, rid: int):
        self._dir    = Path(work_dir)
        self._prefix = prefix
        self._rid    = rid

    def _path(self, cycle: int) -> Path:
        return self._dir / f"{self._prefix}_{self._rid:04d}_{cycle:04d}.signal"

    def set(self, cycle: int) -> None:
        """Write the sentinel file (non-blocking, safe to call from sync code)."""
        self._path(cycle).touch()

    def is_set(self, cycle: int) -> bool:
        return self._path(cycle).exists()

    async def wait(self, cycle: int, poll_interval: float = 0.1) -> None:
        """Async-poll until the sentinel file appears."""
        path = self._path(cycle)
        while not path.exists():
            await asyncio.sleep(poll_interval)

    def clear(self, cycle: int) -> None:
        """Remove the sentinel file (optional cleanup)."""
        p = self._path(cycle)
        if p.exists():
            p.unlink()


class ReplicaSignalSet:
    """
    Holds two lists of _FileSignal — one per direction — for all replicas.

    Attributes
    ----------
    ready  : List[_FileSignal]   sim → exchange  ("I finished my window")
    resume : List[_FileSignal]   exchange → sim  ("swap done, continue")
    """

    def __init__(self, work_dir: Path, num_replicas: int):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self.ready  = [
            _FileSignal(work_dir, "ready",  rid) for rid in range(num_replicas)
        ]
        self.resume = [
            _FileSignal(work_dir, "resume", rid) for rid in range(num_replicas)
        ]
