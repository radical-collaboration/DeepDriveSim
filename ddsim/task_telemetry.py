"""Task-side telemetry interface: lets task processes report observations.

Task scripts call :func:`emit`; the workflow manager receives the events and forwards
them into its RHAPSODY ``TelemetryManager`` (same stream as task lifecycle events and
other ``ddsim.*`` custom events)::

    from ddsim import task_telemetry

    task_telemetry.emit(
        "ddsim.DataRead", object_id=path, version=v, payload_bytes=n, t_start=t0
    )

Design rules
------------
* Scripts depend only on ``emit()``; *how* events travel is a transport detail
  chosen by the manager through environment variables, so the transport can change
  without touching scripts.
* The timestamp (``t``) is taken when ``emit()`` is called, not when the manager
  receives the event.
* Task identity (key, replica) comes from the environment set by the manager.
* ``emit()`` never raises: telemetry problems must not break a task. With no
  transport configured (e.g. a script run by hand) it is a no-op.

Environment (set by the manager)
--------------------------------
DDSIM_TELEMETRY_TRANSPORT   "stdout" (v1). Unset/empty -> telemetry off.
DDSIM_TASK_KEY              identity of this task execution, e.g. "min1.sim3.a0"
DDSIM_REPLICA               workflow replica name, e.g. "min1"

Transport v1: stdout
--------------------
Each event is written as one line ``DDSIM_TEL {json}`` to stdout. asyncflow captures
task stdout (``capture_stdio=True``) and the manager parses these lines when the
task's future resolves (:func:`collect`). Consequence: events arrive when the task
finishes, not in real time, and are lost if the task is killed before its stdout is
flushed. A real-time transport (socket, Dragon queue, OTLP) can replace this behind
the same ``emit()`` call.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time

MARKER = "DDSIM_TEL "
ENV_TRANSPORT = "DDSIM_TELEMETRY_TRANSPORT"
ENV_TASK_KEY = "DDSIM_TASK_KEY"
ENV_REPLICA = "DDSIM_REPLICA"
NAMESPACE = "ddsim."


def enabled() -> bool:
    return os.environ.get(ENV_TRANSPORT, "") != ""


def emit(event: str, **fields) -> None:
    """Report one observation from inside a task. Never raises."""
    try:
        transport = os.environ.get(ENV_TRANSPORT, "")
        if not transport:
            return
        if not event.startswith(NAMESPACE):
            raise ValueError(f"event name must start with {NAMESPACE!r}: {event!r}")
        record = {
            "event": event,
            "t": time.time(),
            "task_key": os.environ.get(ENV_TASK_KEY, ""),
            "replica": os.environ.get(ENV_REPLICA, ""),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **fields,
        }
        if transport == "stdout":
            sys.stdout.write(MARKER + json.dumps(record, default=str) + "\n")
            sys.stdout.flush()
        else:
            raise ValueError(f"unknown {ENV_TRANSPORT}={transport!r}")
    except Exception as e:  # telemetry must never break the task
        try:
            sys.stderr.write(f"[ddsim.task_telemetry] dropped {event!r}: {e!r}\n")
        except Exception:
            pass


# ------------------------------------------------------------------------- data helpers
# Thin wrappers so the data-event schema is defined once (used for ED6 and future ED
# metrics).
def file_version(path) -> str:
    """Version id of a file's current contents (mtime in ns; '' if unavailable)."""
    try:
        return str(os.stat(path).st_mtime_ns)
    except OSError:
        return ""


def data_written(path, payload_bytes: int, t_start: float) -> None:
    """Call right after a write completed (file closed). Emits ``ddsim.DataWritten``."""
    emit(
        "ddsim.DataWritten",
        object_id=os.path.abspath(str(path)),
        version=file_version(path),
        payload_bytes=int(payload_bytes),
        t_start=float(t_start),
    )


def data_read(path, payload_bytes: int, t_start: float, version: str) -> None:
    """Call right after a read completed. Emits ``ddsim.DataRead``.

    ``version`` must be taken *before* the read (``file_version(path)``) so it
    identifies the contents actually read.
    """
    emit(
        "ddsim.DataRead",
        object_id=os.path.abspath(str(path)),
        version=version,
        payload_bytes=int(payload_bytes),
        t_start=float(t_start),
    )


# ------------------------------------------------------------------------- manager side
def env_for_task(task_key: str, replica: str, transport: str = "stdout") -> dict:
    """Environment variables the manager sets when launching a task."""
    return {ENV_TRANSPORT: transport, ENV_TASK_KEY: task_key, ENV_REPLICA: replica}


def collect(task_result) -> list[dict]:
    """Extract task events from a finished task's captured stdout.

    ``task_result`` is what the asyncflow future resolved to: with
    ``capture_stdio=True`` the path
    of the stdout file, otherwise the stdout text itself.
    """
    if task_result is None:
        return []
    text = str(task_result)
    try:
        if "\n" not in text and os.path.isfile(text):
            with open(text, errors="replace") as f:
                text = f.read()
    except OSError:
        return []
    events = []
    for line in text.splitlines():
        if line.startswith(MARKER):
            try:
                events.append(json.loads(line[len(MARKER) :]))
            except json.JSONDecodeError:
                continue
    return events
