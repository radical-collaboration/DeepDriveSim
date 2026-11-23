"""Concurrent command execution with process-safe inference-driven cancellation.

The pinned RHAPSODY concurrent backend cancels its asyncio waiter, but does not
terminate the subprocess. Keep this adapter local to inference-only mini-apps;
stock workflows and installed runtime packages are not modified. This guarantee
covers command tasks, not arbitrary Python functions in a thread executor.

POSIX only: it relies on process groups.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from typing import Any

from rhapsody.backends import ConcurrentExecutionBackend

#: Seconds a command gets to exit after SIGTERM before its group is killed.
TERMINATE_GRACE_S = 2.0


async def _terminate_group(process: asyncio.subprocess.Process) -> None:
    """Terminate and reap ``process`` together with its process group.

    The command runs as the leader of its own session, so its PID is also its
    process group ID.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), TERMINATE_GRACE_S)
    # Kill any descendants that survived TERM, even if their parent exited first.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


class InferenceExecutionBackend(ConcurrentExecutionBackend):
    """Concurrent backend whose cancellation really stops the command.

    Each command runs in its own POSIX process group. Cancellation is
    acknowledged only after that group has been terminated and reaped.
    """

    async def _execute_command(
        self, task: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        """Run one command task; terminate its process group if cancelled.

        Returns:
            The task, updated with ``stdout``, ``stderr`` and ``exit_code``, and
            its final state: ``DONE`` or ``FAILED``.
        """
        options = task.get("task_backend_specific_kwargs", {})
        capture = task.get("capture_stdio", False)
        with contextlib.ExitStack() as files:
            stdout: Any = asyncio.subprocess.PIPE
            stderr: Any = asyncio.subprocess.PIPE
            stdout_path = os.path.join(self._work_dir, f"{task['uid']}.stdout")
            stderr_path = os.path.join(self._work_dir, f"{task['uid']}.stderr")
            if capture:
                # Opened off the event loop; closed when the stack unwinds.
                stdout = files.enter_context(
                    await asyncio.to_thread(open, stdout_path, "wb")
                )
                stderr = files.enter_context(
                    await asyncio.to_thread(open, stderr_path, "wb")
                )
            process = await self._launch(task, options, stdout, stderr)
            task["command_pid"] = process.pid
            try:
                if capture:
                    await process.wait()
                    out, err = stdout_path, stderr_path
                else:
                    raw_out, raw_err = await process.communicate()
                    out, err = raw_out.decode(), raw_err.decode()
            except asyncio.CancelledError:
                await _terminate_group(process)
                task["exit_code"] = process.returncode
                raise
        task.update(stdout=out, stderr=err, exit_code=process.returncode)
        return task, "DONE" if process.returncode == 0 else "FAILED"

    @staticmethod
    async def _launch(
        task: dict[str, Any], options: dict[str, Any], stdout: Any, stderr: Any
    ) -> asyncio.subprocess.Process:
        """Start the command as the leader of a new session and process group."""
        settings = {
            "stdout": stdout,
            "stderr": stderr,
            "cwd": options.get("cwd"),
            "env": options.get("env"),  # None inherits the parent environment
            "start_new_session": True,
        }
        command = [task["executable"], *task.get("arguments", [])]
        if options.get("shell", False):
            starting = asyncio.create_subprocess_shell(shlex.join(command), **settings)
        else:
            starting = asyncio.create_subprocess_exec(*command, **settings)
        launch = asyncio.create_task(starting)
        try:
            return await asyncio.shield(launch)
        except asyncio.CancelledError:
            # Cancellation can arrive while the OS process is being created.
            # Finish the launch so the new process is not leaked, then stop it.
            process = await launch
            await _terminate_group(process)
            raise

    async def cancel_task(self, uid: str) -> bool:
        """Cancel a task and wait until its command has been reaped.

        Returns:
            True if a running task was cancelled, False if it was unknown or
            had already finished.
        """
        task = self.tasks.get(uid)
        if task is None or task["future"].done():
            return False
        future = task["future"]
        future.cancel()
        await asyncio.gather(future, return_exceptions=True)
        self._callback_func(task, "CANCELED")
        return True

    async def cancel_all_tasks(self) -> int:
        """Cancel every running task; return how many were cancelled."""
        running = [uid for uid, task in self.tasks.items() if not task["future"].done()]
        results = await asyncio.gather(*(self.cancel_task(uid) for uid in running))
        self.tasks.clear()
        return sum(results)
