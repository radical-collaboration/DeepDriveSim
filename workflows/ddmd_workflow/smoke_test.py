#!/usr/bin/env python3
"""Smoke test: verifies asyncflow + rhapsody install and capture_stdio support."""
import asyncio, sys

import radical.asyncflow as rf
import rhapsody
from rhapsody.backends import ConcurrentExecutionBackend

print(f"radical.asyncflow : {rf.__version__}")
print(f"rhapsody          : {rhapsody.__version__}")

async def main():
    backend = await ConcurrentExecutionBackend()
    flow    = await rf.WorkflowEngine.create(backend=backend)

    @flow.executable_task(capture_stdio=True)
    async def hello(**kwargs):
        return "echo hello_smoke_test"

    result = await hello()
    print(f"task result       : {result!r}")
    await flow.shutdown()
    print("\nSMOKE TEST PASSED")

asyncio.run(main())
