# Quick Start

## Basic Usage with DummyWorkflow (for testing)

```python
import asyncio
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine
from concurrent.futures import ThreadPoolExecutor
from ddmd import DummyWorkflow

async def main():
    # Create execution backend
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    # Initialize workflow
    workflow = DummyWorkflow(
        asyncflow=asyncflow,
        max_sim_batch=4,
        training_cores=1
    )

    # Run the adaptive learning loop
    await workflow.start()
    await workflow.close()

asyncio.run(main())
```