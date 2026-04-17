# Quick Start

## Basic Usage

Subclass `DDSimManager` and implement the required methods:

```python
import asyncio
from ddsim import DDSimManager


class MyWorkflow(DDSimManager):
    def __init__(self):
        self.max_sim_batch = 4
        self.sim_batch_size = self.max_sim_batch
        self.retrain_model = False
        super().__init__()

    def simulation(self, sim_inputs=None, **kwargs):
        return asyncio.create_task(asyncio.sleep(1.0))

    async def init_sim_queue(self):
        for i in range(8):
            await self.sim_task_queue.put({"sim_idx": f"sim_{i}"})

    async def check_train_status(self):
        return True

    async def post_process_sim(self, sim_idx):
        pass

    async def post_process(self):
        self.run_workflow = False
        self.shutting_down.set()

    async def add_sims_to_queue(self, sim_ids):
        pass

    def stop_simulation(self, **kwargs):
        return False

    async def close(self):
        pass


asyncio.run(MyWorkflow().start())
```
