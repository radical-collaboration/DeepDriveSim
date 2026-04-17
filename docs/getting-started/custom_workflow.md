# Creating a Custom Workflow

## Extend `DDSimManager` to create your own workflow:

```python
import asyncio
from ddsim import DDSimManager


class MyWorkflow(DDSimManager):
    def __init__(self, **kwargs):
        self.max_sim_batch = kwargs.get("max_sim_batch", 4)
        self.sim_batch_size = self.max_sim_batch
        self.retrain_model = True
        super().__init__()

    def simulation(self, sim_inputs=None, **kwargs):
        # Wrap your simulation coroutine in a task
        return asyncio.create_task(self._run_sim(sim_inputs))

    async def _run_sim(self, sim_inputs):
        # Your simulation logic here
        pass

    def stop_simulation(self, prediction, **kwargs):
        # Return True to cancel simulation based on prediction
        return prediction < 0.5

    async def init_sim_queue(self):
        # Populate self.sim_task_queue with simulation inputs
        for i in range(10):
            await self.sim_task_queue.put({"sim_idx": f"sim_{i}"})

    async def check_train_status(self):
        # Return True when enough data is available to start training
        return len(self.completed_sims) >= 5

    async def train_model(self):
        # Your training logic
        pass

    async def run_inference(self):
        # Populate self.sim_predictions with scores for running sims
        pass

    async def add_sims_to_queue(self, sim_ids):
        # Re-queue sims that were cancelled to free resources
        for sim_id in sim_ids:
            await self.sim_task_queue.put({"sim_idx": sim_id})

    async def post_process_sim(self, sim_idx):
        # Cleanup or bookkeeping after each completed sim
        pass

    async def post_process(self):
        # Called after each inference iteration
        if not self.sim_task_queue.qsize() and not self.registered_sims:
            self.run_workflow = False
            self.shutting_down.set()

    async def close(self):
        # Graceful shutdown
        self.shutting_down.set()
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sleep_time` | Seconds between prediction/train checks | 20 |
| `debug` | Enable verbose debug logging | False |
| `free_resources_for_train` | Preempt sims to free resources for training | False |
