# Creating a Custom Workflow

## Extend `DDSimManager` to create your own workflow:

```python
from ddmd import DDSimManager

class MyWorkflow(DDSimManager):
    def __init__(self, asyncflow, **kwargs):
        super().__init__(asyncflow)
        # Your initialization code
        self._register_learner_tasks()

    def _register_learner_tasks(self):
        @self.learner.simulation_task(as_executable=False)
        async def simulation(*args, **kwargs):
            # Your simulation logic
            pass
        self.simulation = simulation

    def stop_simulation(self, prediction):
        # Return True to cancel simulation based on prediction
        return prediction < 0.5

    async def init_sim_queue(self):
        # Populate self.sim_task_queue with simulation inputs
        pass

    async def check_train_data(self):
        # Return True when ready to start training
        return True

    async def train_model(self):
        # Your training logic
        pass

    async def clean_sim_data(self, sim_ind):
        # Cleanup files for canceled simulations
        pass
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_sim_batch` | Maximum concurrent simulations | 4 |
| `training_cores` | CPU cores reserved for training | 1 |
| `training_threshold` | Accuracy threshold for training | 0.5 |
| `prediction_threshold` | Score threshold for cancellation | 0.5 |
| `force_start_training` | Skip waiting for data threshold | False |
| `clean_unregistered_sims` | Delete files from canceled sims | True |