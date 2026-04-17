# DeepDriveSim

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Deep learning-driven Adaptive Simulations**

DeepDriveSim is a toolkit developed by Brookhaven National Laboratory (BNL) / RADICAL Laboratory at Rutgers University, in collaboration with Argonne National Laboratory. It implements an AI-steered ensemble simulation workflow that uses deep learning models to guide and optimize simulations in real-time.

## Features

- **Adaptive Simulation Management**: Dynamically manages molecular simulations based on ML predictions
- **Active Learning Loop**: Implements simulation → training → prediction → cancellation → re-submission cycle
- **Multiple Execution Backends**: Supports local execution, RHAPSODY (HPC), and Dragon distributed computing
- **Resource-Aware Scheduling**: Automatically balances resources between simulations and training
- **GPU Support**: Automatic GPU detection and utilization
- **Extensible Architecture**: Easy to customize for different simulation types and ML models

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DDSim Manager                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Simulation  │  │  Training   │  │     Prediction      │  │
│  │   Queue     │──│   Module    │──│      Module         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │             │
│         ▼                ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              ROSE / RADICAL-AsyncFlow               │    │
│  │           (Execution Backend Abstraction)           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```


### 1. Basic Usage

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

### 2. Creating a Custom Workflow

See the [Custom Workflow](getting-started/custom_workflow.md) guide for a full example.

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sleep_time` | Seconds between prediction/train checks | 20 |
| `debug` | Enable verbose debug logging | False |
| `free_resources_for_train` | Preempt sims to free resources for training | False |
