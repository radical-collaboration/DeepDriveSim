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
│                     DDSim Manager                            │
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

## Installation

### From Source

```bash
git clone https://github.com/radical-collaboration/DeepDriveSim.git
cd DeepDriveSim
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With Documentation Dependencies

```bash
pip install -e ".[doc]"
```

##  Documentation
 [DeepDriveSim Documentation](https://radical-collaboration.github.io/DeepDriveSim/)

## Campaign Manager Integration

`DDMdWorkflow` exposes a lightweight hook that lets an external Campaign
Manager (CM) react to each completed iteration — for example to queue
downstream analysis or inference replicas.

### How it works

`DDMdWorkflow.__init__` accepts an optional `on_ready` keyword argument.
When provided, the method `_signal_ready()` calls it after every completed
iteration (inside `finalize_results()`).

```
DDMdWorkflow iteration N completes
  └─ finalize_results()
       └─ _signal_ready()          ← fires every iteration
            └─ on_ready()          ← CM hook injected at construction
                 └─ CM queues +1 dependent replica
```

`on_ready` may be a plain callable or a coroutine function — `_signal_ready`
handles both:

```python
async def _signal_ready(self) -> None:
    if self._on_ready is not None:
        result = self._on_ready()
        if asyncio.iscoroutine(result):
            await result
```

The hook fires **once per iteration**, not once per replica lifetime.  A
single DDMdWorkflow replica that runs `max_iteration=5` will call `on_ready`
five times, each time the full simulation → training → selection loop
completes.

### Wrapper pattern (SPHERICAL Campaign Manager)

The SPHERICAL CM runs `DDMdWorkflow` via a thin `DDMdWrapperWorkflow` that
inherits `BaseWorkflow`.  It wires `on_ready` to the CM's `_signal_done()`
so that each completed DDMd iteration cascades downstream replicas based on
the pipeline config — without the workflow knowing downstream group names.

```python
# workflows/run_campaign/ddmd_workflow.py
class DDMdWrapperWorkflow(BaseWorkflow):
    workflow_id = "ddmd"

    async def run(self, replica_id: str) -> None:
        workflow = DDMdWorkflow(
            asyncflow=self.asyncflow,
            config=replica_config_path,
            name=replica_id.replace("_", ""),
            on_ready=lambda: self._signal_done(),   # ← CM hook
            policies=self.policies,
            engine_dragon=self.engine_dragon,
        )
        await workflow.start()
```

`_signal_done()` notifies the CM, which adds +1 replica to every group
whose `dependencies` config field lists `"md"`.  The pipeline topology lives
entirely in `config.yaml`; neither `DDMdWorkflow` nor `DDMdWrapperWorkflow`
hardcodes downstream names.

### Standalone usage (no CM)

When running `DDMdWorkflow` directly (e.g. in tests or as a standalone
script), omit `on_ready` or pass `None`:

```python
workflow = DDMdWorkflow(
    asyncflow=asyncflow,
    config="config.yaml",
    name="ddmd",
    # on_ready omitted → _signal_ready() is a no-op
)
await workflow.start()
```

### Constructor parameters summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `asyncflow` | `WorkflowEngine` | required | Shared radical-asyncflow engine |
| `config` | `str` | required | Path to experiment YAML config |
| `name` | `str` | `"ddsim"` | Replica name; used to namespace experiment directories |
| `on_ready` | callable or coroutine function | `None` | Hook called after each completed iteration; CM injects `lambda: self._signal_done()` |
| `policies` | `list[Policy]` | `[]` | Dragon GPU policies (one per assigned GPU); split into per-GPU policies internally |
| `debug` | `bool` | `False` | Enable verbose RHAPSODY debug logging |
| `tf_gpu_wrapper` | `str` | auto-detected | Shell wrapper script for GPU training subprocess |
| `tf_cpu_wrapper` | `str` | auto-detected | Shell wrapper script for CPU agent subprocess |

---

## Examples

See the [examples/](examples/) directory for complete working examples:

- **[ddmd_pipeline/](examples/ddmd_pipeline/)**: Deep-Learning Driven Adaptive Molecular Simulations
    https://github.com/DeepDriveMD/DeepDriveMD-pipeline
- **[dummy_pipeline/](examples/dummy_pipeline/)**: Standalone demo with synthetic data
- **[miniapps_pipeline/](examples/miniapps_pipeline/)**  
    Reference scalable HPC workflow using RADICAL-Cybertools Workflow mini-apps 
    https://github.com/radical-cybertools/workflow-mini-apps


## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=ddsim --cov-report=html
```

## Development

### Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code style
ruff check ddsim tests

# Format code
ruff format ddsim tests
```

### Using tox

```bash
# Run all tests across Python versions
tox

# Run linting
tox -e lint

# Run formatting
tox -e format
```

## Dependencies

- **[RADICAL-AsyncFlow](https://github.com/radical-cybertools/radical-asyncflow)**: Async workflow orchestration
- **[ROSE](https://radical-cybertools.github.io/ROSE/)**: Machine learning integration for HPC
- **[PyYAML](https://pyyaml.org/)**: Configuration file parsing

## Citation

If you use DeepDriveSim in your research, please cite:

```bibtex
@inproceedings{lee2019DeepDriveSim,
  author={Lee, Hyungro and Turilli, Matteo and Jha, Shantenu and Bhowmik, Debsindhu and Ma, Heng and Ramanathan, Arvind},
  booktitle={2019 IEEE/ACM Third Workshop on Deep Learning on Supercomputers (DLS)},
  title={DeepDriveSim: Deep-Learning Driven Adaptive Molecular Simulations},
  year={2019},
  pages={12-19},
  doi={10.1109/DLS49591.2019.00007}
}
```

**Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8945122)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Brookhaven National Laboratory (BNL)
- RADICAL Laboratory at Rutgers University
- Argonne National Laboratory
- This work was supported by the DOE Office of Science
