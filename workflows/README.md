# DeepDriveSim Workflows

AI-steered ensemble simulation workflows built on [DeepDriveSim](https://github.com/radical-collaboration/DeepDriveSim). Each workflow follows the same iterative loop driven by `DDSimManager`:

```
simulate → train → evaluate/predict → select → repeat
```

Tasks are dispatched via [`radical.asyncflow`](https://github.com/radical-cybertools/radical.asyncflow) and scheduled by the [ROSE](https://github.com/radical-cybertools/radical.asyncflow) learner. Two execution backends are supported: `ConcurrentExecutionBackend` (local/testing) and `DragonExecutionBackend` (distributed HPC).

---

## Available Workflows

| Workflow | Simulation | Use Case |
|---|---|---|
| [`miniapps_workflow/`](miniapps_workflow/) | GPU matrix ops via [wfMiniAPI](https://github.com/radical-cybertools/workflow-mini-apps) | Performance benchmarking, infrastructure testing |
| [`ddmd_workflow/`](ddmd_workflow/) | OpenMM molecular dynamics + CVAE | Real protein folding / conformational sampling |
| [`dummy_workflow/`](dummy_workflow/) | NumPy random data + sklearn | Lightweight integration testing |

---

## Shared Design

All workflows extend `DDSimManager` and implement the same set of hooks:

| Method | Purpose |
|---|---|
| `init_sim_queue()` | Populate the queue with initial simulation inputs |
| `simulation(sim_inputs)` | Launch one simulation task (registered with ROSE) |
| `check_train_status()` | Poll until enough data exists to start training |
| `train_model()` | Run one training iteration |
| `evaluate_simulations()` | Run inference / prediction on completed sims |
| `stop_simulation(prediction)` | Return `True` to permanently cancel a running sim |
| `post_process_sim(sim_idx)` | Cleanup after a sim completes |
| `finalize_results()` | Check exit condition; set `run_workflow = False` when done |
| `close()` | Graceful shutdown (no-op when sharing an asyncflow engine) |

### Execution Backends

- **`ConcurrentExecutionBackend`** — runs tasks as local subprocesses; no Dragon required. Good for single-node testing.
- **`DragonExecutionBackend`** — uses the Dragon HPC runtime for distributed multi-node execution. Required for multi-GPU campaigns.

> **Note:** Do not `module load openmpi` when using Dragon — it conflicts with Dragon's bundled MPI and causes a head-process crash (`code -4`). Also run `rm -f /dev/shm/DDyad*` to clear stale Dragon shared memory before resubmitting.

---

## Common Structure

Each workflow directory contains the same set of files:

```
<workflow>/
├── env_setup.sh       # Creates the conda environment(s)
├── config.yaml        # Workflow configuration (see below)
├── run_workflow.py    # Single-replica entry point
├── gpu_sbatch.sh      # Slurm batch script (GPU partition, Bridges-2)
├── requirements.txt   # Pip dependencies beyond the base DDSim install
└── <workflow>.py      # Workflow class (extends DDSimManager)
```

`miniapps_workflow` additionally provides:
- `run_camp.py` — multi-replica campaign entry point (Dragon + telemetry)
- `plot_utilization.py` — plot CPU/GPU utilization from telemetry output

---

## 1. Environment Setup

**Prerequisites:**
- `$PROJECT` environment variable pointing to your project base directory
- DeepDriveSim cloned at `$PROJECT/DeepDriveSim`
- Anaconda available via `module load anaconda3`

Run the setup script once from inside the workflow directory:

```bash
cd $PROJECT/DeepDriveSim/workflows/<workflow>
bash env_setup.sh
```

`miniapps_workflow` also requires [wfMiniAPI](https://github.com/radical-cybertools/workflow-mini-apps); `env_setup.sh` clones it automatically if not already present.

`ddmd_workflow` creates **three** separate environments (`deepdrivesim`, `conda-openmm`, `conda-keras`) because OpenMM and TensorFlow have conflicting dependencies. All other workflows use a single environment.

---

## 2. Configuration

Each workflow reads a `config.yaml`. Common keys shared across all workflows:

```yaml
debug: false               # Enable verbose per-iteration logging
engine: dragon             # Execution backend: dragon | concurrent
home_dir: "~/WorkflowOut" # Base directory for all output data

total_num_sim: 100         # Total simulations to run
max_sim_batch: 4           # Max concurrently running simulations
training_cores: 1          # Slots freed for training when free_resources_for_train=true
free_resources_for_train: true

num_replicas: 4            # Parallel workflow replicas (multi-replica runs only)
src_dir: ""                # Path to task scripts; empty = auto-detect
```

Workflow-specific keys (e.g. `num_step`, `num_epochs`, `phase` for miniapps; experiment YAML path for ddmd) are documented in each workflow's own `config.yaml`.

---

## 3. Running

### Single Replica

```bash
conda activate $PROJECT/conda_env/<workflow>
cd $PROJECT/DeepDriveSim/workflows/<workflow>

# Dragon backend:
dragon -s run_workflow.py --config_file config.yaml

# ConcurrentExecutionBackend (no Dragon required):
python run_workflow.py --config_file config.yaml
```

### Multi-Replica Campaign (`miniapps_workflow` only)

Runs `num_replicas` parallel instances, one per GPU. Each replica gets its own GPU affinity via a Dragon Policy and writes output to a separate subdirectory.

```bash
conda activate $PROJECT/conda_env/miniapps_workflow
cd $PROJECT/DeepDriveSim/workflows/miniapps_workflow

dragon -s run_camp.py --config_file config.yaml
```

### Slurm Batch

```bash
cd $PROJECT/DeepDriveSim/workflows/<workflow>
sbatch gpu_sbatch.sh
```

Edit `#SBATCH --time` and `#SBATCH -A` before submitting. Output goes to `slurm-<jobid>.out` in the working directory.

---

## 4. Plotting Telemetry (`miniapps_workflow`)

After a campaign run, plot CPU and GPU utilization over time:

```bash
cd $PROJECT/DeepDriveSim/workflows/miniapps_workflow

# From Dragon telemetry:
python plot_utilization.py --telemetry_dir telemetry-results

# From NVML telemetry:
python plot_utilization.py --telemetry_dir nvml-telemetry
```

Plots are saved to `plots/`. GPU columns are auto-detected; only GPUs with non-zero activity are shown. Metrics include CPU utilization (%), GPU compute utilization per device (%), and GPU memory utilization per device (%).

> **Note:** `nvidia-ml-py` must be installed for NVML telemetry. If unavailable, GPU metrics are silently skipped.
