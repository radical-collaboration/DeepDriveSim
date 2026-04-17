# DeepDriveMD workflow

The DeepDriveMD (DDMD) workflow implements an AI-steered molecular dynamics workflow that couples MD simulations with machine learning to adaptively guide sampling of protein conformational space. It orchestrates five stages in an iterative loop:

1. **Molecular Dynamics** -- Runs OpenMM simulations in parallel across multiple input PDB structures.
2. **Aggregation** -- Combines simulation outputs into training datasets (optional, can be skipped).
3. **Machine Learning** -- Trains a convolutional variational autoencoder (CVAE) on contact maps from MD trajectories.
4. **Model Selection** -- Selects the best model checkpoint for inference.
5. **Agent (Inference)** -- Uses the trained model to identify outlier conformations and generate restart points for the next iteration.

## Directory Structure

```
ddmd_workflow/
├── run_workflow.py          # Entry point: sets up backend and runs the workflow
├── ddmd_workflow.py         # DDMdWorkflow class (extends DDSimManager)
├── config.py                # Pydantic configuration schema for YAML experiment files
├── env_setup.sh             # Creates conda environments for all stages
├── execute_workflow.sh      # Runs the workflow directly (no scheduler)
├── cpu_sbatch.sh            # SLURM sbatch script for CPU partition
├── gpu_sbatch.sh            # SLURM sbatch script for GPU partition
├── requirements.txt         # Python dependencies for this workflow
├── data/
│   ├── api.py               # Data API for managing experiment directory structure
│   ├── bba/                 # Example BBA protein input data (PDBs, weights)
│   └── lassen-keras-dbscan.yaml  # Example experiment configuration
├── sim/openmm/              # OpenMM simulation task
├── models/keras_cvae/       # Keras CVAE model (training + inference)
├── aggregation/basic/       # Basic aggregation task
├── selection/latest/        # Model selection task
├── agents/lof/              # Local Outlier Factor agent task
└── MD-tools_fix/            # Patches for the MD-tools dependency
```

## Environment Setup

The workflow requires three separate conda environments because the MD engine (OpenMM) and the ML framework (TensorFlow/Keras) have conflicting dependencies. The `env_setup.sh` script creates all three.

### Prerequisites

- Anaconda/Miniconda with `module load anaconda3` available
- GPU nodes with CUDA support (for OpenMM and TensorFlow)

### Running env_setup.sh

Before running, edit the paths at the top of `env_setup.sh` to match your system:

```bash
export BASE_DIR="/path/to/your/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/workflows/ddmd_workflow"
export CONDA_ENV="${WORK_DIR}/conda_env"
```

Then run:

```bash
bash env_setup.sh
```

This creates three environments under `$WORK_DIR/conda_env/`:

| Environment | Purpose | Key Packages |
|---|---|---|
| `deepdrivesim` | Base DDSim environment | DDSim (editable install), radical-asyncflow, ROSE, rhapsody |
| `conda-openmm` | MD simulations | OpenMM, CUDA toolkit, MD-tools |
| `conda-keras` | ML training & inference | TensorFlow, Keras, scikit-learn |

Each environment also gets DDSim installed in editable mode (`pip install -e .`) and the workflow-specific requirements (`requirements.txt`).

## Running the workflow

### 1. Configure the Experiment

The workflow is configured via a YAML file (see `data/lassen-keras-dbscan.yaml` for an example). Key settings:

- `experiment_directory` -- Output directory (must not exist; created automatically)
- `max_iteration` -- Number of simulation-training-inference cycles
- `molecular_dynamics_stage.num_tasks` -- Number of parallel MD simulations
- `molecular_dynamics_stage.task_config.initial_pdb_dir` -- Directory with input PDB files

The YAML file uses environment variable placeholders (`${EXPRMNT_DIR}`, `${CONDA_ENV}`, `${WORK_DIR}`) that are expanded at runtime by the launch scripts.

### 2. Launch with SLURM (sbatch)

For GPU nodes:

```bash
sbatch gpu_sbatch.sh
```

For CPU-only nodes:

```bash
sbatch cpu_sbatch.sh
```

Before submitting, edit the sbatch script to set:
- `#SBATCH -A` -- Your allocation/account
- `#SBATCH --mail-user` -- Your email address
- `BASE_DIR` -- Path to your DeepDriveSim checkout

The sbatch scripts will:
1. Clear any previous experiment directory
2. Activate the `deepdrivesim` conda environment
3. Substitute environment variables into a copy of the YAML config
4. Launch `run_workflow.py` with the processed config

### 3. Launch Directly (without SLURM)

For interactive or local testing:

```bash
bash execute_workflow.sh
```

This runs the same steps as the sbatch scripts but without the scheduler. Edit the paths at the top of the script before running.

### 4. Launch Manually

```bash
module load anaconda3
conda activate /path/to/conda_env/deepdrivesim
python run_workflow.py -c /path/to/config.yaml
```

## Execution Backends

The workflow engine supports two backends, configured in `run_workflow.py`:

- **ConcurrentExecutionBackend** (default) -- Runs tasks as local subprocesses. Good for testing and single-node runs.
- **DragonExecutionBackend** -- Uses the Dragon runtime for distributed HPC execution. Enable by passing `use_dragon=True`.

## Output Structure

The workflow creates the following directory tree under `experiment_directory`:

```
experiment_directory/
├── molecular_dynamics_runs/
│   └── stage0000/
│       ├── task0000/    # Per-simulation output (trajectories, contact maps)
│       ├── task0001/
│       └── ...
├── aggregation_runs/
├── machine_learning_runs/
│   └── stage0000/
│       └── task0000/    # Trained model weights, loss history
├── model_selection_runs/
├── agent_runs/
│   └── stage0000/
│       └── task0000/    # Outlier detection results, restart PDBs
└── lassen-keras-dbscan.yaml  # Backup of the experiment config
```
