#!/bin/bash
#
# MiniApps Workflow — SLURM batch script (Delta HPC / GPU)
#
# Set before calling sbatch:
#   export SBATCH_ACCOUNT=<project>-delta-gpu
#   export WORK_DIR=/work/nvme/bdyk/$USER
#
# Optional env overrides:
#   DDSIM_BACKEND=dragon   (default; set to "local" or "concurrent" for non-Dragon)
#
# Example:
#   sbatch delta_gpu_sbatch.sh
#
# Account: set SBATCH_ACCOUNT=<project>-delta-gpu before calling sbatch
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=miniapp_gpu
#SBATCH --mail-user=<your e-mail>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/miniapp_%j.out
#SBATCH --error=logs/miniapp_%j.err
# NOTE: logs/ must exist before sbatch is called.  Create it once with:
#   mkdir -p <miniapps_workflow_dir>/logs

set -euo pipefail

# Ensure Dragon infrastructure is cleaned up even if the run hangs or crashes
trap 'dragon-cleanup -s 2>/dev/null || true' EXIT

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ -z "${SBATCH_ACCOUNT:-}${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "WARNING: SBATCH_ACCOUNT is not set — job may be charged to default account."
fi
echo "Account  : ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Job ID   : ${SLURM_JOB_ID:-unknown}"
echo "Nodes    : ${SLURM_JOB_NUM_NODES:-?} (${SLURM_JOB_NODELIST:-?})"

if [ -z "${WORK_DIR:-}" ]; then
    echo "ERROR: WORK_DIR is not set."
    echo "       export WORK_DIR=<your_work_dir> && sbatch delta_gpu_sbatch.sh"
    exit 1
fi

# ── System library paths (Delta-specific, required by Dragon) ─────────────────
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
# MPI: cray-mpich names its library libmpi_gnu_112.so.12 (not libmpi.so.12).
# delta_env_setup.sh symlinks libmpi.so.12 → libmpi_gnu_112.so.12 inside the
# venv lib dir; add that dir to LD_LIBRARY_PATH so mpi4py finds the library.
export MPI_LIB=/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib
export FAB_LIB=/opt/cray/libfabric/1.22.0/lib64
MINIAPPS_VENV="${MINIAPPS_VENV:-${WORK_DIR}/ve/miniapps}"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MINIAPPS_VENV}/lib:${MPI_LIB}:${FAB_LIB}:${LD_LIBRARY_PATH:-}

# ── Environment ───────────────────────────────────────────────────────────────
DDSIM_DIR="${DDSIM_DIR:-${WORK_DIR}/DeepDriveSim}"
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
source "${MINIAPPS_VENV}/bin/activate"
dragon-config add --ofi-runtime-lib="${FAB_LIB}"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR="${DDSIM_DIR}/workflows/miniapps_workflow"
cd "${WORKDIR}"
mkdir -p logs

# workflows/ is not installed by pip (pyproject.toml ships ddsim* only);
# add the repo root so `from workflows.miniapps_workflow...` resolves.
export PYTHONPATH="${DDSIM_DIR}:${PYTHONPATH:-}"

echo ""
echo "Python   : $(python --version)"
echo "asyncflow: $(python -c 'import radical.asyncflow; print(radical.asyncflow.__version__)' 2>/dev/null || echo n/a)"
echo "rhapsody : $(python -c 'import rhapsody; print(rhapsody.__version__)' 2>/dev/null || echo n/a)"
echo "dragon   : $(python -c 'import dragon; print(dragon.__version__)' 2>/dev/null || echo n/a)"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
DDSIM_BACKEND="${DDSIM_BACKEND:-dragon}"
rm -f ddict_orc* asyncflow.session*.telemetry.jsonl 2>/dev/null || true

if [ "${DDSIM_BACKEND}" = "dragon" ]; then
    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        # dragon -m fails on Delta: gethostname() returns FQDNs but SLURM NodeName
        # is short (e.g. gpub081.delta.internal vs gpub081), making srun unsatisfiable.
        # Workaround: generate network config yaml, then launch via SSH + TCP transport.
        NET_DIR="/tmp/dragon_net_${SLURM_JOB_ID}"
        mkdir -p "${NET_DIR}"
        (cd "${NET_DIR}" && dragon-network-config --output-to-yaml --no-stdout)
        DRAGON_MODE="-w ssh --network-config ${NET_DIR}/slurm.yaml -t tcp"
    else
        DRAGON_MODE="-s"
    fi
    echo "Running: dragon ${DRAGON_MODE} run_workflow.py  (nodes=${SLURM_NNODES:-1})"
    dragon ${DRAGON_MODE} run_workflow.py
else
    echo "Running: python run_workflow.py  (backend=${DDSIM_BACKEND})"
    python run_workflow.py
fi

echo ""
echo "=== MiniApps workflow done: $(date) ==="
