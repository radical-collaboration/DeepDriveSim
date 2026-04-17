#!/bin/sh -l

#SBATCH -A ***-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=exchange_gpu
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DDSIM_DIR=/scratch/bblj/${USER}/DeepDriveSim
export VE_HOME=/u/${USER}/ve
export WORK_DIR=${DDSIM_DIR}/workflows/exchange_workflow

cd ${WORK_DIR}
rm -rf *telemetry*
rm -f ddict_*

source ${VE_HOME}/exchange/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py
else
    dragon -s run_workflow.py
fi

# --- Plot telemetry ---
#bash ${DDSIM_DIR}/workflows/plot_telemetry.sh telemetry-output/out.jsonl
#bash ${DDSIM_DIR}/workflows/plot_telemetry.sh telemetry-output/out.jsonl --out-dir /some/other/dir
