#!/bin/sh -l

#SBATCH -A ***-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=miniapp_gpu
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
# MPI: cray-mpich names its library libmpi_gnu_112.so.12 (not libmpi.so.12).
# delta_env_setup.sh symlinks libmpi.so.12 → libmpi_gnu_112.so.12 inside the
# venv lib dir; add that dir to LD_LIBRARY_PATH so the pre-built mpi4py wheel
# finds the library at import time via the dynamic linker.
export MPI_LIB=/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${VE_HOME}/miniapps/lib:$MPI_LIB:$LD_LIBRARY_PATH

export DDSIM_DIR=/scratch/bblj/${USER}/DeepDriveSim
export VE_HOME=~/ve
export WORK_DIR=${DDSIM_DIR}/workflows/miniapps_workflow

cd ${WORK_DIR}

source ${VE_HOME}/miniapps/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py
else
    dragon -s run_workflow.py
fi