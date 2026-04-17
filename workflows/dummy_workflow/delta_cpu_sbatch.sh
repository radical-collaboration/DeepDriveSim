#!/bin/sh -l

#SBATCH -A ***-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --job-name=dummy_cpu
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export DDSIM_DIR=/scratch/bblj/${USER}/DeepDriveSim
export WORK_DIR=${DDSIM_DIR}/workflows/dummy_workflow

cd ${WORK_DIR}

source /u/${USER}/ve/ddsim/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py
else
    dragon -s run_workflow.py
fi
