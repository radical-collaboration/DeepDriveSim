#!/bin/sh -l

#SBATCH -A bblj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --job-name=ala_dipep_exchng
#SBATCH --mail-user=rks174@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

export VE_HOME=~/ve
export DDSIM_DIR=/scratch/bblj/${USER}/DeepDriveSim
export WORK_DIR=${DDSIM_DIR}/workflows/exchange_workflow

cd ${WORK_DIR}

source /u/${USER}/ve/exchange/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py --config_file ala_dipep_config.yaml 
else
    dragon -s run_workflow.py --config_file ala_dipep_config.yaml
fi
