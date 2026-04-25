#!/bin/sh -l

#SBATCH -A ***-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=ddmd_gpu
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TF_FORCE_GPU_ALLOW_GROWTH=true

export HOME_DIR=/scratch/***/${USER}
export MD_HOME=${HOME_DIR}/DeepDriveSim/workflows/ddmd_workflow
export MD_INPUT=${MD_HOME}/data

# WARNING: this directory must be empty before running a new experiment!
export EXPRMNT_DIR=${MD_HOME}/ddmd_test_experiments
rm -rf ${EXPRMNT_DIR}

cd ${MD_HOME}

cp  ${MD_INPUT}/lassen-keras-dbscan.yaml ${MD_INPUT}/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|${EXPRMNT_DIR}|g" ${MD_INPUT}/new_lassen-keras-dbscan.yaml
sed -i "s|\${CONDA_ENV}|/u/${USER}/ve|g"  ${MD_INPUT}/new_lassen-keras-dbscan.yaml
sed -i "s|\${MD_HOME}|${MD_HOME}|g"       ${MD_INPUT}/new_lassen-keras-dbscan.yaml


source /u/${USER}/ve/ddmd/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py
else
    dragon -s run_workflow.py
fi
