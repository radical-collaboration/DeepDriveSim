#!/bin/sh -l

#SBATCH -A ***-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --job-name=ddmd_cpu
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export HOME_DIR=/scratch/***/${USER}
export MD_HOME=${HOME_DIR}/DeepDriveSim/workflows/ddmd_workflow
export INPUT_DIR=${MD_HOME}/data

# WARNING: this directory must be empty before running a new experiment!
export EXPRMNT_DIR=${MD_HOME}/ddmd_test_experiments
rm -rf ${EXPRMNT_DIR}

cd ${MD_HOME}

cp  ${INPUT_DIR}/lassen-keras-dbscan.yaml ${INPUT_DIR}/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|${EXPRMNT_DIR}|g" ${INPUT_DIR}/new_lassen-keras-dbscan.yaml
sed -i "s|\${CONDA_ENV}|/u/${USER}/ve|g"  ${INPUT_DIR}/new_lassen-keras-dbscan.yaml
sed -i "s|\${MD_HOME}|${MD_HOME}|g"       ${INPUT_DIR}/new_lassen-keras-dbscan.yaml


source /u/${USER}/ve/ddmd/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py
else
    dragon -s run_workflow.py
fi
