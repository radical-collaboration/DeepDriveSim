#!/bin/sh -l
#SBATCH -A ***
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#xSBATCH --gpus-per-node=8
#SBATCH --export    NONE
#SBATCH --time=01:30:00
#SBATCH --job-name br_ddmd_cpu
#xSBATCH --mail-user=mg2347@soe.rutgers.edu
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${DDSim_DIR}/workflows/ddmd_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"
export INPUT_DIR="${WORK_DIR}/data"

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

unset SLURM_EXPORT_ENV
module load anaconda3
module load anaconda
source activate base
conda activate   $CONDA_ENV/deepdrivesim

cp  $INPUT_DIR/lassen-keras-dbscan.yaml $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml 
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${WORK_DIR}|$WORK_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml

python -m run_workflow -c $INPUT_DIR/new_lassen-keras-dbscan.yaml
