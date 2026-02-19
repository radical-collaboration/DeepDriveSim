#!/bin/sh -l
#SBATCH -A ***
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=***
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)


export BASE_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/ddmd_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"
export INPUT_DIR="${WORK_DIR}/data"

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

unset SLURM_EXPORT_ENV
module load anaconda3
source activate base
conda activate   $CONDA_ENV/deepdrivesim

cp  $INPUT_DIR/lassen-keras-dbscan.yaml $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml 
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${WORK_DIR}|$WORK_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml

python -m run_workflow -c $INPUT_DIR/new_lassen-keras-dbscan.yaml
