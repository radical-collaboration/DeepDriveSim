#!/bin/sh -l

#SBATCH -A *** 
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#xSBATCH --gpus=v100-32:8
#SBATCH --gpus=4
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=mg2347@soe.rutgers.edu
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export MD_HOME="${DDSim_DIR}/workflows/ddmd_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"
export INPUT_DIR="${MD_HOME}/data"

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$MD_HOME/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

unset SLURM_EXPORT_ENV
module load anaconda3
module load cuda/12.6.1
module load cudnn/8.0.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source activate base
conda activate   $CONDA_ENV/ddmd

cp  $INPUT_DIR/lassen-keras-dbscan.yaml $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml 
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${MD_HOME}|$MD_HOME|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml

rm *telemetry*/*

dragon -s run_camp.py
#python -m run_workflow -c $INPUT_DIR/new_lassen-keras-dbscan.yaml
