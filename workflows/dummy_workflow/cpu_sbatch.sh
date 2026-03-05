#!/bin/sh -l
  
#SBATCH -A dmr170002p 
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name dummy
#xSBATCH --mail-user=***
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)


export BASE_DIR="${PROJECT}"
export WORK_DIR="${BASE_DIR}/DeepDriveSim/workflows/dummy_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

module load anaconda3
conda activate $CONDA_ENV/dummy_workflow

cd $WORK_DIR

python run_workflow.py