#!/bin/sh -l
  
#SBATCH -A dmr170002p 
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --export    NONE
#SBATCH --time=01:30:00
#SBATCH --job-name dragon
#SBATCH --mail-user=*** 
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

export BASE_DIR="${PROJECT}"
export WORK_DIR="${BASE_DIR}/DeepDriveSim/workflows/dummy_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

cd  $WORK_DIR

module load anaconda3
conda activate $CONDA_ENV/dragon_workflow 

#python run_workflow.py

dragon-network-config --output-to-yaml 

dragon -w ssh --network-config slurm.yaml run_workflow.py --use_dragon
