#!/bin/sh -l
  
#SBATCH -A *** 
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --export    NONE
#SBATCH --time=01:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=*** 
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

export BASE_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/dummy_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"

module load anaconda3
conda activate $CONDA_ENV/test_dragon 
dragon-network-config --output-to-yaml 

dragon -w ssh --network-config slurm.yaml run_dragon.py 
