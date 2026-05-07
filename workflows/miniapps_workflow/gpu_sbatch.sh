#!/bin/sh -l
  
#SBATCH -A *** 
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#xSBATCH --gpus=v100-32:4
#SBATCH --gpus=4
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name miniapp_gpu
#SBATCH --mail-user=mg2347@soe.rutgers.edu
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)


export BASE_DIR="${PROJECT}"
export WORK_DIR="${BASE_DIR}/DeepDriveSim/workflows/miniapps_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

unset SLURM_EXPORT_ENV
#module load anaconda
module load anaconda3
source activate base
conda activate $CONDA_ENV/miniapps

module load cuda/12.6.1
module load cudnn/8.0.4

cd  $WORK_DIR

rm telemetry-results/*
rm nvml-telemetry/*

dragon -s run_camp.py
#dragon -s run_camp.py
