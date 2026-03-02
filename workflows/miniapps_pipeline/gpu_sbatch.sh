#!/bin/sh -l
  
#SBATCH -A dmr170002p 
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

export BASE_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/miniapps_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"

unset SLURM_EXPORT_ENV
module load anaconda
module load anaconda3
module load cuda/12.6.1
module load cudnn/8.0.4
module load openmpi/5.0.8-gcc13.3.1
source activate base
conda activate   $CONDA_ENV/miniapp_pipeline

python run_miniapps_pipeline.py
