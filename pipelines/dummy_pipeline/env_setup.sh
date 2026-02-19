#!/bin/bash
export BASE_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/dummy_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"

mkdir $CONDA_ENV

module load anaconda3


##############################################
# 1. DeepDriveSim base env
##############################################
conda create -y -p $CONDA_ENV/dummy_pipeline python=3.9
conda activate $CONDA_ENV/dummy_pipeline
pip install --upgrade pip setuptools wheel
cd $BASE_DIR
pip install -e .
pip install -r "$WORK_DIR/requirements.txt"

conda deactivate
