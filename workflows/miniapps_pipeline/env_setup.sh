#!/bin/bash
export BASE_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/miniapps_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"

mkdir $CONDA_ENV

module load anaconda3


##############################################
# 1. DeepDriveSim base env
##############################################
#conda create -y -p $CONDA_ENV/miniapp_pipeline python=3.9
conda activate $CONDA_ENV/miniapp_pipeline
pip install --upgrade pip setuptools wheel
cd $BASE_DIR
pip install -e .
pip install -r "$WORK_DIR/requirements.txt"
cd $WORK_DIR
git clone -b Tutorial_reprod https://github.com/radical-cybertools/workflow-mini-apps.git
pip install -e workflow-mini-apps/wfMiniAPI
pip install -r "$WORK_DIR/requirements.txt"
conda deactivate