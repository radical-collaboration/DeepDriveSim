#!/bin/bash

export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${DDSim_DIR}/workflows/miniapps_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

#mkdir $CONDA_ENV

module load anaconda3
#module load anaconda


##############################################
# 1. DeepDriveSim base env
##############################################
conda create -y -p $CONDA_ENV/miniapps_workflow python=3.10
conda activate $CONDA_ENV/miniapps_workflow
pip install --upgrade pip setuptools wheel
cd $DDSim_DIR
pip install -e .
cd $BASE_DIR
git clone -b Tutorial_reprod https://github.com/radical-cybertools/workflow-mini-apps.git
pip install -e workflow-mini-apps/wfMiniAPI
cd $WORK_DIR
python -m pip install -r "requirements.txt"
conda deactivate
