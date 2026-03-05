#!/bin/bash

export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${DDSim_DIR}/workflows/dummy_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

#mkdir $CONDA_ENV

module load anaconda3
#module load anaconda


##############################################
# 1. DeepDriveSim base env
##############################################
conda create -y -p $CONDA_ENV/miniapp_workflow python=3.10
conda activate $CONDA_ENV/miniapp_workflow
pip install --upgrade pip setuptools wheel
cd $DDSim_DIR
pip install -e .
pip install -r "$WORK_DIR/requirements.txt"
cd $BASE_DIR
git clone -b Tutorial_reprod https://github.com/radical-cybertools/workflow-mini-apps.git
pip install -e workflow-mini-apps/wfMiniAPI
pip install -r "$WORK_DIR/requirements.txt"
conda deactivate
cd $WORK_DIR