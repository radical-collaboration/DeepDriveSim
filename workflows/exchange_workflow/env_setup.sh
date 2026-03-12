#!/bin/bash
export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${DDSim_DIR}/workflows/exchange_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

#mkdir $CONDA_ENV

module load anaconda3
#module load anaconda


##############################################
# 1. Dummy workflow env
##############################################
conda create -y -p $CONDA_ENV/exchange_workflow python=3.10
conda activate $CONDA_ENV/exchange_workflow
pip install --upgrade pip setuptools wheel
cd $DDSim_DIR
pip install -e .
cd $WORK_DIR
pip install -r "requirements.txt"
conda deactivate


##############################################
# 2. Dragon - Dummy workflow  env
##############################################
conda create -y -p $CONDA_ENV/dragon_exchange python=3.10
conda activate $CONDA_ENV/dragon_exchange
pip install --upgrade pip setuptools wheel
cd $WORK_DIR
pip install -r "requirements.txt"
cd $DDSim_DIR
pip install -e ".[dragon]"
conda deactivate
cd $WORK_DIR