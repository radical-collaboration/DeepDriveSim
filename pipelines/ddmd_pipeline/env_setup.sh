#!/bin/bash
set -euo pipefail


export BASE_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/ddmd_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"

mkdir $CONDA_ENV

module load anaconda3


##############################################
# 1. DeepDriveSim base env
##############################################
conda create -y -p $CONDA_ENV/deepdrivesim python=3.9
conda activate $CONDA_ENV/deepdrivesim
pip install --upgrade pip setuptools wheel
cd $BASE_DIR
pip install -e .
pip install -r "$WORK_DIR/requirements.txt"

conda deactivate

##############################################
# 2. OpenMM env
##############################################
conda create -y -p $CONDA_ENV/conda-openmm python=3.9
conda activate $CONDA_ENV/conda-openmm
conda install -y -c conda-forge "openmm>=8.0" "cudatoolkit=11.8"
pip install --upgrade pip setuptools wheel
cd $BASE_DIR
pip install -e .
pip install -r "$WORK_DIR/requirements.txt"
cd $WORK_DIR
git clone https://github.com/braceal/MD-tools.git
cp -r MD-tools_fix/* MD-tools
cd MD-tools
pip install -e .

conda deactivate

##############################################
# 3. Keras / TensorFlow env
##############################################
conda create -y -p $CONDA_ENV/conda-keras python=3.9
conda activate $CONDA_ENV/conda-keras
conda install -y scikit-learn
pip install --upgrade pip setuptools wheel
pip install tensorflow==2.20.0 pandas
pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
cd $BASE_DIR
pip install -e .
pip install -r "$WORK_DIR/requirements.txt"

conda deactivate
