#!/bin/bash
set -euo pipefail

export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${DDSim_DIR}/workflows/ddmd_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"

#mkdir $CONDA_ENV

module load anaconda3
#module load anaconda


# ##############################################
# # 1. DeepDriveSim base env
# ##############################################
# conda create -y -p $CONDA_ENV/deepdrivesim python=3.10
# conda activate $CONDA_ENV/deepdrivesim
# pip install --upgrade pip setuptools wheel
# cd $DDSim_DIR
# pip install -e ".[dev,dragon]"
# pip install -r "$WORK_DIR/requirements.txt"

# conda deactivate

# ##############################################
# # 2. OpenMM env
# ##############################################
# conda create -y -p $CONDA_ENV/conda-openmm python=3.10
# conda activate $CONDA_ENV/conda-openmm
# conda install -y -c conda-forge "openmm>=8.0" "cudatoolkit=11.8"
# pip install --upgrade pip setuptools wheel
# cd $DDSim_DIR
# pip install -e ".[dev,dragon]"
# pip install -r "$WORK_DIR/requirements.txt"
# cd $BASE_DIR
# if [ ! -d "$BASE_DIR/MD-tools" ]; then
#     git clone https://github.com/braceal/MD-tools.git
# fi
# cp -r $WORK_DIR/MD-tools_fix/* MD-tools
# cd MD-tools
# pip install -e .

# conda deactivate

##############################################
# 3. Keras / TensorFlow env (cuDNN 8.9.7.29 for V100 compatibility)
##############################################
conda create -y -p $CONDA_ENV/conda-keras python=3.10
conda activate $CONDA_ENV/conda-keras
pip install --upgrade pip setuptools wheel
pip install "tensorflow[and-cuda]==2.16.1" "nvidia-cudnn-cu12==8.9.7.29" "keras==3.0.5"
cd $DDSim_DIR
pip install -e ".[dev,dragon]"
pip install -r "$WORK_DIR/requirements.txt"

conda deactivate

cd $WORK_DIR
