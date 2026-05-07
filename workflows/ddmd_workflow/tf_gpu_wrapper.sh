#!/bin/bash
# gpu_wrapper.sh: GPU training wrapper for TF 2.16.1 + cuDNN 8.9.7.29 on V100.
#
# cuDNN 9.x on V100 (compute 7.0) fails with status 5003 in XLA's conv
# algorithm picker.  This wrapper uses cuDNN 8.9.7.29 (bundled via
# nvidia-cudnn-cu12==8.9.7.29) which is compatible with V100.
#
# Derives conda env prefix from $1 (the python executable path) and adds
# the bundled nvidia libs to LD_LIBRARY_PATH so TF can find cuDNN/cuBLAS.
#
# Usage: gpu_wrapper.sh <python-executable> [args...]

KERAS_PREFIX="$(dirname "$(dirname "$1")")"
SITE_NVIDIA="${KERAS_PREFIX}/lib/python3.10/site-packages/nvidia"

# CUDA toolkit and driver compat libs (must be set before referencing $CUDA_HOME)
export CUDA_HOME=/opt/packages/cuda/v12.6.1
# /usr/lib64 provides libcuda.so.1 (560.35.3 driver, matches kernel)
export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# conda-keras bundled nvidia libs (cuDNN 9.2.0, cuBLAS 12, CUDA runtime 12)
export LD_LIBRARY_PATH="${SITE_NVIDIA}/cudnn/lib:${SITE_NVIDIA}/cublas/lib:${SITE_NVIDIA}/cuda_runtime/lib:${LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_cuda_data_dir=${CUDA_HOME}"

exec "$@"
