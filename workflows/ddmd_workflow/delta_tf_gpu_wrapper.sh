#!/bin/bash
# delta_tf_gpu_wrapper.sh: GPU training wrapper for TF 2.16.1 + cuDNN 8.9.7.29 on A40.
#
# cuDNN 9.x can cause issues with XLA's conv algorithm picker.
# This wrapper uses cuDNN 8.9.7.29 (bundled via nvidia-cudnn-cu12==8.9.7.29)
# which is compatible with the A40 GPUs on Delta.
#
# Derives venv prefix from $1 (the python executable path) and adds
# the bundled nvidia libs to LD_LIBRARY_PATH so TF can find cuDNN/cuBLAS.
#
# Usage: delta_tf_gpu_wrapper.sh <python-executable> [args...]

VENV_PREFIX="$(dirname "$(dirname "$1")")"
SITE_NVIDIA="${VENV_PREFIX}/lib/python3.11/site-packages/nvidia"

# CUDA toolkit and driver compat libs
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# venv-bundled nvidia libs (cuDNN 8.9.7.29, cuBLAS, CUDA runtime)
export LD_LIBRARY_PATH="${SITE_NVIDIA}/cudnn/lib:${SITE_NVIDIA}/cublas/lib:${SITE_NVIDIA}/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Dragon sets CUDA_VISIBLE_DEVICES=-1 inside function tasks to prevent GPU stealing.
# For asyncio subprocesses (not Dragon tasks), set CUDA_VISIBLE_DEVICES to the
# Dragon-assigned GPU index if SPHERICAL_TRAINING_GPU is provided; otherwise unset.
if [ -n "${SPHERICAL_TRAINING_GPU+x}" ]; then
    export CUDA_VISIBLE_DEVICES="${SPHERICAL_TRAINING_GPU}"
else
    unset CUDA_VISIBLE_DEVICES
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_cuda_data_dir=${CUDA_HOME}"

exec "$@"
