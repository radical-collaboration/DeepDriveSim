#!/bin/bash
# tf_cpu_wrapper.sh: CPU inference wrapper for TF — forces CPU-only execution.
#
# Same LD_LIBRARY_PATH setup as tf_gpu_wrapper.sh so TF can load its shared
# libs, but sets CUDA_VISIBLE_DEVICES=-1 to prevent any GPU/cuDNN init.
# Use for tasks that hit cuDNN/V100 incompatibilities (e.g. lof.py agent stage).
#
# Usage: tf_cpu_wrapper.sh <python-executable> [args...]

KERAS_PREFIX="$(dirname "$(dirname "$1")")"
SITE_NVIDIA="${KERAS_PREFIX}/lib/python3.10/site-packages/nvidia"

# CUDA toolkit and driver compat libs
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# conda-keras bundled nvidia libs (cuDNN, cuBLAS, CUDA runtime)
export LD_LIBRARY_PATH="${SITE_NVIDIA}/cudnn/lib:${SITE_NVIDIA}/cublas/lib:${SITE_NVIDIA}/cuda_runtime/lib:${LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=-1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_cuda_data_dir=${CUDA_HOME}"

exec "$@"
