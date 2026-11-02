#!/bin/bash
# Run miniapps_workflow locally (no SLURM, concurrent backend).
# Usage: bash run_local.sh [config.yaml]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DDSIM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -z "${VENV:-}" ]; then
    echo "ERROR: VENV is not set."
    echo "       export VENV=/path/to/your/virtualenv && bash run_local.sh"
    exit 1
fi
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: VENV does not look like a virtualenv: ${VENV}"
    echo "       Check that the path is correct and the venv was created."
    exit 1
fi

source "${VENV}/bin/activate"

export DDSIM_BACKEND=concurrent
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${DDSIM_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# CUDA libraries (required by CuPy on GPU nodes)
# Try known locations in preference order; warn if none found.
for CUDA_LIB in \
    /opt/nvidia/cuda-12.9/targets/x86_64-linux/lib \
    /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/12.9/targets/x86_64-linux/lib \
    /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/lib64 \
    /usr/local/cuda/lib64
do
    if [ -f "${CUDA_LIB}/libcublas.so.12" ]; then
        export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
        echo "CUDA lib  : ${CUDA_LIB}"
        break
    fi
done
if ! ldconfig -p 2>/dev/null | grep -q "libcublas.so.12" && \
   ! echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | xargs -I{} sh -c 'test -f "{}/libcublas.so.12"' 2>/dev/null; then
    echo "WARNING: libcublas.so.12 not found — CuPy GPU ops will fail"
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

CONFIG="${1:-config.yaml}"
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"

echo "DDSIM_DIR : ${DDSIM_DIR}"
echo "Python    : $(python --version)"
echo "Config    : ${CONFIG}"
echo "Log       : ${SCRIPT_DIR}/${LOG}"
echo ""

python run_workflow.py --config_file "${CONFIG}" 2>&1 | tee "${LOG}"
