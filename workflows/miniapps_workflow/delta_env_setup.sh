#!/bin/bash
# =============================================================================
# MiniApps workflow environment setup — Delta HPC (NCSA)
#
# Creates a Python 3.11+ venv and installs all dependencies.
# Requires OpenMPI to be loaded before running (for mpi4py).
#
# Usage:
#   export WORK_DIR=/work/nvme/bdyk/$USER
#   module load openmpi
#   bash delta_env_setup.sh [--env-dir DIR] [--ddsim-dir DIR] [--base-dir DIR] [--python PATH]
#
# Defaults:
#   ENV_DIR   = $WORK_DIR/ve/miniapps
#   DDSIM_DIR = $WORK_DIR/DeepDriveSim
#   BASE_DIR  = $WORK_DIR
#   python    = auto-detected via `module load python` (Delta default: 3.13+)
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Initialize lmod (needed when run as non-interactive bash script) ──────────
if ! declare -f module &>/dev/null; then
    _lmod_init=/usr/share/lmod/lmod/init/bash
    [ -f "${_lmod_init}" ] && source "${_lmod_init}"
fi

# ── Require WORK_DIR ──────────────────────────────────────────────────────────
if [[ -z "${WORK_DIR:-}" ]]; then
    echo "ERROR: set WORK_DIR to your nvme work root, e.g.:"
    echo "  export WORK_DIR=/work/nvme/bdyk/\$USER"
    echo "  bash delta_env_setup.sh"
    exit 1
fi

# ── Defaults / arg parsing ────────────────────────────────────────────────────
ENV_DIR="${ENV_DIR:-${WORK_DIR}/ve/miniapps}"
DDSIM_DIR="${DDSIM_DIR:-${WORK_DIR}/DeepDriveSim}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}}"
BASE_PY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-dir)   ENV_DIR="$2";          shift 2 ;;
        --ddsim-dir) DDSIM_DIR="$2";        shift 2 ;;
        --base-dir)  BASE_DIR="$2";         shift 2 ;;
        --python)    BASE_PY_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

WF_DIR="${DDSIM_DIR}/workflows/miniapps_workflow"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  WORK_DIR  = ${WORK_DIR}"
echo "  ENV_DIR   = ${ENV_DIR}"
echo "  DDSIM_DIR = ${DDSIM_DIR}"
echo "  BASE_DIR  = ${BASE_DIR}"
echo "  WF_DIR    = ${WF_DIR}"
echo "================================================================="

# ── 1. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Creating venv ──"

if [ -n "${BASE_PY_OVERRIDE}" ]; then
    BASE_PY="${BASE_PY_OVERRIDE}"
    echo "Using Python override: ${BASE_PY}"
else
    # On Delta, `module load python` gives the default Python 3.13+.
    module load python 2>/dev/null || true
    BASE_PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
    if [ -z "${BASE_PY}" ]; then
        echo "ERROR: no Python found after 'module load python'."
        echo "       Pass an explicit interpreter: --python /path/to/python3"
        exit 1
    fi
    ver=$("${BASE_PY}" -c "import sys; v=sys.version_info; print(v.major*100+v.minor)")
    if [ "${ver}" -lt 311 ]; then
        echo "ERROR: ${BASE_PY} is Python ${ver} — need 3.11+."
        echo "       Pass an explicit interpreter: --python /path/to/python3.11"
        exit 1
    fi
fi
echo "Using Python: ${BASE_PY} ($(${BASE_PY} --version))"

if [ ! -x "${PY}" ]; then
    "${BASE_PY}" -m venv "${ENV_DIR}"
else
    echo "venv already exists at ${ENV_DIR}"
fi
echo "Python: $("${PY}" --version)"

# ── 2. Bootstrap pip ──────────────────────────────────────────────────────────
echo ""
echo "── Step 2: Bootstrapping pip ──"
"${PY}" -m pip install -q --upgrade pip wheel
"${PIP}" install -q --force-reinstall "setuptools<71"

# ── 3. Workflow requirements ──────────────────────────────────────────────────
echo ""
echo "── Step 3: Workflow requirements ──"
"${PIP}" install -q -r "${WF_DIR}/requirements.txt"

# ── 4. mpi4py (cray-mpich) ───────────────────────────────────────────────────
# mpi4py 4.x uses runtime ABI discovery (_mpiabi.py): it searches for libmpi.so
# in {venv}/lib/ before LD_LIBRARY_PATH.  Build from source so the extension
# links against cray-mpich, then symlink libmpi.so into the venv lib dir so
# the ABI loader finds it at import time without needing LD_LIBRARY_PATH set.
echo ""
echo "── Step 4: mpi4py (cray-mpich) ──"
if ! command -v mpicc &>/dev/null; then
    module load cray-mpich 2>/dev/null || true
fi
if command -v mpicc &>/dev/null; then
    # Extract the -L library dir from mpicc -show
    MPI_LIB_DIR=$(mpicc -show 2>/dev/null | grep -oP '(?<=-L)\S+' | head -1 || true)
    echo "  mpicc: $(which mpicc)"
    echo "  MPI_LIB_DIR: ${MPI_LIB_DIR}"

    # Build from source so mpi4py links against the loaded cray-mpich.
    MPICC="$(which mpicc)" "${PIP}" install -q --no-binary mpi4py mpi4py

    # Symlink MPI shared libs into the venv lib dir.
    if [ -n "${MPI_LIB_DIR}" ] && [ -d "${MPI_LIB_DIR}" ]; then
        [ -f "${MPI_LIB_DIR}/libmpi.so" ] && \
            ln -sf "${MPI_LIB_DIR}/libmpi.so" "${ENV_DIR}/lib/libmpi.so" 2>/dev/null || true
        for suffix in 12 40; do
            for src in "libmpi_gnu_112.so.${suffix}" "libmpi.so.${suffix}"; do
                if [ -f "${MPI_LIB_DIR}/${src}" ]; then
                    ln -sf "${MPI_LIB_DIR}/${src}" "${ENV_DIR}/lib/libmpi.so.${suffix}" 2>/dev/null || true
                    break
                fi
            done
        done
        echo "  libmpi.so.* symlinked from ${MPI_LIB_DIR} into ${ENV_DIR}/lib/"

        ACTIVATE="${ENV_DIR}/bin/activate"
        if ! grep -q "cray-mpich" "${ACTIVATE}" 2>/dev/null; then
            cat >> "${ACTIVATE}" << ACTIVATE_EOF

# ── Added by delta_env_setup.sh: cray-mpich for mpi4py ───────────────────────
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${MPI_LIB_DIR}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ACTIVATE_EOF
            echo "  LD_LIBRARY_PATH hook added to ${ACTIVATE}"
        else
            echo "  LD_LIBRARY_PATH hook already present in activate script"
        fi
    fi
else
    echo "WARNING: mpicc not found — load cray-mpich module first, then re-run."
    echo "  module load cray-mpich && bash delta_env_setup.sh"
fi

# ── 5. workflow-mini-apps ─────────────────────────────────────────────────────
echo ""
echo "── Step 5: workflow-mini-apps ──"
MINIAPPS_DIR="${BASE_DIR}/workflow-mini-apps"
if [ ! -d "${MINIAPPS_DIR}" ]; then
    echo "Cloning workflow-mini-apps → ${MINIAPPS_DIR}"
    git clone -b Tutorial_reprod \
        https://github.com/radical-cybertools/workflow-mini-apps.git \
        "${MINIAPPS_DIR}"
else
    echo "workflow-mini-apps already cloned at ${MINIAPPS_DIR}"
fi
"${PIP}" install -q -e "${MINIAPPS_DIR}/wfMiniAPI"

# ── 6. DeepDriveSim + Dragon (editable) ───────────────────────────────────────
echo ""
echo "── Step 6: DeepDriveSim [dragon] (editable) ──"
"${PIP}" install -q -e "${DDSIM_DIR}[dragon]"

# ── 7. radical.asyncflow + rhapsody ──────────────────────────────────────────
echo ""
echo "── Step 7: radical.asyncflow + rhapsody ──"
"${PIP}" install -q "radical.asyncflow>=0.3.0"
"${PIP}" install -q "rhapsody-py[dragon,telemetry]"
"${PIP}" install -q matplotlib

# ── 8. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "── Verifying installation ──"
_check() {
    local label="$1"; shift
    if out=$("$@" 2>&1); then
        echo "  ${label}: OK  (${out})"
    else
        echo "  WARNING: ${label} failed"
        echo "    ${out}" | head -3
    fi
}

_check "cupy"              "${PY}" -c "import cupy; print(cupy.__version__)"
_check "h5py"              "${PY}" -c "import h5py; print(h5py.__version__)"
_check "mpi4py"            "${PY}" -c "import mpi4py; print(mpi4py.__version__)"
_check "radical.asyncflow" "${PY}" -c "import radical.asyncflow; print('ok')"
_check "rhapsody"          "${PY}" -c "import rhapsody; print('ok')"
_check "dragonhpc"         "${PY}" -c "import dragon; print('ok')" 2>/dev/null || echo "  dragonhpc: install separately if needed"

echo ""
echo "================================================================="
echo "Setup complete."
echo ""
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo ""
echo "Run the workflow:"
echo "  export WORK_DIR=${WORK_DIR}"
echo "  export SBATCH_ACCOUNT=<project>-delta-gpu"
echo "  cd ${WF_DIR}"
echo "  sbatch delta_gpu_sbatch.sh"
echo "================================================================="
