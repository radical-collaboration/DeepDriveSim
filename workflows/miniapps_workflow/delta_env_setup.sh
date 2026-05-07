#!/bin/bash
# =============================================================================
# MiniApps workflow environment setup — Delta HPC (NCSA)
#
# Creates a Python 3.11 venv and installs all dependencies.
# Requires OpenMPI to be loaded before running (for mpi4py).
#
# Usage:
#   module load openmpi
#   bash delta_env_setup.sh [--env-dir DIR] [--ddsim-dir DIR] [--base-dir DIR]
#
# Defaults:
#   ENV_DIR  = /u/$USER/ve/miniapps
#   DDSIM_DIR = /scratch/bblj/$USER/DeepDriveSim
#   BASE_DIR  = /scratch/bblj/$USER
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Parse optional overrides ──────────────────────────────────────────────────
ENV_DIR="${ENV_DIR:-/u/${USER}/ve/miniapps}"
DDSIM_DIR="${DDSIM_DIR:-/scratch/bblj/${USER}/DeepDriveSim}"
BASE_DIR="${BASE_DIR:-/scratch/bblj/${USER}}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-dir)   ENV_DIR="$2";   shift 2 ;;
        --ddsim-dir) DDSIM_DIR="$2"; shift 2 ;;
        --base-dir)  BASE_DIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

WORK_DIR="${DDSIM_DIR}/workflows/miniapps_workflow"
PY="${ENV_DIR}/bin/python3.11"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  ENV_DIR   = ${ENV_DIR}"
echo "  DDSIM_DIR = ${DDSIM_DIR}"
echo "  BASE_DIR  = ${BASE_DIR}"
echo "  WORK_DIR  = ${WORK_DIR}"
echo "================================================================="

# ── 0. Ensure 'module' is available (needed when run as bash script.sh) ───────
if ! command -v module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi

# ── 1. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Creating venv ──"

# Find a Python 3.11 interpreter.  On Delta, 'module load anaconda3' exposes
# 'python3' (not 'python3.11'), so we try the versioned name first, then load
# the module and fall back to any python3 that reports version 3.11.x.
_find_python311() {
    for candidate in python3.11 python3 python; do
        local p
        p=$(command -v "${candidate}" 2>/dev/null) || continue
        local ver
        ver=$("${p}" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null) || continue
        [ "${ver}" = "3.11" ] && echo "${p}" && return 0
    done
    return 1
}

BASE_PY=$(_find_python311 || true)
if [ -z "${BASE_PY}" ]; then
    echo "python3.11 not in PATH — loading cray-python/3.11.7 module..."
    module load cray-python/3.11.7 2>/dev/null || true
    BASE_PY=$(_find_python311 || true)
fi
if [ -z "${BASE_PY}" ]; then
    echo "ERROR: no Python 3.11 interpreter found."
    echo "       Try: module load cray-python/3.11.7"
    exit 1
fi
echo "Using Python: ${BASE_PY} ($(${BASE_PY} --version))"

if [ ! -x "${PY}" ]; then
    "${BASE_PY}" -m venv "${ENV_DIR}"
else
    echo "venv already exists at ${ENV_DIR}"
fi

# Ensure python3.11 exists in the venv regardless of base interpreter name.
if [ ! -x "${PY}" ]; then
    ln -sf "${BASE_PY}" "${ENV_DIR}/bin/python3.11"
fi
ln -sf "${ENV_DIR}/bin/python3.11" "${ENV_DIR}/bin/python"  2>/dev/null || true
ln -sf "${ENV_DIR}/bin/python3.11" "${ENV_DIR}/bin/python3" 2>/dev/null || true

echo "Python: $("${PY}" --version)"

# ── 2. Bootstrap pip ──────────────────────────────────────────────────────────
echo ""
echo "── Step 2: Bootstrapping pip ──"
"${PY}" -m pip install -q --upgrade pip wheel
"${PIP}" install -q --force-reinstall "setuptools<71"

# ── 3. Workflow requirements ──────────────────────────────────────────────────
echo ""
echo "── Step 3: Workflow requirements ──"
"${PIP}" install -q -r "${WORK_DIR}/requirements.txt"

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
    # mpi4py 4.x (_mpiabi.py) and the dynamic linker both search {venv}/lib/
    # when it is in LD_LIBRARY_PATH (see delta_gpu_sbatch.sh).
    # cray-mpich uses libmpi_gnu_112.so.12 — create generic libmpi.so.* aliases
    # so the pre-built MPICH wheel variant can find them.
    if [ -n "${MPI_LIB_DIR}" ] && [ -d "${MPI_LIB_DIR}" ]; then
        # Generic unversioned alias (for _mpiabi.py discovery)
        [ -f "${MPI_LIB_DIR}/libmpi.so" ] && \
            ln -sf "${MPI_LIB_DIR}/libmpi.so" "${ENV_DIR}/lib/libmpi.so" 2>/dev/null || true
        # Versioned aliases that the pre-built mpi4py MPICH wheel links against
        for suffix in 12 40; do
            # Try the cray vendor name first, then the generic name
            for src in "libmpi_gnu_112.so.${suffix}" "libmpi.so.${suffix}"; do
                if [ -f "${MPI_LIB_DIR}/${src}" ]; then
                    ln -sf "${MPI_LIB_DIR}/${src}" "${ENV_DIR}/lib/libmpi.so.${suffix}" 2>/dev/null || true
                    break
                fi
            done
        done
        echo "  libmpi.so.* symlinked from ${MPI_LIB_DIR} into ${ENV_DIR}/lib/"

        # Inject MPI lib paths into the venv activate script so LD_LIBRARY_PATH
        # is set automatically for both sbatch jobs and interactive use.
        ACTIVATE="${ENV_DIR}/bin/activate"
        if ! grep -q "cray-mpich" "${ACTIVATE}" 2>/dev/null; then
            cat >> "${ACTIVATE}" << ACTIVATE_EOF

# ── Added by delta_env_setup.sh: cray-mpich for mpi4py ───────────────────────
# ${ENV_DIR}/lib/ holds libmpi.so.12 → libmpi_gnu_112.so.12 symlinks so the
# dynamic linker can satisfy mpi4py's NEEDED entry without LD_LIBRARY_PATH tricks.
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

# ── 7. radical.asyncflow ──────────────────────────────────────────────────────
echo ""
echo "── Step 7: radical.asyncflow ──"
"${PIP}" install -q "radical.asyncflow>=0.3.0"
"${PIP}" install -q matplotlib

# ── 8. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "── Verifying installation ──"
_check() {
    local label="$1"; shift
    if out=$("$@" 2>&1); then
        echo "${label}: OK  (${out})"
    else
        echo "WARNING: ${label} failed"
        echo "  ${out}" | head -3
    fi
}

_check "cupy"              "${PY}" -c "import cupy; print(cupy.__version__)"
_check "h5py"              "${PY}" -c "import h5py; print(h5py.__version__)"
_check "mpi4py"            "${PY}" -c "import mpi4py; print(mpi4py.__version__)"
_check "radical.asyncflow" "${PY}" -c "import radical.asyncflow; print('ok')"
_check "rhapsody"          "${PY}" -c "import rhapsody; print('ok')"
_check "dragonhpc"         "${PY}" -c "import dragon; print('ok')" 2>/dev/null || echo "dragonhpc: install separately if needed"

echo ""
echo "================================================================="
echo "Setup complete."
echo ""
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo "================================================================="
