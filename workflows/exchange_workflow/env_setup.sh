#!/bin/bash
# =============================================================================
# Exchange workflow environment setup — Delta HPC (NCSA)
#
# Creates a Python 3.11 venv and installs all dependencies.
#
# Usage:
#   bash delta_env_setup.sh [--env-dir DIR] [--ddsim-dir DIR] [--base-dir DIR]
#
# Defaults:
#   ENV_DIR   = /u/$USER/ve/exchange
#   DDSIM_DIR = /scratch/bblj/$USER/DeepDriveSim
#   BASE_DIR  = /scratch/bblj/$USER
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Parse optional overrides ──────────────────────────────────────────────────
ENV_DIR="${ENV_DIR:-/u/${USER}/ve/exchange}"
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

WORK_DIR="${DDSIM_DIR}/workflows/exchange_workflow"
ASYNCFLOW_DIR="${BASE_DIR}/radical.asyncflow"
RHAPSODY_DIR="${BASE_DIR}/rhapsody"
PY="${ENV_DIR}/bin/python3.11"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  ENV_DIR       = ${ENV_DIR}"
echo "  DDSIM_DIR     = ${DDSIM_DIR}"
echo "  BASE_DIR      = ${BASE_DIR}"
echo "  ASYNCFLOW_DIR = ${ASYNCFLOW_DIR}"
echo "  RHAPSODY_DIR  = ${RHAPSODY_DIR}"
echo "================================================================="

# ── 0. Ensure 'module' is available ──────────────────────────────────────────
if ! command -v module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi

# ── 1. Find Python 3.11 ───────────────────────────────────────────────────────
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

# ── 2. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: Creating venv ──"
if [ ! -x "${PY}" ]; then
    "${BASE_PY}" -m venv "${ENV_DIR}"
else
    echo "  venv already exists at ${ENV_DIR}"
fi
if [ ! -x "${PY}" ]; then
    ln -sf "${BASE_PY}" "${PY}"
fi
ln -sf "${PY}" "${ENV_DIR}/bin/python"  2>/dev/null || true
ln -sf "${PY}" "${ENV_DIR}/bin/python3" 2>/dev/null || true
echo "  Python: $("${PY}" --version)"

# ── 3. Bootstrap pip ──────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Bootstrapping pip ──"
"${PY}" -m pip install -q --upgrade pip wheel
"${PIP}" install -q --force-reinstall "setuptools<71"

# ── 4. Workflow requirements ──────────────────────────────────────────────────
echo ""
echo "── Step 4: Workflow requirements ──"
"${PIP}" install -q -r "${WORK_DIR}/requirements.txt"

# ── 5. DeepDriveSim + Dragon (editable) ──────────────────────────────────────
echo ""
echo "── Step 5: DeepDriveSim [dragon] (editable) ──"
"${PIP}" install -q -e "${DDSIM_DIR}[dragon]"

# ── 6. Clone / update radical.asyncflow (prototype/telemetry) ────────────────
echo ""
echo "── Step 6: radical.asyncflow ──"
if [ ! -d "${ASYNCFLOW_DIR}/.git" ]; then
    echo "  Cloning radical.asyncflow (prototype/telemetry) → ${ASYNCFLOW_DIR}"
    git clone -b prototype/telemetry \
        https://github.com/radical-cybertools/radical.asyncflow.git \
        "${ASYNCFLOW_DIR}"
else
    echo "  Updating radical.asyncflow..."
    git -C "${ASYNCFLOW_DIR}" pull --ff-only
fi
"${PIP}" install -q -e "${ASYNCFLOW_DIR}[telemetry]"

# ── 7. Clone / update rhapsody (feature/telemetry) ───────────────────────────
echo ""
echo "── Step 7: rhapsody ──"
if [ ! -d "${RHAPSODY_DIR}/.git" ]; then
    echo "  Cloning rhapsody (feature/telemetry) → ${RHAPSODY_DIR}"
    git clone -b feature/telemetry \
        https://github.com/radical-cybertools/rhapsody.git \
        "${RHAPSODY_DIR}"
else
    echo "  Updating rhapsody..."
    git -C "${RHAPSODY_DIR}" pull --ff-only
fi
"${PIP}" install -q -e "${RHAPSODY_DIR}[dragon]"
"${PIP}" install -q -e "${RHAPSODY_DIR}[telemetry]"

# ── 8. Re-pin critical versions ───────────────────────────────────────────────
echo ""
echo "── Step 8: Re-pinning critical versions ──"
"${PIP}" install -q --force-reinstall \
    "protobuf>=3.20.3,<5.0.0dev" \
    "setuptools<71"

# ── 9. Install OpenMM ───────────────────────────────────────────────
echo "  Installing OpenMM..."
"${PIP}" install -q "openmm>=8.0"

# ── 10. Verify ─────────────────────────────────────────────────────────────────
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

_check "pyyaml"            "${PY}" -c "import yaml; print(yaml.__version__)"
_check "radical.asyncflow" "${PY}" -c "import radical.asyncflow; print('ok')"
_check "rhapsody"          "${PY}" -c "import rhapsody; print('ok')"
_check "matplotlib"        "${PY}" -c "import matplotlib; print(matplotlib.__version__)"
_check "dragonhpc"         "${PY}" -c "import dragon; print('ok')"

echo ""
echo "================================================================="
echo "Setup complete."
echo ""
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo "================================================================="
