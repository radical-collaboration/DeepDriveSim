#!/bin/bash
# =============================================================================
# Exchange workflow environment setup — Delta HPC (NCSA)
#
# Creates a Python 3.11+ venv and installs all dependencies.
#
# radical.asyncflow and rhapsody are installed from specific development branches:
#   AsyncFlow : https://github.com/radical-cybertools/radical.asyncflow  (prototype/telemetry)
#   Rhapsody  : https://github.com/radical-cybertools/rhapsody            (feature/telemetry)
#
# Usage:
#   export WORK_DIR=/work/nvme/bdyk/$USER
#   bash delta_env_setup.sh [--env-dir DIR] [--ddsim-dir DIR] [--base-dir DIR] [--python PATH]
#
# Defaults:
#   ENV_DIR   = $WORK_DIR/ve/exchange
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
ENV_DIR="${ENV_DIR:-${WORK_DIR}/ve/exchange}"
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

WF_DIR="${DDSIM_DIR}/workflows/exchange_workflow"
ASYNCFLOW_DIR="${BASE_DIR}/radical.asyncflow"
RHAPSODY_DIR="${BASE_DIR}/rhapsody"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  WORK_DIR      = ${WORK_DIR}"
echo "  ENV_DIR       = ${ENV_DIR}"
echo "  DDSIM_DIR     = ${DDSIM_DIR}"
echo "  BASE_DIR      = ${BASE_DIR}"
echo "  ASYNCFLOW_DIR = ${ASYNCFLOW_DIR}"
echo "  RHAPSODY_DIR  = ${RHAPSODY_DIR}"
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
    echo "  venv already exists at ${ENV_DIR}"
fi
echo "  Python: $("${PY}" --version)"

# ── 2. Bootstrap pip ──────────────────────────────────────────────────────────
echo ""
echo "── Step 2: Bootstrapping pip ──"
"${PY}" -m pip install -q --upgrade pip wheel
"${PIP}" install -q --force-reinstall "setuptools<71"

# ── 3. Workflow requirements ──────────────────────────────────────────────────
echo ""
echo "── Step 3: Workflow requirements ──"
"${PIP}" install -q -r "${WF_DIR}/requirements.txt"

# ── 4. DeepDriveSim + Dragon (editable) ──────────────────────────────────────
echo ""
echo "── Step 4: DeepDriveSim [dragon] (editable) ──"
"${PIP}" install -q -e "${DDSIM_DIR}[dragon]"

# ── 5. Clone / update radical.asyncflow (prototype/telemetry) ────────────────
echo ""
echo "── Step 5: radical.asyncflow ──"
if [ ! -d "${ASYNCFLOW_DIR}/.git" ]; then
    echo "  Cloning radical.asyncflow (prototype/telemetry) → ${ASYNCFLOW_DIR}"
    git clone -b prototype/telemetry \
        https://github.com/radical-cybertools/radical.asyncflow.git \
        "${ASYNCFLOW_DIR}"
else
    echo "  Updating radical.asyncflow..."
    git -C "${ASYNCFLOW_DIR}" pull --ff-only
fi
"${PIP}" install -q -e "${ASYNCFLOW_DIR}"

# ── 6. Clone / update rhapsody (feature/telemetry) ───────────────────────────
echo ""
echo "── Step 6: rhapsody ──"
if [ ! -d "${RHAPSODY_DIR}/.git" ]; then
    echo "  Cloning rhapsody (feature/telemetry) → ${RHAPSODY_DIR}"
    git clone -b feature/telemetry \
        https://github.com/radical-cybertools/rhapsody.git \
        "${RHAPSODY_DIR}"
else
    echo "  Updating rhapsody..."
    git -C "${RHAPSODY_DIR}" pull --ff-only
fi
"${PIP}" install -q -e "${RHAPSODY_DIR}[dragon,telemetry]"

# ── 7. Re-pin critical versions ───────────────────────────────────────────────
echo ""
echo "── Step 7: Re-pinning critical versions ──"
"${PIP}" install -q --force-reinstall \
    "protobuf>=3.20.3,<5.0.0dev" \
    "setuptools<71"

# ── 8. OpenMM + matplotlib ───────────────────────────────────────────────────
echo ""
echo "── Step 8: OpenMM + matplotlib ──"
"${PIP}" install -q "openmm>=8.0"
"${PIP}" install -q matplotlib

# ── 9. Verify ────────────────────────────────────────────────────────────────
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
_check "openmm"            "${PY}" -c "import openmm; print(openmm.__version__)"
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
echo ""
echo "Run the workflow:"
echo "  export WORK_DIR=${WORK_DIR}"
echo "  export SBATCH_ACCOUNT=<project>-delta-gpu"
echo "  cd ${WF_DIR}"
echo "  sbatch delta_gpu_sbatch.sh"
echo "================================================================="
