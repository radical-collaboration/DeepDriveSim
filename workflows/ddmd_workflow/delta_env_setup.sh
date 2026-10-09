#!/bin/bash
# =============================================================================
# DDMd workflow environment setup — Delta HPC (NCSA)
#
# Creates three Python 3.11 venvs:
#   1. ve/ddmd        — DeepDriveSim + Dragon  (orchestration, simulation, selection, agent)
#   2. ve/ddmd-openmm — OpenMM + MD-tools       (GPU-accelerated MD)
#   3. ve/ddmd-keras  — TensorFlow/Keras         (training / inference)
#
# radical.asyncflow and rhapsody are installed from specific development branches:
#   AsyncFlow : https://github.com/radical-cybertools/radical.asyncflow  (prototype/telemetry)
#   Rhapsody  : https://github.com/radical-cybertools/rhapsody            (feature/telemetry)
#               installed as: pip install .[dragon]  then  pip install .[telemetry]
#
# Usage:
#   export WORK_DIR=/work/nvme/bdyk/$USER
#   bash delta_env_setup.sh [--ve-home DIR] [--ddsim-dir DIR] [--base-dir DIR] [--python PATH]
#
# Defaults:
#   VE_HOME   = $WORK_DIR/ve
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

# ── Parse optional overrides ──────────────────────────────────────────────────
VE_HOME="${VE_HOME:-${WORK_DIR}/ve}"
DDSIM_DIR="${DDSIM_DIR:-${WORK_DIR}/DeepDriveSim}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}}"
BASE_PY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ve-home)   VE_HOME="$2";          shift 2 ;;
        --ddsim-dir) DDSIM_DIR="$2";        shift 2 ;;
        --base-dir)  BASE_DIR="$2";         shift 2 ;;
        --python)    BASE_PY_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

WF_DIR="${DDSIM_DIR}/workflows/ddmd_workflow"
ASYNCFLOW_DIR="${BASE_DIR}/radical.asyncflow"
RHAPSODY_DIR="${BASE_DIR}/rhapsody"

echo "================================================================="
echo "  WORK_DIR      = ${WORK_DIR}"
echo "  VE_HOME       = ${VE_HOME}"
echo "  DDSIM_DIR     = ${DDSIM_DIR}"
echo "  BASE_DIR      = ${BASE_DIR}"
echo "  WF_DIR        = ${WF_DIR}"
echo "  ASYNCFLOW_DIR = ${ASYNCFLOW_DIR}"
echo "  RHAPSODY_DIR  = ${RHAPSODY_DIR}"
echo "================================================================="

# ── Find Python 3.11+ ─────────────────────────────────────────────────────────
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

# ── Helper: create a venv and bootstrap pip ───────────────────────────────────
_make_venv() {
    local env_dir="$1"
    local py="${env_dir}/bin/python"
    local pip="${env_dir}/bin/pip"

    echo ""
    echo "── Creating venv: ${env_dir} ──"
    if [ ! -x "${py}" ]; then
        "${BASE_PY}" -m venv "${env_dir}"
    else
        echo "  venv already exists at ${env_dir}"
    fi
    echo "  Python: $("${py}" --version)"

    echo "  Bootstrapping pip..."
    "${py}" -m pip install -q --upgrade pip wheel
    "${pip}" install -q --force-reinstall "setuptools<71"
}

# ── Helper: verify a package import inside a venv ────────────────────────────
_check() {
    local py="$1"; local label="$2"; shift 2
    if out=$("${py}" -c "$@" 2>&1); then
        echo "  ${label}: OK  (${out})"
    else
        echo "  WARNING: ${label} failed — ${out}" | head -2
    fi
}

# ── Helper: install radical.asyncflow from the prototype/telemetry branch ────
_install_asyncflow() {
    local pip="$1"
    echo "  Installing radical.asyncflow (prototype/telemetry)..."
    "${pip}" install -q -e "${ASYNCFLOW_DIR}"
}

# ── Helper: install rhapsody from the feature/telemetry branch ───────────────
# Must be installed in two passes: .[dragon] first, then .[telemetry].
_install_rhapsody() {
    local pip="$1"
    echo "  Installing rhapsody [dragon,telemetry] (feature/telemetry)..."
    "${pip}" install -q -e "${RHAPSODY_DIR}[dragon,telemetry]"
}

# ── Clone / update dependency repos ──────────────────────────────────────────
echo ""
echo "── Cloning/updating dependency repos ──"

if [ ! -d "${ASYNCFLOW_DIR}/.git" ]; then
    echo "  Cloning radical.asyncflow (prototype/telemetry) → ${ASYNCFLOW_DIR}"
    git clone -b prototype/telemetry \
        https://github.com/radical-cybertools/radical.asyncflow.git \
        "${ASYNCFLOW_DIR}"
else
    echo "  Updating radical.asyncflow..."
    git -C "${ASYNCFLOW_DIR}" pull --ff-only
fi

if [ ! -d "${RHAPSODY_DIR}/.git" ]; then
    echo "  Cloning rhapsody (feature/telemetry) → ${RHAPSODY_DIR}"
    git clone -b feature/telemetry \
        https://github.com/radical-cybertools/rhapsody.git \
        "${RHAPSODY_DIR}"
else
    echo "  Updating rhapsody..."
    git -C "${RHAPSODY_DIR}" pull --ff-only
fi


# ══════════════════════════════════════════════════════════════════════════════
# 1. ve/ddmd — main workflow env (DeepDriveSim + Dragon)
#    Used by: workflow orchestrator (dragon), simulation cmd, model selection, agent.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══ Env 1/3: ve/ddmd ══════════════════════════════════════════════"
ENV1="${VE_HOME}/ddmd"
PY1="${ENV1}/bin/python"
PIP1="${ENV1}/bin/pip"

_make_venv "${ENV1}"

echo "  Installing workflow requirements..."
"${PIP1}" install -q -r "${WF_DIR}/requirements.txt"

echo "  Installing DeepDriveSim [dev,dragon] (editable)..."
"${PIP1}" install -q -e "${DDSIM_DIR}[dev,dragon]"

_install_asyncflow "${PIP1}"
_install_rhapsody  "${PIP1}"

echo "  Re-pinning critical versions..."
"${PIP1}" install -q --force-reinstall \
    "protobuf>=6.31.1" \
    "setuptools<71"

echo ""
echo "  Verifying ve/ddmd..."
_check "${PY1}" "tensorflow"        "import tensorflow as tf; print(tf.__version__)"
_check "${PY1}" "keras"             "import keras; print(keras.__version__)"
_check "${PY1}" "MDAnalysis"        "import MDAnalysis; print(MDAnalysis.__version__)"
_check "${PY1}" "radical.asyncflow" "import radical.asyncflow; print('ok')"
_check "${PY1}" "rhapsody"          "import rhapsody; print('ok')"


# ══════════════════════════════════════════════════════════════════════════════
# 2. ve/ddmd-openmm — OpenMM + MD-tools env
#    Used by: MD simulation stage (GPU-accelerated OpenMM).
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══ Env 2/3: ve/ddmd-openmm ═══════════════════════════════════════"
ENV2="${VE_HOME}/ddmd-openmm"
PY2="${ENV2}/bin/python"
PIP2="${ENV2}/bin/pip"

_make_venv "${ENV2}"

echo "  Installing OpenMM..."
"${PIP2}" install -q "openmm>=8.0"

echo "  Installing workflow requirements..."
"${PIP2}" install -q -r "${WF_DIR}/requirements.txt"

echo "  Installing DeepDriveSim [dev,dragon] (editable)..."
"${PIP2}" install -q -e "${DDSIM_DIR}[dev,dragon]"

# Clone and install MD-tools (with Delta-compatible patches)
MDTOOLS_DIR="${BASE_DIR}/MD-tools"
if [ ! -d "${MDTOOLS_DIR}" ]; then
    echo "  Cloning MD-tools → ${MDTOOLS_DIR}"
    git clone https://github.com/braceal/MD-tools.git "${MDTOOLS_DIR}"
fi
echo "  Applying MD-tools_fix patches..."
cp -r "${WF_DIR}/MD-tools_fix/." "${MDTOOLS_DIR}/"
"${PIP2}" install -q -e "${MDTOOLS_DIR}"

_install_asyncflow "${PIP2}"
_install_rhapsody  "${PIP2}"

echo "  Re-pinning critical versions..."
"${PIP2}" install -q --force-reinstall \
    "protobuf>=6.31.1" \
    "setuptools<71"

echo ""
echo "  Verifying ve/ddmd-openmm..."
_check "${PY2}" "openmm"            "import openmm; print(openmm.__version__)"
_check "${PY2}" "MDAnalysis"        "import MDAnalysis; print(MDAnalysis.__version__)"
_check "${PY2}" "radical.asyncflow" "import radical.asyncflow; print('ok')"
_check "${PY2}" "rhapsody"          "import rhapsody; print('ok')"


# ══════════════════════════════════════════════════════════════════════════════
# 3. ve/ddmd-keras — TensorFlow / Keras env
#    Used by: delta_tf_gpu_wrapper.sh (training) and delta_tf_cpu_wrapper.sh (agent/LOF).
#    cuDNN 8.9.7.29 for A40 compatibility.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══ Env 3/3: ve/ddmd-keras ════════════════════════════════════════"
ENV3="${VE_HOME}/ddmd-keras"
PY3="${ENV3}/bin/python"
PIP3="${ENV3}/bin/pip"

_make_venv "${ENV3}"

echo "  Installing TensorFlow + Keras..."
"${PIP3}" install -q \
    "tensorflow[and-cuda]>=2.20.0" \
    "keras>=3.4.0" \
    "protobuf>=6.31.1"

echo "  Installing workflow requirements..."
"${PIP3}" install -q -r "${WF_DIR}/requirements.txt"

echo "  Installing DeepDriveSim [dev,dragon] (editable)..."
"${PIP3}" install -q -e "${DDSIM_DIR}[dev,dragon]"

_install_asyncflow "${PIP3}"
_install_rhapsody  "${PIP3}"

echo "  Re-pinning critical versions..."
"${PIP3}" install -q --force-reinstall \
    "protobuf>=6.31.1" \
    "setuptools<71"

echo ""
echo "  Verifying ve/ddmd-keras..."
_check "${PY3}" "tensorflow"        "import tensorflow as tf; print(tf.__version__)"
_check "${PY3}" "keras"             "import keras; print(keras.__version__)"
_check "${PY3}" "MDAnalysis"        "import MDAnalysis; print(MDAnalysis.__version__)"
_check "${PY3}" "radical.asyncflow" "import radical.asyncflow; print('ok')"
_check "${PY3}" "rhapsody"          "import rhapsody; print('ok')"


# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================="
echo "Setup complete.  Three venvs created under ${VE_HOME}:"
echo ""
echo "  ${VE_HOME}/ddmd        — main workflow (dragon, simulation, selection, agent)"
echo "  ${VE_HOME}/ddmd-openmm — OpenMM MD simulation + MD-tools"
echo "  ${VE_HOME}/ddmd-keras  — TensorFlow/Keras (training / inference)"
echo ""
echo "Activate the main env with:"
echo "  source ${VE_HOME}/ddmd/bin/activate"
echo ""
echo "Run the workflow:"
echo "  export WORK_DIR=${WORK_DIR}"
echo "  export SBATCH_ACCOUNT=<project>-delta-gpu"
echo "  cd ${WF_DIR}"
echo "  sbatch delta_gpu_sbatch.sh"
echo "================================================================="
