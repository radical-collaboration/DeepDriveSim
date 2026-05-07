#!/bin/bash
# =============================================================================
# DDMd workflow environment setup — Delta HPC (NCSA)
#
# Creates three Python 3.11 venvs mirroring the three conda envs in env_setup.sh:
#   1. ve/ddmd        — DeepDriveSim + Dragon  (orchestration, simulation cmd, selection, agent)
#   2. ve/ddmd-openmm — OpenMM + MD-tools       (GPU-accelerated MD)
#   3. ve/ddmd-keras  — TensorFlow/Keras         (training / inference)
#
# radical.asyncflow and rhapsody are installed from specific development branches:
#   AsyncFlow : https://github.com/radical-cybertools/radical.asyncflow  (prototype/telemetry)
#   Rhapsody  : https://github.com/radical-cybertools/rhapsody            (feature/telemetry)
#               installed as: pip install .[dragon]  then  pip install .[telemetry]
#
# Usage:
#   bash delta_env_setup.sh [--ve-home DIR] [--ddsim-dir DIR] [--base-dir DIR]
#
# Defaults:
#   VE_HOME   = /u/$USER/ve
#   DDSIM_DIR = /scratch/bblj/$USER/DeepDriveSim
#   BASE_DIR  = /scratch/bblj/$USER
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Parse optional overrides ──────────────────────────────────────────────────
VE_HOME="${VE_HOME:-/u/${USER}/ve}"
DDSIM_DIR="${DDSIM_DIR:-/scratch/bblj/${USER}/DeepDriveSim}"
BASE_DIR="${BASE_DIR:-/scratch/bblj/${USER}}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ve-home)   VE_HOME="$2";   shift 2 ;;
        --ddsim-dir) DDSIM_DIR="$2"; shift 2 ;;
        --base-dir)  BASE_DIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

WORK_DIR="${DDSIM_DIR}/workflows/ddmd_workflow"
ASYNCFLOW_DIR="${BASE_DIR}/radical.asyncflow"
RHAPSODY_DIR="${BASE_DIR}/rhapsody"

echo "================================================================="
echo "  VE_HOME       = ${VE_HOME}"
echo "  DDSIM_DIR     = ${DDSIM_DIR}"
echo "  BASE_DIR      = ${BASE_DIR}"
echo "  WORK_DIR      = ${WORK_DIR}"
echo "  ASYNCFLOW_DIR = ${ASYNCFLOW_DIR}"
echo "  RHAPSODY_DIR  = ${RHAPSODY_DIR}"
echo "================================================================="

# ── 0. Ensure 'module' is available (needed when run as bash script.sh) ───────
if ! command -v module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi

# ── Helper: find Python 3.11, loading cray-python/3.11.7 if needed ───────────
# On Delta, 'module load cray-python/3.11.7' exposes 'python3.11'.
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

# ── Helper: create a venv and bootstrap pip ───────────────────────────────────
_make_venv() {
    local env_dir="$1"
    local py="${env_dir}/bin/python3.11"
    local pip="${env_dir}/bin/pip"

    echo ""
    echo "── Creating venv: ${env_dir} ──"
    if [ ! -x "${py}" ]; then
        "${BASE_PY}" -m venv "${env_dir}"
        # Ensure python3.11 symlink exists regardless of base interpreter name.
        if [ ! -x "${py}" ]; then
            ln -sf "${BASE_PY}" "${py}"
        fi
        ln -sf "${py}" "${env_dir}/bin/python"  2>/dev/null || true
        ln -sf "${py}" "${env_dir}/bin/python3" 2>/dev/null || true
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
    "${pip}" install -q -e "${ASYNCFLOW_DIR}[telemetry]"
}

# ── Helper: install rhapsody from the feature/telemetry branch ───────────────
# Must be installed in two passes: .[dragon] first, then .[telemetry].
_install_rhapsody() {
    local pip="$1"
    echo "  Installing rhapsody [dragon] (feature/telemetry)..."
    "${pip}" install -q -e "${RHAPSODY_DIR}[dragon]"
    echo "  Installing rhapsody [telemetry] (feature/telemetry)..."
    "${pip}" install -q -e "${RHAPSODY_DIR}[telemetry]"
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
#    Equivalent to conda env 'deepdrivesim'.
#    Used by: workflow orchestrator (dragon), simulation cmd, model selection, agent.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══ Env 1/3: ve/ddmd ══════════════════════════════════════════════"
ENV1="${VE_HOME}/ddmd"
PY1="${ENV1}/bin/python3.11"
PIP1="${ENV1}/bin/pip"

_make_venv "${ENV1}"

echo "  Installing workflow requirements..."
"${PIP1}" install -q -r "${WORK_DIR}/requirements.txt"

echo "  Installing DeepDriveSim [dev,dragon] (editable)..."
"${PIP1}" install -q -e "${DDSIM_DIR}[dev,dragon]"

_install_asyncflow "${PIP1}"
_install_rhapsody  "${PIP1}"

echo "  Re-pinning critical versions..."
"${PIP1}" install -q --force-reinstall \
    "protobuf>=3.20.3,<5.0.0dev" \
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
#    Equivalent to conda env 'conda-openmm'.
#    Used by: MD simulation stage (GPU-accelerated OpenMM).
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══ Env 2/3: ve/ddmd-openmm ═══════════════════════════════════════"
ENV2="${VE_HOME}/ddmd-openmm"
PY2="${ENV2}/bin/python3.11"
PIP2="${ENV2}/bin/pip"

_make_venv "${ENV2}"

echo "  Installing OpenMM..."
"${PIP2}" install -q "openmm>=8.0"

echo "  Installing workflow requirements..."
"${PIP2}" install -q -r "${WORK_DIR}/requirements.txt"

echo "  Installing DeepDriveSim [dev,dragon] (editable)..."
"${PIP2}" install -q -e "${DDSIM_DIR}[dev,dragon]"

# Clone and install MD-tools (with Delta-compatible patches)
MDTOOLS_DIR="${BASE_DIR}/MD-tools"
if [ ! -d "${MDTOOLS_DIR}" ]; then
    echo "  Cloning MD-tools → ${MDTOOLS_DIR}"
    git clone https://github.com/braceal/MD-tools.git "${MDTOOLS_DIR}"
fi
echo "  Applying MD-tools_fix patches..."
cp -r "${WORK_DIR}/MD-tools_fix/." "${MDTOOLS_DIR}/"
"${PIP2}" install -q -e "${MDTOOLS_DIR}"

_install_asyncflow "${PIP2}"
_install_rhapsody  "${PIP2}"

echo "  Re-pinning critical versions..."
"${PIP2}" install -q --force-reinstall \
    "protobuf>=3.20.3,<5.0.0dev" \
    "setuptools<71"

echo ""
echo "  Verifying ve/ddmd-openmm..."
_check "${PY2}" "openmm"            "import openmm; print(openmm.__version__)"
_check "${PY2}" "MDAnalysis"        "import MDAnalysis; print(MDAnalysis.__version__)"
_check "${PY2}" "radical.asyncflow" "import radical.asyncflow; print('ok')"
_check "${PY2}" "rhapsody"          "import rhapsody; print('ok')"


# ══════════════════════════════════════════════════════════════════════════════
# 3. ve/ddmd-keras — TensorFlow / Keras env
#    Equivalent to conda env 'conda-keras'.
#    Used by: delta_tf_gpu_wrapper.sh (training) and delta_tf_cpu_wrapper.sh (agent/LOF).
#    cuDNN 8.9.7.29 for A40 compatibility (same as Bridges2 V100 fix).
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══ Env 3/3: ve/ddmd-keras ════════════════════════════════════════"
ENV3="${VE_HOME}/ddmd-keras"
PY3="${ENV3}/bin/python3.11"
PIP3="${ENV3}/bin/pip"

_make_venv "${ENV3}"

echo "  Installing TensorFlow 2.16.1 + Keras + cuDNN 8.9.7.29..."
"${PIP3}" install -q \
    "tensorflow[and-cuda]==2.16.1" \
    "nvidia-cudnn-cu12==8.9.7.29" \
    "keras==3.0.5"

echo "  Installing workflow requirements..."
"${PIP3}" install -q -r "${WORK_DIR}/requirements.txt"

echo "  Installing DeepDriveSim [dev,dragon] (editable)..."
"${PIP3}" install -q -e "${DDSIM_DIR}[dev,dragon]"

_install_asyncflow "${PIP3}"
_install_rhapsody  "${PIP3}"

echo "  Re-pinning critical versions..."
"${PIP3}" install -q --force-reinstall \
    "protobuf>=3.20.3,<5.0.0dev" \
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
echo "================================================================="
