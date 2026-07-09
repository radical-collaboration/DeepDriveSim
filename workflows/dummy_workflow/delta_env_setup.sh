#!/bin/bash
# =============================================================================
# Dummy workflow environment setup — Delta HPC (NCSA)
#
# Creates a Python 3.11 venv and installs all dependencies.
#
# Usage:
#   bash delta_env_setup.sh [--env-dir DIR] [--ddsim-dir DIR]
#
# Defaults (set SCRATCH to your allocation scratch root, e.g. /scratch/bblj):
#   ENV_DIR   = /u/$USER/ve/ddsim
#   DDSIM_DIR = $SCRATCH/$USER/DeepDriveSim
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Parse optional overrides ──────────────────────────────────────────────────
ENV_DIR="${ENV_DIR:-/u/${USER}/ve/ddsim}"
if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: set the SCRATCH env var to your allocation scratch root, e.g.:"
    echo "  export SCRATCH=/scratch/<allocation>"
    echo "  bash delta_env_setup.sh"
    exit 1
fi
DDSIM_DIR="${DDSIM_DIR:-${SCRATCH}/${USER}/DeepDriveSim}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-dir)   ENV_DIR="$2";   shift 2 ;;
        --ddsim-dir) DDSIM_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

WORK_DIR="${DDSIM_DIR}/workflows/dummy_workflow"
PY="${ENV_DIR}/bin/python3.11"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  ENV_DIR   = ${ENV_DIR}"
echo "  DDSIM_DIR = ${DDSIM_DIR}"
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

# ── 4. DeepDriveSim + Dragon (editable) ───────────────────────────────────────
echo ""
echo "── Step 4: DeepDriveSim [dragon] (editable) ──"
"${PIP}" install -q -e "${DDSIM_DIR}[dragon]"

# ── 5. radical.asyncflow ──────────────────────────────────────────────────────
echo ""
echo "── Step 5: radical.asyncflow + rhapsody ──"
"${PIP}" install --force-reinstall "radical.asyncflow>=0.3.0"
"${PIP}" install --force-reinstall "rhapsody-py>=0.3.0"
"${PIP}" install -q matplotlib

# ── 6. Verify ─────────────────────────────────────────────────────────────────
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

_check "scikit-learn"      "${PY}" -c "import sklearn; print(sklearn.__version__)"
_check "pyyaml"            "${PY}" -c "import yaml; print(yaml.__version__)"
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
