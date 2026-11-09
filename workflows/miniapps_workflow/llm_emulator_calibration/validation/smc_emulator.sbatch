#!/bin/bash
#SBATCH -J llm-emulator-val
#SBATCH -p smc-ftc
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH -t 01:30:00
# Node sharing: dragon-cleanup kills ALL of the user's Dragon processes on a node, so
# never run two of these jobs on one node (chain them with --dependency=afterany).
# Pass --exclusive when a whole node is free (it was for the measured campaign).
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# End-to-end validation of the LLM emulator: the agentic workflow with
# llm.backend: emulated, same config and node type as the measured campaign
# (exp2/llm-service, SMC smc-ftc, --exclusive, dragon -s).
#
#   VENV=<venv with dragonhpc 0.15.0, rhapsody, asyncflow, wfMiniAPI>
#   DDSIM=<checkout of this branch>  OUT=<runs dir>  LABEL=emulated_cpu  REPS=5
#   GPU=none|load   sbatch [--exclusive] [-p smc-ftc-cpu --gres=gpu:0] smc_emulator.sbatch
#
# Writes OUT/LABEL/rep_NN/{config.yaml, run.log, meta.json, telemetry/, task_stdio/,
# ps_after.txt, gpu_mem.csv (GPU=load: nvidia-smi memory.used of the job's GPU, 1 s)}.
# =============================================================================
set -euo pipefail
: "${VENV:?}" "${DDSIM:?}" "${OUT:?}"
LABEL="${LABEL:-emulated_cpu}"
REPS="${REPS:-5}"
GPU="${GPU:-none}"
TIMEOUT_S="${TIMEOUT_S:-1200}"
# Optional: raise the soft open-files limit before Dragon starts (default 1024 on SMC).
# The full config failed on the 256-core CPU node smc11 at task launch with
# DRAGON_MEMORY_ERRNO (shm_open of the pool manifest) at 1024; see the README.
NOFILE="${NOFILE:-}"
if [ -n "${NOFILE}" ]; then ulimit -n "${NOFILE}"; fi
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
[ -f "${HERE}/config_validation.yaml" ] || HERE="${DDSIM}/workflows/miniapps_workflow/llm_emulator_calibration/validation"
CONFIG="${HERE}/config_validation.yaml"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
RUNNER="${DDSIM}/workflows/miniapps_workflow/run_agentic_workflow.py"
export PYTHONPATH="${DDSIM}" DDSIM_BACKEND=dragon OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

for rep in $(seq 1 "${REPS}"); do
    RUN="${OUT}/${LABEL}/rep_$(printf %02d "${rep}")"
    mkdir -p "${RUN}/task_stdio"
    python - "${CONFIG}" "${RUN}" "${GPU}" "${rep}" <<'EOF'
import sys, yaml
src, run, gpu, rep = sys.argv[1:]
c = yaml.safe_load(open(src))
c["home_dir"], c["telemetry_dir"] = f"{run}/MiniApps", f"{run}/telemetry"
c["llm"]["emulator_gpu"] = gpu
c["llm"]["emulator_seed"] = int(rep)  # different draws per repetition, reproducible
yaml.safe_dump(c, open(f"{run}/config.yaml", "w"), sort_keys=False)
EOF
    dragon-cleanup > /dev/null 2>&1 || true
    mon=""
    if [ "${GPU}" = load ]; then
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" -l 1 > "${RUN}/gpu_mem.csv" &
        # also the job's GPU before the run (baseline) is the first lines of this file
        mon=$!
    fi
    t0=$(date +%s.%N)
    set +e
    # without Slurm's launch variables the system OpenMPI initializes as a singleton
    (unset $(env | grep -oE "^(SLURM|PMIX|PMI)_[A-Za-z0-9_]*") \
        && cd "${RUN}/task_stdio" && timeout -k 30 "${TIMEOUT_S}" \
        dragon -s "${RUNNER}" --config_file "${RUN}/config.yaml") > "${RUN}/run.log" 2>&1
    rc=$?
    set -e
    t1=$(date +%s.%N)
    [ -n "${mon}" ] && kill "${mon}" 2>/dev/null || true
    dragon-cleanup > /dev/null 2>&1 || true
    sleep 2
    ps -u "${USER}" -o pid=,etime=,comm=,args= | awk '$3 ~ /^(python|dragon)/' > "${RUN}/ps_after.txt" || true
    python - "${RUN}" "${rc}" "${t0}" "${t1}" "${LABEL}" "${GPU}" <<'EOF'
import json, os, socket, sys
run, rc, t0, t1, label, gpu = sys.argv[1:]
meta = {"label": label, "workflow": "agentic", "llm_backend": "emulated", "emulator_gpu": gpu,
        "backend": "dragon", "exit_code": int(rc), "t_start": float(t0), "t_end": float(t1),
        "wall_s": float(t1) - float(t0), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "leftover_processes": sum(1 for _ in open(f"{run}/ps_after.txt")),
        "nofile_soft": int(os.popen("ulimit -n").read() or 0), "cores": os.cpu_count()}
json.dump(meta, open(f"{run}/meta.json", "w"), indent=2)
print(f"[{label} {os.path.basename(run)}] exit={rc} wall={meta['wall_s']:.1f}s leftover={meta['leftover_processes']}")
EOF
done
