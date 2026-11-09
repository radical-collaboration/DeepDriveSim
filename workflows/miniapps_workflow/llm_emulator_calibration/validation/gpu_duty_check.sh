# GPU duty-cycle check (SMC H100, --exclusive): times the emulator's CUDA-graph duty cycle with
# CUDA events, idle and with 80 busy-loop processes. Result (2026-11-09, smc4):
#   idle:    GPU time per window 45.0 ms, cycle 100.0 ms, resets 0, achieved duty 45.0%
#   80 hogs: GPU time per window 44.4 ms, cycle 100.0 ms, resets 0, achieved duty 44.4%
# while nvidia-smi utilization.gpu read 45.9% (idle) and 37.7% (80 hogs) for the same load.
# Usage: srun -p smc-ftc --exclusive -N1 --gres=gpu:1 bash gpu_duty_check.sh
source ~/ai-hpc-applications/exp2-agentic-deep-drive-sim/llm-service/.venv/bin/activate
echo "affinity: $(python -c "import os; print(len(os.sched_getaffinity(0)))") cpus"
python - <<"PY"
import torch, time, subprocess, sys, statistics as st
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); out = torch.empty_like(a)
for _ in range(10): torch.matmul(a, a, out=out)
torch.cuda.synchronize()
n = 258; s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): torch.matmul(a, a, out=out)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(n): torch.matmul(a, a, out=out)
torch.cuda.synchronize()
def run(sec):
    gpu, cyc, resets = [], [], 0
    nxt = t0 = time.perf_counter(); last = t0
    while time.perf_counter() - t0 < sec:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        gpu.append(e0.elapsed_time(e1))
        nxt += 0.1; now = time.perf_counter()
        if nxt > now: time.sleep(nxt - now)
        else: nxt = now; resets += 1
        t = time.perf_counter(); cyc.append((t - last) * 1000); last = t
    return st.fmean(gpu), st.fmean(cyc), resets, sum(gpu) / (sum(cyc))
for cond in ("idle", "80 hogs"):
    hogs = [] if cond == "idle" else [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(80)]
    time.sleep(1)
    gm, cm, r, duty = run(6)
    for h in hogs: h.kill()
    print(f"{cond}: GPU time per window {gm:.1f} ms, cycle {cm:.1f} ms, resets {r}, achieved duty {duty*100:.1f}%")
PY
