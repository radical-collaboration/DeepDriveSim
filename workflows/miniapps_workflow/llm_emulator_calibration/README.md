# LLM-service emulator: how it was built and how to check it

`llm.backend: emulated` replaces the real LLM service (RHAPSODY's `DragonVllmInferenceBackend`: Dragon Inference running vLLM on a GPU) with an emulator. The emulator reproduces the service's **measured properties**, not its content. The reason (Andrew, 11/04): in a mini-app, only the steering step's resource profile matters, not what the LLM says.

The emulator is `../llm_emulator.py`, and everything it needs to know is in one file, `profile_qwen2.5-7b-instruct_h100_smc.json`. This folder holds the measurements that file was built from, the scripts that build it, and the checks.

| | |
|---|---|
| Measured service | Qwen2.5-7B-Instruct, vLLM 0.19.1 through Dragon 0.15.0 Inference, RHAPSODY `142f396`, 1 H100 80GB, SMC node smc4 (`--exclusive`) |
| Measured workflow | agentic DeepDriveSim MiniApp, `configs/smc_default.yaml` (4 replicas, 10 sims, ~2,048-token prompts, `max_tokens` 512), 5 runs, 39 LLM requests |
| Source | `radical-experiments/ai-hpc-applications`, branch `exp2-agentic-deep-drive-sim`, commit `f85aa8b`, `exp2-agentic-deep-drive-sim/llm-service` (`inputs/SOURCES.json`) |

## What is emulated, and what is not

| Property | How the emulator produces it | Calibrated from |
|---|---|---|
| **Batching and queueing** | The same loop as RHAPSODY's backend: wait for a request, collect more for 0.5 s, send all pending requests as one batch, return every answer when the batch is done. Requests that arrive meanwhile wait for the next batch. | RHAPSODY code (`max_batch_wait_ms=500`, `max_batch_size=1024`), confirmed on the data (see "Checks", 1) |
| **Request latency (EQ3)** | Batch time = `overhead_s + s_per_token × longest answer in the batch (tokens)`, multiplied by `exp(N(−σ²/2, σ))` (mean 1) | 24 batches rebuilt from the measured requests |
| **Service startup (EC5)** | A wait drawn from the measured startups, `warm` (weights in page cache) or `cold`. With GPU load, its setup (~2 s) is part of the wait, as vLLM's GPU allocation is part of the real startup. | `llm_service.csv` |
| **Answer size** | sims in the prompt × tokens-per-sim, drawn from the 39 measured answers; capped at `max_tokens`; 1 token = 4 characters, the same estimate the workflow uses | `llm_requests.csv` |
| **Answer format** | Filler sentences plus `sim_<k>: <score>` per sim. A sim is left unscored with the measured probability. | `llm_requests.csv` |
| **Stop decisions** | Per request, with probability `stop_all_running_prob`, every sim listed as `state=running` in the prompt gets score 0.0; otherwise none does. Finished sims score ≥ 0.5. | `requests_running.csv` (see below) |
| **GPU load** (optional, `emulator_gpu: load`) | A child process reserves `gpu_memory_utilization` of the job's GPU, and keeps it busy (bf16 matmuls) for the measured fraction of each 100 ms while a request is in flight. On an SMC H100: 42.8 % busy at a 43.7 % target, 1.9 % idle. It's a process rather than a thread, so it doesn't compete with the workflow for the GIL; a thread version reached only 37 % and slowed the workflow. | `gpu_by_inflight.csv` (busy %); memory from the vLLM **config** (0.85), since it wasn't measured |
| Not emulated | The model's reasoning (answers are filler); host CPU and memory of the vLLM processes. Measured: node CPU 63.0 % agentic vs 60.8 % stock, memory 4.01 % vs 3.88 % (`host_not_emulated` in the profile). | |

## How the profile was derived

### Latency: two models, one rejected

1. **First model (rejected):** each request decodes its own answer, and requests in flight share the GPU (processor sharing).
   - It explained 47 % of the variation in the fitted data.
   - On held-out runs it was inconsistent: for runs 3 and 4 it was off by more than 4 s.
2. **What the data showed:** requests submitted together **finished at the same instant, whatever their answer lengths.**
   - Run 4, iteration 1: four answers of 158–187 tokens all returned at +5.4 s.
   - Run 3, iteration 2: answers of 197, 409 and 381 tokens all returned at +49.9 s.
   - Requests submitted while another was being processed waited for it to finish, then went out together.
3. **Where that comes from:** RHAPSODY's `DragonVllmInferenceBackend._batch_processor` (rhapsody `142f396`, `backends/ai/vllm.py`) does exactly that: collect for `max_batch_wait_ms = 500`, send the batch, wait for all its answers.
4. **What was fitted:** the emulator reproduces that loop and fits only the batch's model time. That is the time from the end of the collection window to the answers, rebuilt per batch in `calibration/batches.csv`:
   - `model_time = overhead_s + s_per_token × longest answer`, by least squares with `overhead_s ≥ 0`;
   - result: `overhead_s = 0`, `s_per_token = 0.01583` s (about 63 tokens/s), over 24 batches;
   - adding batch size, or using the sum of answers instead of the longest, barely helped (R² 0.51 → 0.53–0.55, `calibration/batch_models.csv`), so they were left out.
5. **The remaining spread:** about half the variation in batch time is not explained by any measured variable. For example, 409 tokens took 12.5 s in one batch, while 517 took 4.6 s in another. The emulator reproduces this as a random factor per batch, with the measured log-spread σ = 0.355 and mean 1.

### Stop decisions

- **A decision can only stop sims still running when its answer arrives.** The workflow counts stops against its registered (running) sims at that moment.
- **Count of running simulations:** `extract_running.py` counts, for every measured request, the replica's running **simulation** tasks, at submission and at the answer.
  - Each AsyncFlow task carries its replica's `asyncflow.workflow_id`.
  - Simulation and training tasks are told apart by the first line of their stdout (`Temp for Darshan, sim|ml, ...`).
- **The result:** all 23 requests whose sims were still running at the answer either **stopped all of them (13) or none (10), never some.** Every stop was scored 0.0 (`run.log`).
- **How it's emulated:** an all-or-nothing decision per request, `stop_all_running_prob = 13/23 = 0.565`. When it fires, **every** sim gets 0.0, so every running one is stopped.
- **By iteration:** 10/20 at iteration 1 and 3/3 at iteration 2. The model may stop more readily later on, but 3 cases can't support a separate rule.
- **Why every sim, and not just those the prompt calls running:** the workflow labels a sim `state=finished` whenever an output file with its index exists, even while that sim runs again. The real model saw those labels and still always stopped every running sim. Scoring all sims 0.0 reproduces that; the scores the real model gave finished sims aren't recorded, and don't affect the workflow.
- **Three earlier versions failed the end-to-end check, which is how this was found.** All their runs are superseded.
  1. "A score is low with probability 13.3 %, regardless of state" stopped 7× too few sims, and made runs longer (9.8 steering requests per run instead of 7.8).
  2. "Stop the sims labelled `state=running`" stopped all running sims in only 7 % of requests, because of the labelling above.
  3. "Stop all, with q counted against sims running **at submission**" (q = 13/29 = 0.448) stopped too rarely. In 6 of the 29 measured requests, all the sims finished during the request, so the decision could not stop anything; counting them diluted q.

### Everything else

- **Startup:** startups more than 1.3 × the median are cold. That gives 4 warm runs (21.98–22.39 s) and 1 cold run (35.07 s).
- **Unscored sims:** 1 − 330/333 = 0.009 of sims got no score.
- **GPU:**
  - busy 43.7 % while requests were in flight (1,849 samples), 0.8 % otherwise (7,263 samples);
  - by number in flight: 42.9 / 43.4 / 50.2 / 35.6 % for 1 / 2 / 3 / 4;
  - the other 7 GPUs read 0.0 %.

Every number in the profile is computed or fitted by `calibrate.py`. Only four things are set by hand, each with its source: `batch_wait_s` and `max_batch_size` (RHAPSODY defaults), `memory_fraction_config` (the campaign's config), and the provenance.

## Checks

### 1. Batching rules: exact (`calibration/mechanism_check.csv`)

Each run's requests were replayed through the emulator's own batch loop (`simulate_batches`), using the **observed** batch times. That isolates the batching rules (window, queueing, membership) from the timing model.

| run | requests | batches | same members | max latency error |
|---|---|---|---|---|
| rep_01 | 7 | 5 | 5 | 1.4 ms |
| rep_02 | 8 | 6 | 6 | 0.7 ms |
| rep_03 | 8 | 4 | 4 | 1.8 ms |
| rep_04 | 8 | 3 | 3 | 1.2 ms |
| rep_05 | 8 | 6 | 6 | 2.8 ms |

### 2. Latency distribution: inside the measured 95 % intervals (`calibration/distribution.csv`)

The 39 requests were replayed at their measured submission times, 2,000 times, with the calibrated randomness on. Measured intervals are 95 % bootstrap intervals from resampling requests.

| | mean (s) | p50 (s) | p95 (s) | max (s) |
|---|---|---|---|---|
| measured | 5.31 [4.42, 6.37] | 4.48 [3.80, 6.06] | 13.00 [6.50, 14.36] | 14.36 |
| emulated, measured answer sizes | 5.07 ✓ | 4.24 ✓ | 11.59 ✓ | 28.9 |
| emulated, sampled answers | 5.23 ✓ | 4.63 ✓ | 10.64 ✓ | 28.0 |
| answer tokens: measured / sampled | 217 / 218 | 182 / 204 | 409 / 414 | 517 / 512 |

- **The tail beyond the measurements is extrapolation.** The emulator can produce latencies up to about twice the longest measured one; with 24 batches, that tail isn't pinned down.

### 3. Per-request prediction: weak, as expected (`calibration/models.csv`, `calibration/loro.csv`)

Noise-free, the emulator predicts individual requests only roughly. Most of the variation is the batch-to-batch spread that it reproduces as randomness rather than predicts.

| model (all 39 requests) | MAE (s) | RMSE (s) | R² |
|---|---|---|---|
| constant (mean) | 2.23 | 3.10 | 0 |
| per request, own answer length, no batching | 1.88 | 2.53 | 0.34 |
| batch loop + batch time (emulator, noise-free) | 1.81 | 2.81 | 0.18 |

**Leave one run out** (fit on 4 runs, predict the 5th; MAE in s):

| held out | emulator | no batching | constant |
|---|---|---|---|
| rep_01 | **0.85** | 1.22 | 1.40 |
| rep_02 | **0.47** | 1.06 | 2.45 |
| rep_03 | **4.88** | 4.85 | 5.71 |
| rep_04 | 4.00 | 2.36 | **0.70** |
| rep_05 | **1.05** | 2.43 | 3.12 |

Run 3 has the slowest batches. Without it, `s_per_token` comes out at 0.0090 instead of 0.0158.

### 4. Unit tests (`tests/unit/test_llm_emulator.py`)

- **Batching rules,** hand-computed: one request; two in one window return together at the longest answer; a request during a batch waits; batch size limit; given batch times.
- **Noise** has mean 1.
- **Answers:** size and parse rate match the profile; stops are all running sims or none, never finished sims, at the profile rate.
- **The real-time backend** returns a batch together.
- **Provenance:** every input matches the SHA-256 in `inputs/SOURCES.json`, and the profile carries that provenance.
- **Reproducibility:** refitting from `inputs/` gives exactly the profile's parameters, and the batch loop with the observed batch times reproduces every measured latency within 10 ms.

### 5. End to end: the agentic workflow with the emulator

`validation/smc_emulator.sbatch` reruns the measured campaign's configuration with `llm.backend: emulated` (`validation/config_validation.yaml`, identical to the campaign's config apart from the `llm` section). `validation/validate_workflow.py` then compares each metric between the emulated and the measured runs, with a two-sample bootstrap (described below). It resamples requests for request-level metrics and runs for run-level metrics; with 5 runs per side, the run-level intervals are wide.

**Final runs:** 2026-11-09, SMC smc4, `--exclusive` (all 128 cores, as in the measured campaign), 5 runs per cell, all exit 0, no leftover processes, profile sha256 `1be59a10…`:
- **CPU-only emulator** (`emulator_gpu: none`): Slurm job 10000, commit `aee077c`.
- **GPU-load emulator** (`emulator_gpu: load`): job 10033, commit `0f44157`. Its only change since `aee077c` is the GPU-load child process; the profile is the same.

**The test.** "Consistent" means the 95 % bootstrap interval of the difference (emulated − measured), resampling both sides, contains 0. `validation/workflow_comparison.csv` also has the measured value's own interval, and the stricter one-sample check (`in_measured_ci95`). That one-sample check ignores the emulated side's sampling noise: across repeated 5-run cells, run-level metrics fell in and out of it.

| metric | stat | measured | CPU-only emulator | | GPU-load emulator | |
|---|---|---|---|---|---|---|
| EQ3 request latency (s) | mean | 5.311 | 5.94 (+11.8 %) | consistent | 5.126 (-3.5 %) | consistent |
| EQ3 request latency (s) | p50 | 4.483 | 5.77 (+28.7 %) | consistent | 4.917 (+9.7 %) | consistent |
| EQ3 request latency (s) | p95 | 12.996 | 13.468 (+3.6 %) | consistent | 8.925 (-31.3 %) | consistent |
| answer size (est. tokens) | mean | 217.154 | 248.976 (+14.7 %) | consistent | 238.619 (+9.9 %) | consistent |
| answer size (est. tokens) | p95 | 409.0 | 389.0 (-4.9 %) | consistent | 459.0 (+12.2 %) | consistent |
| prompt size (est. tokens) | mean | 2059.385 | 2056.732 (-0.1 %) | **differs** | 2059.333 (-0.0 %) | consistent |
| parse coverage | mean | 0.992 | 0.997 (+0.5 %) | consistent | 0.997 (+0.5 %) | consistent |
| stop-all decision, requests with running sims | mean | 0.565 | 0.5 (-11.5 %) | consistent | 0.462 (-18.3 %) | consistent |
| EQ4 completed | mean | 1.0 | 1.0 (+0.0 %) | consistent | 1.0 (+0.0 %) | consistent |
| batch size (requests) | mean | 1.625 | 1.64 (+0.9 %) | consistent | 1.273 (-21.7 %) | consistent |
| EQ1 requests per run | mean | 7.8 | 8.2 (+5.1 %) | consistent | 8.4 (+7.7 %) | consistent |
| EC5 service startup, warm (s) | mean | 22.167 | 22.109 (-0.3 %) | consistent | 22.106 (-0.3 %) | consistent |
| stage llm_steering (s) | mean | 5.311 | 5.94 (+11.8 %) | consistent | 5.126 (-3.5 %) | consistent |
| stage training (s) | mean | 28.481 | 29.281 (+2.8 %) | consistent | 27.105 (-4.8 %) | **differs** |
| workflow duration (s) | mean | 113.881 | 123.564 (+8.5 %) | consistent | 126.758 (+11.3 %) | consistent |
| GPU busy % while requests in flight | mean | 46.089 | – |  | 42.204 (-8.4 %) | consistent |
| GPU busy % with no request in flight | mean | 0.853 | – |  | 0.585 (-31.4 %) | consistent |
| node CPU % (host load, not emulated) | mean | 63.016 | 62.733 (-0.5 %) | reported | 61.87 (-1.8 %) | reported |
| node memory % (host load, not emulated) | mean | 4.009 | 3.9 (-2.7 %) | reported | 3.91 (-2.5 %) | reported |
| GPU memory reserved, max (MiB) | mean | not measured (config 0.85 × 81,559 = 69,325) | – | | 69910.0 | reported |

**What the table shows:**
- **Everything the emulator produces is consistent with the measured service,** in both cells. That covers latency (mean, median, p95), answer size, parse coverage, stop decisions, completion, batching, requests per run, warm startup, steering stage, and GPU busy % (in flight and idle).
- **Two rows differ, and neither is produced by the emulator:**
  - prompt size in the CPU cell: 3 tokens (0.1 %) short. The workflow builds the prompt, and its padding is nearly deterministic, so its interval is tiny.
  - training in the GPU-load cell: 4.8 % faster. Training doesn't use the LLM, and the CPU cell's training was consistent.
- **Workflow duration is consistent but runs +8 to +11 % high,** with wide intervals. Five runs per side can't pin down a difference of that size.
- **GPU busy % depends on CPU load in a subtle way.** Timed on the GPU itself, the emulator's duty cycle is 44–45 %, idle node or loaded (`validation/gpu_duty_check.sh`). But `nvidia-smi`/NVML utilization reads lower under host CPU load (37.7 % with 80 busy cores). In the workflow it read 42.2 %, against vLLM's 46.1 % under the same load. The emulator's target was **not** tuned to close that gap.
- **Host CPU and memory** (not emulated) come out close to the measured run anyway (−1.8 % and −2.5 %): the simulations and training dominate the host load.

### Validation history

Each round below found a problem, which was then fixed. Runs from superseded rounds are kept as `superseded/` data.

| round | change it tested | result | fix |
|---|---|---|---|
| 1 | stop: low score with p = 13.3 % | stopped 7× too few sims, so runs were longer | stop decision per request (all or none) |
| 2 | stop: sims labelled `state=running` | stopped all running sims in only 7 % of requests | score every sim 0.0; prompt labels are unreliable |
| 3 | q = 0.448 counted at submission; 32 cores; GPU load as a thread | stop rate low (decisions on sims that ended during the request); training and workflow slow (32 vs 128 cores); GPU 37 % (GIL) | count q at the answer (0.565); all 128 cores; GPU load as a child process |
| 4 | the above, `--exclusive` | everything the emulator produces consistent; GPU 35.7 % under CPU load | GPU work as one CUDA-graph launch per window |
| 5 (final) | the above | see the table | – |

Not part of any round: on the CPU partition (smc11, 256 cores), the full config failed at task launch with Dragon `DRAGON_MEMORY_ERRNO` (`shm_open` of the pool manifest) in 3 of 3 attempts. The small config ran fine there; raising the open-files limit to 65,536 did not help. On smc2 the system OpenMPI library (`libmpi.so.40`) is missing, so the simulations fail. Both nodes were left out; the cause on smc11 is open.

## Reproduce

```bash
# the measured runs: data/runs.tar.xz of the source folder (inputs/SOURCES.json has its sha256)
tar -xJf data/runs.tar.xz                                   # -> runs/
python extract_gpu.py runs        # -> inputs/gpu_by_inflight.csv, inputs/node_usage.csv
python extract_running.py runs    # -> inputs/requests_running.csv
python calibrate.py               # -> profile_*.json, calibration/*.csv (needs radical.asyncflow)
pytest tests/unit/test_llm_emulator.py

# end to end, on SMC (pass --exclusive when a node is free)
VENV=... DDSIM=<this checkout> OUT=runs LABEL=emulated_cpu REPS=5 GPU=none sbatch validation/smc_emulator.sbatch
python validation/validate_workflow.py runs emulated_cpu emulated_gpuload   # -> validation/workflow_comparison.csv, validation/emulated/
```

## Use

```yaml
llm:
  backend: emulated
  emulator_profile: ""          # "" = the profile in this folder
  emulator_startup: warm        # warm | cold | seconds
  emulator_startup_scale: 1.0   # e.g. 0.1 for quick tests
  emulator_gpu: none            # none | load (reserve GPU memory, busy while serving)
  emulator_seed: 0
  max_tokens: 512
  gpu_memory_utilization: 0.85  # used by emulator_gpu: load
```

Each run's `ddsim.InferenceService` events record `llm_backend: emulated`, the profile file, and its SHA-256.

## Limits

- **Calibrated range:**
  - prompts of 2,051–2,066 tokens (the prefill time is folded into the batch time);
  - answers of 78–517 tokens;
  - 6–10 sims per prompt;
  - 1–4 concurrent requests;
  - one model, one GPU, one node type.

  Outside this range the emulator extrapolates.
- **Small sample:** 39 requests, 24 batches, 5 runs. The tails, and the per-iteration stop rates, are uncertain.
- **Token counts are estimates** (characters ÷ 4), both in the measurements and in the emulator.
- **GPU memory is not measured.** The emulator reserves the configured fraction; vLLM's actual footprint may differ.
- **GPU busy % is matched in GPU time, not in how NVML samples it.** vLLM's many short decoding steps and the emulator's one burst per window read differently under host CPU load (see "Checks", 5).
- **Validated on one node type and one config:** SMC smc-ftc, `smc_default.yaml`, 5 runs per cell.
- **Content is not emulated.** The stop rule reproduces how often the real model stopped running sims under this prompt, not why.
