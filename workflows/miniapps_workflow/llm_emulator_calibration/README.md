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
| **Service startup (EC5)** | A sleep drawn from the measured startups, `warm` (weights in page cache) or `cold` | `llm_service.csv` |
| **Answer size** | sims in the prompt × tokens-per-sim, drawn from the 39 measured answers; capped at `max_tokens`; 1 token = 4 characters, the same estimate the workflow uses | `llm_requests.csv` |
| **Answer format** | Filler sentences plus `sim_<k>: <score>` per sim. A sim is left unscored with the measured probability. | `llm_requests.csv` |
| **Stop decisions** | Per request, with probability `stop_all_running_prob`, every sim listed as `state=running` in the prompt gets score 0.0; otherwise none does. Finished sims score ≥ 0.5. | `requests_running.csv` (see below) |
| **GPU load** (optional, `emulator_gpu: load`) | Reserves `gpu_memory_utilization` of the job's GPU, and keeps it busy for the measured fraction of time while a request is in flight | `gpu_by_inflight.csv` (busy %); memory from the vLLM **config** (0.85), since it wasn't measured |
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

- **The workflow can only stop a running simulation;** a low score for a finished one has no effect.
- **Count of running simulations:** `extract_running.py` counts, for every measured request, the running simulation tasks of the same replica at submission. Each AsyncFlow task carries its replica's `asyncflow.workflow_id`.
- **The result:** all 29 requests that had running sims either **stopped all of them (13) or none (16), never some.** Every stop was scored 0.0 (`run.log`).
- **How it's emulated:** an all-or-nothing decision per request, `stop_all_running_prob = 13/29 = 0.448`. When it fires, **every** sim gets 0.0, so every running one is stopped. By iteration the split was 10/20 at iteration 1 and 3/9 at iteration 2; that's too few to model separately.
- **Why every sim, and not just those the prompt calls running:** the workflow labels a sim `state=finished` whenever an output file with its index exists, even while that sim runs again. The real model saw those labels and still always stopped every running sim. Scoring all sims 0.0 reproduces that; the scores the real model gave finished sims aren't recorded, and don't affect the workflow.
- **Two earlier versions failed the end-to-end check, which is how this was found.**
  1. "A score is low with probability 13.3 %, regardless of state" stopped 7× too few sims, and made runs longer (9.8 steering requests per run instead of 7.8).
  2. "Stop the sims labelled `state=running`" stopped all running sims in only 7 % of requests, because of the labelling above.

  Both sets of runs are superseded.

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

`validation/smc_emulator.sbatch` reruns the measured campaign's configuration with `llm.backend: emulated` (`validation/config_validation.yaml`, identical to the campaign's config apart from the `llm` section). `validation/validate_workflow.py` then compares each metric with the measured value's 95 % bootstrap interval. It resamples requests for request-level metrics and runs for run-level metrics; with 5 runs, the run-level intervals are wide.

_Results: see `validation/workflow_comparison.csv` (filled in below once the runs finish)._

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
python validation/validate_workflow.py runs emulated_cpu emulated_gpuload
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
- **Content is not emulated.** The stop rule reproduces how often the real model stopped running sims under this prompt, not why.
