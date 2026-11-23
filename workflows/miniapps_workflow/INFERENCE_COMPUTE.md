# Device-portable inference computation for mini-app steering

Status: first functional implementation, November 18, 2026. **Synthetic decoder,
not measured Qwen trace replay or validated Qwen resource equivalence.**

## Target

Execute the same inference tensor program on CPU or CUDA. Inference latency comes
from doing the work, not from sleeping for an H100-calibrated duration. No training,
backward pass, model download, or trained weights are involved.

`inference_compute.py` implements:

- deterministic synthetic embeddings and untied vocabulary projection;
- decoder layers with RMSNorm, rotary positions, grouped-query causal attention;
- gated SiLU MLPs and residual connections;
- a real per-layer key/value cache;
- one prompt prefill followed by autoregressive decoding with greedy token feedback.

The dimensions, dtype and seed are explicit. Changing `device: cpu` to
`device: cuda` or `cuda:1` selects the backend for the **same tensor code**. Unsupported
devices/operators/dtypes fail rather than silently selecting a different precision
or falling back to CPU. PyTorch can use different lower-level implementations on
each device; CUDA kernels themselves are not portable to a CPU.

Logical work, parameter storage and cache structure can be compared across devices.
Identical physical memory usage, utilization, hardware counters, bitwise results or
runtime are **not** promised. In particular, do not impose an expected CPU slowdown
factor. CUDA timing synchronizes the selected device around execution.

## Integration

`llm_compute_backend.py` implements `llm.backend: compute` through the existing
AsyncFlow `prompt_task` steering interface. It owns one workload and serializes
batches on a worker thread, so tensor execution does not block the workflow's
async event loop. Requests already queued are grouped up to `max_batch_size`;
unequal prompt sizes are padded to the longest. This is **not vLLM's scheduler**.
The request queue is bounded; oversize requests fail instead of being truncated.
Shutdown waits for active work, fails queued work and does not leave a compute
thread running. This means shutdown is not immediate for a large computation.

Prompt lengths use the existing rough characters/4 convention; there is no real
tokenizer. Input IDs are synthetic, representative by shape only. `new_tokens` is
the fixed number of tokens actually generated, not the length of the formatted
score response. After computation completes, seeded pseudo-random scores are
returned as `sim_<id>: <score>` lines. `llm.compute.steering_seed` (default 0) and
the prompt's replica/iteration/simulation context determine each score, independent
of device, generated token IDs, batch membership and request order. The full
prefill/decode computation still runs; it is not replaced by a random-number draw.
**These are not meaningful scientific decisions or a reproduction of measured
Qwen stop probabilities.**

`InferenceOnlyMiniAppsWorkflow` registers only simulation and steering tasks. It:

1. submits simulations with no training reservation;
2. waits for at least one output file from a successfully completed producer;
3. advances the inference iteration, asks for scores and applies cancellation;
4. repeats while simulations remain, then cleans up.

Steering polls at `agentic.steering_interval_s`; that is workflow cadence, not an
inference delay. A simulation failure or failed inference exits with an error,
instead of waiting for nonexistent training data. Simulations that all finish
before steering is needed can complete without any inference request. Output
content is still synthetic; only file metadata and generated observations enter
the prompt. Real scientific steering would require a separate validated input and
decision design.

The new runner configuration requires `agentic.inference_only: true` when using
`llm.backend: compute`. The old stock workflow, old agentic config, measured
artifacts and calibrated timing emulator are preserved unchanged. Their published
whole-workflow validation **does not apply** to this no-training computation mode.

For concurrent execution the inference-only runner uses `InferenceExecutionBackend`,
a local adapter that terminates/reaps the command's POSIX process group before
acknowledging cancellation. The pinned RHAPSODY backend otherwise cancels only its
asyncio waiter, leaving the command alive. Stock execution and installed runtime
packages are untouched; this adapter does not promise to kill arbitrary Python
functions in a thread executor.

Optional inference-only `simulation` settings (`mat_size`, `read_size`, `write_size`,
`steps_by_sim`) bound the real mini-app kernels; they do not substitute toy tasks.
`agentic.keep_outputs: true` retains outputs for auditing (default false). A
`run_summary_path` records completed/stopped simulation IDs and request diagnostics;
it refuses to overwrite an existing summary. The runner also fails if MiniApp
telemetry verification is incomplete.

## Tiny CPU example

PyTorch >=2.0 is an optional dependency (`inference-compute` extra). Use an existing
PyTorch environment; installing dependencies or downloading models is not part of
running the code below. From the repository root:

```python
from workflows.miniapps_workflow.inference_compute import (
    InferenceSpec,
    InferenceWorkload,
)

spec = InferenceSpec(
    hidden_size=16,
    intermediate_size=24,
    num_layers=1,
    num_heads=4,
    num_kv_heads=2,
    vocab_size=32,
    max_batch_size=1,
    max_prompt_tokens=16,
    max_new_tokens=4,
)
workload = InferenceWorkload(spec, device="cpu")
print(workload.run(batch_size=1, prompt_tokens=8, new_tokens=3))
```

This executes one prefill and two decode calls, with 11,968 bytes of parameter
storage and a 640-byte KV cache. It is a correctness-sized workload, not a model
performance measurement. `elapsed_s` includes request tensor setup, computation,
GPU synchronization when applicable, and transfer of generated IDs back to host;
it does not include initial synthetic-weight construction or request queueing.

There is also a standalone smoke-check command (defaults to tiny dimensions):

```bash
python -m workflows.miniapps_workflow.inference_compute --device cpu \
  --prompt-tokens 8 --new-tokens 3
# On approved GPU hardware, the same command with --device cuda executes the
# same program. Keep --spec-json, dtype and request sizes identical to compare.
```

`--spec-json FILE` accepts a JSON object of `InferenceSpec` fields. The command
prints the effective specification and execution diagnostics. It does not
initialize AsyncFlow, Dragon, vLLM or a trained model.

`config_inference_compute.yaml` is a larger but still tiny integration example.
With the normal workflow dependencies and simulation kernels already installed:

```bash
DDSIM_BACKEND=concurrent python -m workflows.miniapps_workflow.run_agentic_workflow \
  --config_file workflows/miniapps_workflow/config_inference_compute.yaml
```

That command runs the full mini-app and is **not** part of the tiny unit-test
verification. Review resources before using it. Direct CPU/GPU computation was
subsequently tested on Nimitz with Vijay's explicit authorization; see the hardware
validation section below. That does not authorize unrelated future GPU campaigns.

The factory returns the effective compute configuration and its SHA-256; the
runner includes both in service events. Per-batch compute diagnostics are attached
to the backend task dictionary as `inference_compute`; the workflow records them
in `ddsim.LLMRequest.compute_json`, alongside `scores_json`. This links measured
computation/device information, AsyncFlow task UID and steering decisions.

## Bounds and fidelity gaps

- Defaults are small and have no claim to match the 7B model. All request dimensions
  are bounded; estimated weights, cache and scratch tensors are checked against
  `max_tensor_bytes` before allocation. This is **not** an OS memory cap: library
  workspaces, initialization temporaries, allocator overhead and total process RSS
  are not measured or strictly bounded by it. Cache capacity is for each actual
  batch, not a vLLM-style paged global pool.
- The decoder is architecture-derived, not extracted from an operator trace. No
  operator/shape trace was found in the initial bounded artifact inspection.
- No attention/MLP fusion equivalence, CUDA graph schedule, quantization, vLLM
  memory management, distributed inference or hardware portability result is claimed.
- Random weights remove the need for learned model content; they do not prove
  equivalent runtime or memory traffic. PyTorch SDPA, GQA expansion and temporaries
  can differ materially from the measured vLLM implementation.
- No host-memory reservation or unrelated GPU duty-cycle generator is used to
  manufacture a utilization match in computation mode.
- Next fidelity milestone: establish representative operator/tensor dimensions
  from a real inference profile or checked model architecture, then perform a
  separately approved CPU/GPU validation. Retain the old timing backend as a
  distinct baseline, not evidence that the new computation is already faithful.

## Local verification

Selected files: `test_inference_compute.py`, `test_llm_compute_backend.py`,
`test_inference_workflow.py`, plus existing `test_agentic_workflow.py` and
`test_llm_emulator.py`.

CPU tests cover bounds, real prefill/decode, cache accounting, cached versus full
causal attention, deterministic weights, disabled gradients, explicit precision,
queue/error/shutdown handling, AsyncFlow request integration, and a complete toy
simulation -> real CPU computation -> score -> cancellation loop. The inference
loop also tests completion, failures, external cancellation and no-training
behavior. A completion/cancellation race is guarded so a finished simulation is
not counted twice.

CUDA tests require **explicit** `DDSIM_TEST_CUDA=1`; they are skipped by default.
No CPU/GPU timing ratio or hardware-fidelity result is inferred from CPU tests.

## Authorized Nimitz CPU/GPU validation — November 18

[Complete results and reproduction](../../../../runs/inference-compute/2026-11-18-nimitz/README.md).
The compute test file passed **23 tests** on Nimitz, including CUDA FP32/BF16
weight, numerical-logit and cache checks. Five CPU/GPU workload pairs completed
with identical program/spec hashes, request shapes and logical storage. The
608-million-parameter synthetic case took a median **1,003.79 ms CPU / 114.35 ms
H200 (8.78×)** in FP32, and **592.15 / 141.91 ms (4.17×)** in BF16. CPU here means
eight pinned physical cores of the Nimitz Xeon 6530P, not the whole machine.

The new `validate_inference_compute.py` harness runs one device per process,
performs two warmups and five timed requests, captures real allocation/RSS counters
and an untimed forward/cache shape check. A standalone invocation, using a bounded
spec JSON and fresh output path, is:

```bash
python -m workflows.miniapps_workflow.validate_inference_compute \
  --spec-json spec.json --device cpu --batch-size 1 --prompt-tokens 256 \
  --new-tokens 32 --threads 8 --warmups 2 --repetitions 5 --output cpu.json
```

On approved GPU hardware use `--device cuda` and a different output file, keeping
all other settings identical. Bare `cuda` resolves to the current visible device.
FP32 generated IDs matched across devices in these cases. BF16 IDs did not,
although same-input small-config numerical checks passed; greedy choices can change
with floating-point differences. Logical tensor storage matches, physical
CPU/GPU overhead does not. This is **not** Qwen/vLLM fidelity validation or a full
GPU steering-workflow campaign by itself. No trained weights were loaded.

## Full mini-app smoke tests — November 18 follow-up

[Actual-runner CPU/GPU results and artifacts](../../../../runs/inference-compute/2026-11-18-full-miniapp-nimitz/README.md).
At Vijay's explicit request, ran the normal `run_agentic_workflow` entry point on
Nimitz with the real `simulation.py`/wfMiniAPI compute and HDF5 kernels, using
inference on CPU and then CUDA. Both passed: four simulations, two actual inference
requests, one normal completion and three physical process cancellations, no
training tasks, zero failed/abandoned tasks and complete telemetry. Seeded scores
and stopped simulation IDs matched across devices; all processes were reaped.

These are one-host **concurrent execution** smoke tests, not Dragon or a real
molecular-dynamics campaign. The inference workload is still an architecture-derived
synthetic decoder. This closes the full mini-app integration smoke-test gap, not
representative Qwen trace/resource fidelity. Local regression then: **180 passed,
4 skipped**, including new seeded-policy, readiness and real-process-cancellation
checks. The archived source exactly matches the tested working-tree Python files.

## Coding-standards pass — November 18

The new modules were audited against `CONTRIBUTING.md` (Ruff lint and format, type
hints, docstrings on classes and public methods, `asyncio.to_thread` for blocking
I/O, tests for new functionality) and refactored without changing behavior.

- Every function in the five new modules is annotated and documented. Validation
  no longer monkeypatches a private method: `InferenceWorkload.forward`,
  `new_caches`, `parameters` and the `run(..., observer=...)` hook are public.
  `ComputeLLMBackend.start` and `AgenticMiniAppsWorkflow.requests` replace reaches
  into private state from the runner and the tests.
- File opens in async code go through `asyncio.to_thread`; process cleanup uses
  `contextlib.suppress`; the workflow's `__init__` validation is split into helpers.
- `ruff check` and `ruff format --check` pass on every file added or modified,
  including the Python blocks in this document. Remaining advisory findings from
  rules the project does not enable are deliberate: torch is imported lazily
  because it is an optional dependency, the worker catches `Exception` so a
  workload failure becomes a failed request, and configuration errors stay
  `ValueError`.
- Equivalence: the refactored `inference_compute.py` produces bitwise-identical
  weights, caches, logits and results to the version tested earlier, in FP32 and
  BF16. On Nimitz the generated tokens and forward traces match the earlier CPU
  and H200 runs, and an interleaved old/new timing comparison differs by under
  1%, inside the run-to-run spread.
- Local regression: **182 passed, 4 skipped**; the new tests pass ten times in a
  row. On Nimitz, 24 compute tests pass with CUDA enabled and the full mini-app
  CPU/GPU smoke tests pass again with this exact source
  ([recheck](../../../../runs/inference-compute/2026-11-18-standards-recheck-nimitz/README.md)).

Not covered: the repository's own CI lint (`ruff check .`) already fails on files
this work did not touch, and there is no Python 3.9 interpreter here to run the
suite on the oldest supported version (Ruff's py39 target check passes).

## Inference profiles — November 22

A profile is a stored description of a real model that the emulator loads in
place of hand-chosen dimensions. Profiles live in `inference_profiles/`, one JSON
per model, with the files they were derived from under
`inference_profiles/sources/<name>/`.

```yaml
llm:
  backend: compute
  compute:
    device: cuda                      # or cpu; the profile is the same
    profile: qwen2.5-7b-instruct      # packaged name, or a path to a profile
    new_tokens: 217
    spec:                             # per-run limits only
      max_batch_size: 1
      max_prompt_tokens: 2100
      max_new_tokens: 256
      max_tensor_bytes: 21474836480   # 20 GiB; must be stated for a real model
```

With `profile`, the architecture cannot be overridden; `spec` holds only the
limits and an optional `seed`. Each run records the profile's name and hash.

**What `qwen2.5-7b-instruct` contains.** Schema `ddsim.inference_profile/1`, level
`architecture`: the model's `config.json` and the tensor names, shapes and dtypes
read from the headers of its four weight files. Nothing was loaded or run. The
source is the copy used by the November 8 campaign (revision `a09a3545…` as that
campaign recorded it; not re-verified against the Hub).
`read_safetensors_headers.py` reads the headers; `inference_profile.py` builds
and checks the profile; a unit test rebuilds the shipped profile from its sources.

**What it guarantees.** The emulator allocates exactly the real model's tensors:
339 tensors, 7,615,616,512 parameters, identical shapes, BF16. To get there the
decoder gained two options that default to the previous behavior: bias terms on
the query/key/value projections and a configurable rotary base. Building a
profile fails if the model has a feature the decoder does not reproduce (tied
embeddings, a different activation, sliding-window attention) or if the two
tensor inventories differ.

**What it does not contain.** No measurement of real inference: no operator
timings, no measured memory of a real process, nothing about a serving engine.
`measurements` is empty and `not_captured` lists these. Weights are still random.

**Real-size run.** [Results](../../../../runs/inference-profile/2026-11-22-qwen2.5-7b/README.md):
on Nimitz the emulator ran at full Qwen size on eight CPU cores and one H200.
A request shaped like the November 8 campaign (2,059 prompt tokens, 217
generated) took a median 2.42 s on the H200 and 47.08 s on CPU.
