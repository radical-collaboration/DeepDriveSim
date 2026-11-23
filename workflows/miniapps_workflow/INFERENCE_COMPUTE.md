# Compute emulator for LLM steering

The agentic mini-app asks an LLM which simulations to keep. This emulator stands
in for that LLM. It executes the inference computation of a decoder-only model
on a CPU or a GPU, so a workflow can be exercised with a realistic inference
cost on either device, without a model service and without the model's weights.

```
simulations ──► outputs ──► inference request ──► scores ──► stop / continue
                                 │
                 prefill + autoregressive decode, executed
                 on the configured device (cpu or cuda)
```

Select it with `llm.backend: compute`. It runs with the inference-only workflow,
which has no training stage.

What it is and is not:

- The time a request takes comes from executing the work. Nothing is slept, and
  there is no utilization target. A CPU is slower because it does the same work
  more slowly.
- Weights are random by default. Inference cost does not depend on their values,
  so the footprint of a model is reproduced without the model.
- Steering scores are seeded pseudo-random numbers, produced after the
  computation has run. They are not scientific decisions.
- It emulates a model's computation in eager PyTorch. It does not emulate a
  serving engine such as vLLM.

## Quick start

A small end-to-end run on one node, CPU only, with the concurrent backend:

```bash
DDSIM_BACKEND=concurrent python -m workflows.miniapps_workflow.run_agentic_workflow \
  --config_file workflows/miniapps_workflow/config_inference_compute.yaml
```

The workload on its own, with no workflow dependencies (PyTorch only):

```bash
python -m workflows.miniapps_workflow.inference_compute --device cpu \
  --prompt-tokens 8 --new-tokens 3
```

PyTorch is an optional dependency: `pip install -e ".[inference-compute]"`.

## Configuration

```yaml
agentic:
  inference_only: true        # required with the compute backend
  steering_interval_s: 0.1    # how often the workflow asks; not inference time
  stop_threshold: 0.5         # stop a running simulation scoring below this

llm:
  backend: compute
  compute:
    device: cpu               # or cuda, cuda:1; the program is the same
    new_tokens: 16            # tokens generated per request
    steering_seed: 0          # seed of the steering scores
    max_pending: 64           # queued requests before new ones are refused
    spec:                     # model dimensions and per-run limits
      hidden_size: 64
      intermediate_size: 128
      num_layers: 2
      num_heads: 4
      num_kv_heads: 2
      vocab_size: 128
      dtype: float32          # float32, float16 or bfloat16
      max_batch_size: 4
      max_prompt_tokens: 512
      max_new_tokens: 32
      max_tensor_bytes: 268435456
```

To emulate a real model, name an inference profile instead of giving dimensions.
The profile fixes the architecture; `spec` then holds only the limits:

```yaml
llm:
  backend: compute
  compute:
    device: cuda
    profile: qwen2.5-7b-instruct   # packaged name, or a path to a profile
    new_tokens: 217
    spec:
      max_batch_size: 1
      max_prompt_tokens: 2100
      max_new_tokens: 256
      max_tensor_bytes: 21474836480   # 20 GiB; must be stated for a real model
```

Other settings of the inference-only workflow:

| Key | Meaning |
|---|---|
| `simulation.mat_size`, `read_size`, `write_size` | Sizes passed to the simulation kernel |
| `simulation.steps_by_sim` | Step count per simulation ID, overriding `num_step` |
| `agentic.keep_outputs` | Keep simulation outputs after the run (default: remove) |
| `run_summary_path` | Write a JSON summary of requests and stopped simulations |

A request that does not fit is refused, not truncated: a prompt longer than
`max_prompt_tokens`, a full queue, or dimensions whose estimated tensors exceed
`max_tensor_bytes`. That budget covers tensors only, not total process memory.

## How it works

| File | Role |
|---|---|
| `inference_compute.py` | The decoder as a tensor program, and its size limits |
| `llm_compute_backend.py` | AsyncFlow backend: queues requests, runs them, returns scores |
| `inference_workflow.py` | The inference-only steering loop |
| `inference_execution.py` | Concurrent execution whose cancellation stops the process |
| `inference_profile.py` | Stored model profiles; loading real weights |
| `inference_profiles/` | The profiles, their source files, and the header reader |
| `validate_inference_compute.py` | Times one device and records memory and shapes |

**The workload.** `InferenceWorkload.run` executes one request: a prefill over the
prompt, then greedy decoding one token at a time with a key/value cache. The
layers are those of current decoder LLMs: RMSNorm, rotary positions,
grouped-query attention and a gated SiLU MLP. Requesting CUDA never falls back
to the CPU, and the configured precision is never substituted.

**The backend.** One worker owns the workload and runs batches on a thread, so the
event loop stays responsive. Requests already waiting are grouped up to
`max_batch_size` and padded to the longest prompt. Prompt length is estimated as
characters divided by four; there is no tokenizer. Scores are a hash of the
seed, replica, iteration and simulation ID, so they are the same on any device
and in any request order.

**The workflow.** `InferenceOnlyMiniAppsWorkflow` submits simulations, waits for
one to finish and leave its output, asks for scores, cancels the simulations
scoring below the threshold, and repeats. A failed simulation or request ends
the run with an error.

**Cancellation.** With the concurrent backend, cancelling a task normally cancels
only the wait for it and leaves the process running. `InferenceExecutionBackend`
puts each command in its own process group and reaps it before reporting the
cancellation.

**Telemetry.** Each request is an AsyncFlow prompt task. Its `ddsim.LLMRequest`
event carries the task ID, the scores (`scores_json`) and what was computed
(`compute_json`: device, token counts, storage, elapsed time). The service event
records the effective configuration and its hash, and the profile if one was
used.

## Inference profiles

A profile describes a real model so the emulator can take its shape: layer count,
widths, attention heads, vocabulary, precision, and the list of weight tensors.
It is built from the model's `config.json` and the headers of its weight files.
The weights are not loaded and the model is not run.

`qwen2.5-7b-instruct` is included. With it the emulator allocates exactly the
real model's tensors: 339 tensors and 7,615,616,512 parameters, in bfloat16.

To add a model of a supported family (currently Qwen2):

```bash
# 1. Where the weights are: read tensor names and shapes from the file headers.
python workflows/miniapps_workflow/inference_profiles/read_safetensors_headers.py \
  MODEL_DIR > SOURCE_DIR/safetensors-headers.json
cp MODEL_DIR/config.json SOURCE_DIR/

# 2. Build the profile.
python -m workflows.miniapps_workflow.inference_profile NAME SOURCE_DIR \
  workflows/miniapps_workflow/inference_profiles/NAME.json \
  --model ORG/MODEL --revision REVISION
```

Building fails if the model has a feature the decoder does not reproduce (tied
embeddings, another activation, sliding-window attention) or if the emulator's
tensors would differ from the model's in name or shape.

A profile's `measurements` section records comparisons with real inference of
that model. It is evidence, not an input.

## How it was checked against a real model

Reference: the real Qwen2.5-7B-Instruct weights run by the `transformers`
implementation in eager PyTorch, greedy decoding with the model's own cache. One
NVIDIA H200 NVL and eight cores of an Intel Xeon 6530P; bfloat16; batch size 1.

**Same computation.** The emulator can load a real model's weights
(`inference_profile.workload_from_model`). With them, its logits are bitwise
identical to the reference on the GPU and on the CPU at every step of a 32-token
generation, and it produces the same text. In float32 the largest difference is
3.5e-5 on logits spanning -29 to 42.

**Same operations.** For a request of 256 prompt and 32 generated tokens, both
make 6,304 `linear` calls, 896 attention calls, 896 SiLU and 1,824 RMSNorm
powers.

**Same cost, with random weights.**

| Request (prompt + generated tokens) | Device | Real model | Emulator | Emulator / real |
|---|---|---:|---:|---:|
| 256 + 32 | H200 | 0.322 s | 0.325 s | 1.01 |
| 256 + 32 | CPU, 8 cores | 5.96 s | 5.75 s | 0.96 |
| 2,059 + 217 | H200 | 2.28 s | 2.23 s | 0.98 |
| 2,059 + 217 | CPU, 8 cores | 45.78 s | 44.30 s | 0.97 |

Medians of 4 to 10 timed requests per cell, in two interleaved rounds. The
reference itself varied by up to about 6% between sessions.

**Memory.** GPU peak allocation is within 1% of the real model (14.73 against
14.65 GiB on the long request). CPU peak memory is 7–9% higher with random
weights, because generating them leaves temporaries in the peak.

**The workflow.** The full mini-app was run on one node with the concurrent
backend and the real simulation kernels, with inference on the CPU and then on
the GPU. Both runs launched four simulations, made two inference requests,
finished one simulation and stopped three, with identical decisions and complete
telemetry. That run used a small hand-chosen model, not the Qwen profile.

## Limits

- One model, one node, two request shapes, batch size 1. Batched requests have
  not been compared with a real model.
- The reference is eager PyTorch. Batching, fused kernels, paged caches and
  memory reservation of a serving engine are not represented.
- The full mini-app has not been run with the Qwen profile, and not on the
  Dragon backend.
- On the GPU, large bfloat16 runs can generate different token IDs from one run
  to the next. Timing and steering are unaffected, since neither depends on the
  IDs.
- Scores are random. Nothing here says whether an LLM would steer well.

## Tests

```bash
pytest tests/unit/test_inference_compute.py tests/unit/test_inference_profile.py \
       tests/unit/test_llm_compute_backend.py tests/unit/test_inference_workflow.py \
       tests/unit/test_inference_execution.py
```

Tests that need PyTorch, `transformers` or AsyncFlow skip when it is missing. GPU
tests run only with `DDSIM_TEST_CUDA=1`. With `transformers` installed,
`test_emulator_program_reproduces_a_transformers_qwen2_model` checks that the
emulator computes the same logits as a Qwen2 model given its weights.
