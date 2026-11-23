"""Bounded per-device inference validation, with measured runtime and memory.

Run CPU and CUDA in separate processes with the same spec/request arguments.
This tests the synthetic decoder, not fidelity to a real model. No model download,
training, service-latency sleeps or GPU utilization targets are involved.

Linux only: it reads CPU affinity and peak RSS through POSIX interfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from workflows.miniapps_workflow import inference_compute
from workflows.miniapps_workflow.inference_compute import (
    FIDELITY,
    InferenceSpec,
    InferenceWorkload,
)
from workflows.miniapps_workflow.inference_profile import (
    load_profile,
    profile_sha256,
    spec_from_profile,
)

if TYPE_CHECKING:
    import torch

    from workflows.miniapps_workflow.inference_compute import LayerCache

SCHEMA = "ddsim.inference_compute.device_validation/1"
_KIB = 1024  # Linux reports ru_maxrss in KiB
_SUMMARY_KEYS = (
    "host",
    "device",
    "gpu_name",
    "median_s",
    "min_s",
    "max_s",
    "memory",
    "repeat_outputs_identical",
)


def positive_int(value: str) -> int:
    """Parse a command-line value that must be a positive integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file, to tie a result to the code that made it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _memory_counters(device: torch.device) -> dict[str, Any]:
    """Collect process and, for CUDA, device memory counters after the runs.

    Peak RSS includes the interpreter, libraries and weight initialization.
    """
    import torch

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory: dict[str, Any] = {"process_peak_rss_bytes": peak_kib * _KIB}
    if device.type != "cuda":
        return memory
    memory.update(
        {
            "cuda_allocated_bytes_after_runs": torch.cuda.memory_allocated(device),
            "cuda_reserved_bytes_after_runs": torch.cuda.memory_reserved(device),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    )
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        query = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        memory["nvidia_smi_current_process_rows"] = [
            row
            for row in query.stdout.splitlines()
            if row.split(",", 1)[0].strip() == str(os.getpid())
        ]
    return memory


def _checked_run(
    workload: InferenceWorkload, batch_size: int, prompt_tokens: int, new_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run once, untimed, recording and checking every forward pass.

    Returns:
        The run result and one record per forward pass (shapes, cache dtype,
        cache offset and gradient state).

    Raises:
        AssertionError: Logits are not finite, gradients are enabled, or the
            forward sequence or cache offsets are not those of inference.
    """
    import torch

    trace: list[dict[str, Any]] = []

    def observe(
        tokens: torch.Tensor,
        caches: list[LayerCache],
        offset: int,
        logits: torch.Tensor,
    ) -> None:
        if not torch.isfinite(logits).all().item():
            raise AssertionError("non-finite inference logits")
        trace.append(
            {
                "token_shape": list(tokens.shape),
                "offset": offset,
                "cache_shapes": [[list(k.shape), list(v.shape)] for k, v in caches],
                "cache_dtype": str(caches[0][0].dtype),
                "logit_shape": list(logits.shape),
                "grad_enabled": torch.is_grad_enabled(),
            }
        )

    result = workload.run(batch_size, prompt_tokens, new_tokens, observer=observe)
    if len(trace) != new_tokens or any(row["grad_enabled"] for row in trace):
        raise AssertionError("incorrect inference-only forward sequence")
    decode_offsets = range(prompt_tokens, prompt_tokens + new_tokens - 1)
    if [row["offset"] for row in trace] != [0, *decode_offsets]:
        raise AssertionError("incorrect KV-cache offsets")
    return result, trace


def run_validation(
    spec: InferenceSpec,
    device: str,
    *,
    batch_size: int,
    prompt_tokens: int,
    new_tokens: int,
    threads: int = 8,
    warmups: int = 2,
    repetitions: int = 5,
    profile: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure one device; the caller controls affinity and process lifetime.

    Args:
        spec: Workload dimensions; use the same spec on every device compared.
        device: ``cpu`` or ``cuda[:index]``; bare ``cuda`` is the current device.
        batch_size: Requests per batch.
        prompt_tokens: Prompt length.
        new_tokens: Tokens generated per request.
        threads: Torch intra-op threads.
        warmups: Untimed requests before measuring.
        repetitions: Timed requests.
        profile: Name and hash of the inference profile the spec came from, if
            any; recorded with the result.

    Returns:
        A JSON-serializable record of the environment, timings, memory counters,
        every run and the forward-pass trace.

    Raises:
        RuntimeError: CUDA was requested but is unavailable.
    """
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    cuda = target.type == "cuda"
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(target)
        torch.cuda.reset_peak_memory_stats(target)
    start = time.perf_counter()
    workload = InferenceWorkload(spec, target)
    initialization_s = time.perf_counter() - start
    request = (batch_size, prompt_tokens, new_tokens)
    for _ in range(warmups):
        workload.run(*request)
    runs = [workload.run(*request) for _ in range(repetitions)]
    # Memory is captured before the separate, untimed validation pass.
    memory = _memory_counters(target)
    checked, trace = _checked_run(workload, *request)

    times = [run["elapsed_s"] for run in runs]
    spec_json = json.dumps(asdict(spec), sort_keys=True)
    return {
        "schema": SCHEMA,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "device": str(workload.device),
        "gpu_name": torch.cuda.get_device_name(target) if cuda else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
        "source_sha256": _sha256(inference_compute.__file__),
        "harness_sha256": _sha256(__file__),
        "profile": profile,
        "spec": asdict(spec),
        "spec_sha256": hashlib.sha256(spec_json.encode()).hexdigest(),
        "request": {
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
        },
        "warmups": warmups,
        "repetitions": repetitions,
        "initialization_s": initialization_s,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "memory": memory,
        "runs": runs,
        "forward_trace": trace,
        "repeat_outputs_identical": all(
            run["token_ids"] == checked["token_ids"] for run in runs
        ),
        "fidelity": FIDELITY,
    }


def main() -> None:
    """Validate one device and write the result to a fresh JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec-json", help="JSON object of InferenceSpec fields")
    source.add_argument("--profile", help="inference profile name or path")
    parser.add_argument(
        "--max-tensor-gib",
        type=float,
        help="tensor budget in GiB; required with --profile",
    )
    parser.add_argument("--device", required=True, help="cpu or cuda[:index]")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--prompt-tokens", type=positive_int, default=256)
    parser.add_argument("--new-tokens", type=positive_int, default=32)
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--warmups", type=positive_int, default=2)
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a fresh result path")
    recorded = None
    if args.profile:
        if args.max_tensor_gib is None:
            parser.error("--profile needs --max-tensor-gib")
        profile = load_profile(args.profile)
        recorded = {"name": profile["name"], "sha256": profile_sha256(profile)}
        spec = spec_from_profile(
            profile,
            max_batch_size=args.batch_size,
            max_prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.new_tokens,
            max_tensor_bytes=int(args.max_tensor_gib * 2**30),
        )
    else:
        spec = InferenceSpec(**json.loads(Path(args.spec_json).read_text()))
    result = run_validation(
        spec,
        args.device,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        new_tokens=args.new_tokens,
        threads=args.threads,
        warmups=args.warmups,
        repetitions=args.repetitions,
        profile=recorded,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in _SUMMARY_KEYS}, indent=2))


if __name__ == "__main__":
    main()
