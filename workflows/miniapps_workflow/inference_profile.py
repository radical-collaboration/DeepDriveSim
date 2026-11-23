"""Stored inference profiles: what a real model is, as data the emulator loads.

A profile records a real model's architecture and tensor inventory, with the
provenance of the files it was derived from. The compute emulator turns it into
an :class:`InferenceSpec`, so the synthetic decoder has the real model's layer
structure, tensor shapes and precision instead of hand-chosen dimensions.

Schema ``ddsim.inference_profile/1`` holds the ``architecture`` level only: it is
derived from the model's ``config.json`` and the headers of its weight files,
without loading or running the model. Runtime measurements of the real model are
a separate level; ``measurements`` is empty until one has been captured.

Build a profile from source files with::

    python -m workflows.miniapps_workflow.inference_profile NAME SOURCE_DIR OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from workflows.miniapps_workflow.inference_compute import RMS_NORM_EPS, InferenceSpec

SCHEMA = "ddsim.inference_profile/1"
PROFILE_DIR = Path(__file__).resolve().parent / "inference_profiles"
CONFIG_FILE = "config.json"
HEADERS_FILE = "safetensors-headers.json"

#: Decoder families whose layer structure the emulator reproduces.
_SUPPORTED_MODEL_TYPES = ("qwen2",)
_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
#: Sizes a caller chooses per run; they are not properties of the model.
_BOUNDS = ("max_batch_size", "max_prompt_tokens", "max_new_tokens", "max_tensor_bytes")
_LAYER_INDEX = re.compile(r"\.layers\.\d+\.")

#: Properties of real inference that an architecture-level profile cannot give.
NOT_CAPTURED = (
    "operator-level timing of the real model",
    "measured memory of a real inference process",
    "serving-engine behavior: batching, kernel fusion, paged KV cache",
    "tokenizer and generated content",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unsupported model for the compute emulator: {message}")


def _architecture(config: dict[str, Any], tensors: dict[str, Any]) -> dict[str, Any]:
    """Return the architecture section; reject models the emulator cannot mirror."""
    _require(
        config.get("model_type") in _SUPPORTED_MODEL_TYPES,
        f"model_type={config.get('model_type')!r}",
    )
    _require(config.get("hidden_act") == "silu", "hidden_act is not silu")
    _require(not config.get("tie_word_embeddings"), "tied input/output embeddings")
    _require(not config.get("use_sliding_window"), "sliding-window attention")
    _require(
        config.get("rms_norm_eps") == RMS_NORM_EPS,
        f"rms_norm_eps={config.get('rms_norm_eps')}",
    )
    dtypes = {info["dtype"] for info in tensors.values()}
    _require(len(dtypes) == 1, f"mixed tensor dtypes {sorted(dtypes)}")
    (stored,) = dtypes
    _require(stored in _SAFETENSORS_DTYPES, f"tensor dtype {stored}")
    return {
        "family": "decoder",
        "model_type": config["model_type"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_layers": config["num_hidden_layers"],
        "num_heads": config["num_attention_heads"],
        "num_kv_heads": config["num_key_value_heads"],
        "vocab_size": config["vocab_size"],
        "dtype": _SAFETENSORS_DTYPES[stored],
        "attention_bias": any(name.endswith("q_proj.bias") for name in tensors),
        "rope_theta": float(config["rope_theta"]),
        "rms_norm_eps": config["rms_norm_eps"],
        "max_position_embeddings": config["max_position_embeddings"],
    }


def _tensor_summary(tensors: dict[str, Any]) -> dict[str, Any]:
    """Summarize the real tensor inventory by kind, with per-layer names folded."""
    kinds: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        kind = _LAYER_INDEX.sub(".layers.N.", name)
        entry = kinds.setdefault(
            kind, {"name": kind, "count": 0, "shape": tensors[name]["shape"]}
        )
        _require(entry["shape"] == tensors[name]["shape"], f"{kind} shapes differ")
        entry["count"] += 1
    return {
        "count": len(tensors),
        "parameters": sum(math.prod(info["shape"]) for info in tensors.values()),
        "kinds": [kinds[kind] for kind in sorted(kinds)],
    }


def build_profile(
    name: str, source_dir: str | Path, source: dict[str, Any]
) -> dict[str, Any]:
    """Build an architecture-level profile from a model's config and weight headers.

    Args:
        name: Profile name, also its file stem.
        source_dir: Directory holding ``config.json`` and
            ``safetensors-headers.json`` (tensor name, dtype and shape for every
            tensor, read from the weight files' headers).
        source: Where the files came from: model ID, revision and capture notes.

    Raises:
        ValueError: The model has a feature the emulator does not reproduce, or
            the emulator's tensor inventory would differ from the real one.
    """
    root = Path(source_dir)
    config = json.loads((root / CONFIG_FILE).read_text())
    headers = json.loads((root / HEADERS_FILE).read_text())
    tensors = headers["tensors"]
    profile = {
        "schema": SCHEMA,
        "name": name,
        "level": "architecture",
        "source": {
            **source,
            "files_sha256": {
                CONFIG_FILE: _sha256(root / CONFIG_FILE),
                HEADERS_FILE: _sha256(root / HEADERS_FILE),
            },
            "weight_header_sha256": {
                file: info["header_sha256"]
                for file, info in sorted(headers["files"].items())
            },
        },
        "architecture": _architecture(config, tensors),
        "tensors": _tensor_summary(tensors),
        "measurements": [],
        "not_captured": list(NOT_CAPTURED),
    }
    check_inventory(profile)
    return profile


def load_profile(name_or_path: str | Path) -> dict[str, Any]:
    """Load a profile by packaged name (``qwen2.5-7b-instruct``) or by path.

    Raises:
        FileNotFoundError: No such profile.
        ValueError: The file is not a supported profile.
    """
    path = Path(name_or_path)
    if path.suffix != ".json" and len(path.parts) == 1:
        path = PROFILE_DIR / f"{path.name}.json"
    profile = json.loads(path.read_text())
    if profile.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}")
    for section in ("name", "source", "architecture", "tensors"):
        if section not in profile:
            raise ValueError(f"{path}: profile has no {section!r} section")
    return profile


def profile_sha256(profile: dict[str, Any]) -> str:
    """Stable hash of a profile's content, recorded with every run that uses it."""
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def spec_from_profile(profile: dict[str, Any], **bounds: Any) -> InferenceSpec:
    """Return the emulator spec for a profile's architecture.

    Args:
        profile: A loaded profile.
        **bounds: The per-run limits ``max_batch_size``, ``max_prompt_tokens``,
            ``max_new_tokens`` and ``max_tensor_bytes`` (all required: a real
            model is large, so the tensor budget must be stated), and optionally
            ``seed``. Architecture fields cannot be overridden.

    Raises:
        ValueError: A limit is missing, or an architecture field was supplied.
    """
    extra = set(bounds) - {*_BOUNDS, "seed"}
    if extra:
        raise ValueError(
            f"a profile fixes the architecture; cannot override {sorted(extra)}"
        )
    missing = [key for key in _BOUNDS if key not in bounds]
    if missing:
        raise ValueError(f"a profile needs explicit limits: missing {missing}")
    arch = profile["architecture"]
    return InferenceSpec(
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        num_kv_heads=arch["num_kv_heads"],
        vocab_size=arch["vocab_size"],
        dtype=arch["dtype"],
        attention_bias=arch["attention_bias"],
        rope_theta=arch["rope_theta"],
        **bounds,
    )


def check_inventory(profile: dict[str, Any]) -> None:
    """Check that the emulator would allocate exactly the real model's tensors.

    Compares shapes and counts only; nothing is allocated, so this works at any
    model size.

    Raises:
        ValueError: The inventories differ.
    """
    # Bounds do not affect the weight inventory; these are only placeholders.
    spec = spec_from_profile(
        profile,
        max_batch_size=1,
        max_prompt_tokens=1,
        max_new_tokens=1,
        max_tensor_bytes=2**62,
    )
    emulated = Counter(shape for _, shape in spec.tensor_shapes())
    real: Counter[tuple[int, ...]] = Counter()
    for kind in profile["tensors"]["kinds"]:
        real[tuple(kind["shape"])] += kind["count"]
    if emulated != real or spec.parameter_count != profile["tensors"]["parameters"]:
        raise ValueError(
            "emulator tensor inventory differs from the real model: "
            f"only emulated {dict(emulated - real)}, only real {dict(real - emulated)}"
        )


def main() -> None:
    """Build a profile from a source directory and write it to a new file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("name")
    parser.add_argument("source_dir", help="holds config.json and the headers file")
    parser.add_argument("output", help="profile JSON to create; must not exist")
    parser.add_argument("--model", required=True, help="model ID, e.g. org/name")
    parser.add_argument("--revision", required=True, help="model snapshot revision")
    parser.add_argument("--note", default="", help="how the files were obtained")
    args = parser.parse_args()
    source = {"model": args.model, "revision": args.revision, "note": args.note}
    profile = build_profile(args.name, args.source_dir, source)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
