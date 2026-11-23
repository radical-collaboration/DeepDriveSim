"""Inference profiles: provenance, the real tensor inventory, and emulator use."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from workflows.miniapps_workflow import inference_profile as profiles
from workflows.miniapps_workflow.inference_compute import (
    InferenceSpec,
    InferenceWorkload,
)

QWEN = "qwen2.5-7b-instruct"
LIMITS = {
    "max_batch_size": 1,
    "max_prompt_tokens": 16,
    "max_new_tokens": 4,
    "max_tensor_bytes": 2**62,
}


def write_source(root, *, bias=True, **config_changes):
    """Write a tiny Qwen2-style source directory; return its path."""
    hidden, mlp, layers, heads, kv_heads, vocab = 16, 24, 2, 4, 2, 32
    kv = kv_heads * hidden // heads
    config = {
        "model_type": "qwen2",
        "hidden_act": "silu",
        "hidden_size": hidden,
        "intermediate_size": mlp,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "vocab_size": vocab,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        **config_changes,
    }
    shapes = {
        "lm_head.weight": [vocab, hidden],
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        shapes.update(
            {
                prefix + "self_attn.q_proj.weight": [hidden, hidden],
                prefix + "self_attn.k_proj.weight": [kv, hidden],
                prefix + "self_attn.v_proj.weight": [kv, hidden],
                prefix + "self_attn.o_proj.weight": [hidden, hidden],
                prefix + "mlp.gate_proj.weight": [mlp, hidden],
                prefix + "mlp.up_proj.weight": [mlp, hidden],
                prefix + "mlp.down_proj.weight": [hidden, mlp],
                prefix + "input_layernorm.weight": [hidden],
                prefix + "post_attention_layernorm.weight": [hidden],
            }
        )
        if bias:
            shapes.update(
                {
                    prefix + "self_attn.q_proj.bias": [hidden],
                    prefix + "self_attn.k_proj.bias": [kv],
                    prefix + "self_attn.v_proj.bias": [kv],
                }
            )
    headers = {
        "files": {"model-00001-of-00001.safetensors": {"header_sha256": "0" * 64}},
        "tensors": {
            name: {"dtype": "F32", "shape": shape} for name, shape in shapes.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / profiles.CONFIG_FILE).write_text(json.dumps(config))
    (root / profiles.HEADERS_FILE).write_text(json.dumps(headers))
    return root


@pytest.fixture
def tiny(tmp_path):
    source = write_source(tmp_path / "source")
    return profiles.build_profile("tiny", source, {"model": "test/tiny"})


# --- the shipped Qwen profile ---
def test_shipped_profile_rebuilds_from_its_recorded_sources():
    shipped = profiles.load_profile(QWEN)
    source_dir = profiles.PROFILE_DIR / "sources" / QWEN
    recorded = {key: shipped["source"][key] for key in ("model", "revision", "note")}
    # Everything the emulator consumes is rebuilt; measurements are added evidence.
    assert profiles.build_profile(QWEN, source_dir, recorded) == {
        **shipped,
        "measurements": [],
    }


def test_shipped_profile_measurements_are_well_formed():
    shipped = profiles.load_profile(QWEN)  # load_profile checks the required keys
    for entry in shipped["measurements"]:
        assert entry["kind"] == "reference_comparison"
        for request in entry["requests"]:
            assert request["reference_s"] > 0 and request["emulator_s"] > 0
            assert request["device"] in ("cuda", "cpu")


def test_emulator_allocates_exactly_the_real_qwen_tensors():
    profile = profiles.load_profile(QWEN)
    profiles.check_inventory(profile)  # shapes and counts; nothing is allocated
    spec = profiles.spec_from_profile(profile, **LIMITS)
    assert spec.parameter_count == profile["tensors"]["parameters"] == 7_615_616_512
    assert len(spec.tensor_shapes()) == profile["tensors"]["count"] == 339
    assert (spec.dtype, spec.attention_bias, spec.rope_theta) == (
        "bfloat16",
        True,
        1_000_000.0,
    )
    assert (spec.num_layers, spec.hidden_size, spec.num_heads, spec.num_kv_heads) == (
        28,
        3584,
        28,
        4,
    )


def test_real_size_spec_is_refused_without_an_adequate_tensor_budget():
    profile = profiles.load_profile(QWEN)
    with pytest.raises(ValueError, match="max_tensor_bytes"):
        profiles.spec_from_profile(profile, **{**LIMITS, "max_tensor_bytes": 2**30})


# --- building and loading ---
def test_profile_records_provenance_and_what_it_does_not_capture(tiny, tmp_path):
    assert tiny["schema"] == profiles.SCHEMA and tiny["level"] == "architecture"
    assert set(tiny["source"]["files_sha256"]) == {
        profiles.CONFIG_FILE,
        profiles.HEADERS_FILE,
    }
    assert tiny["measurements"] == [] and tiny["not_captured"]
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(tiny))
    assert profiles.load_profile(path) == tiny
    assert profiles.profile_sha256(tiny) == profiles.profile_sha256(
        json.loads(path.read_text())
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "llama"},
        {"hidden_act": "gelu"},
        {"tie_word_embeddings": True},
        {"use_sliding_window": True},
        {"rms_norm_eps": 1e-5},
    ],
)
def test_models_the_emulator_cannot_mirror_are_rejected(tmp_path, changes):
    source = write_source(tmp_path / "source", **changes)
    with pytest.raises(ValueError, match="unsupported model"):
        profiles.build_profile("bad", source, {})


def test_inventory_mismatch_is_detected(tmp_path):
    source = write_source(tmp_path / "source")
    headers = json.loads((source / profiles.HEADERS_FILE).read_text())
    headers["tensors"]["model.extra.weight"] = {"dtype": "F32", "shape": [7, 7]}
    (source / profiles.HEADERS_FILE).write_text(json.dumps(headers))
    with pytest.raises(ValueError, match="inventory differs"):
        profiles.build_profile("bad", source, {})


def test_load_profile_rejects_other_files(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something/else"}))
    with pytest.raises(ValueError, match="schema"):
        profiles.load_profile(path)
    with pytest.raises(FileNotFoundError):
        profiles.load_profile("no-such-profile")


def test_load_profile_rejects_malformed_measurements(tiny, tmp_path):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps({**tiny, "measurements": [{"kind": "x"}]}))
    with pytest.raises(ValueError, match="measurement lacks"):
        profiles.load_profile(path)


# --- profile -> spec ---
def test_spec_takes_architecture_from_profile_and_limits_from_caller(tiny):
    spec = profiles.spec_from_profile(tiny, seed=3, **LIMITS)
    assert (spec.hidden_size, spec.num_layers, spec.vocab_size) == (16, 2, 32)
    assert spec.attention_bias is True and spec.rope_theta == 1_000_000.0
    assert spec.seed == 3 and spec.max_prompt_tokens == 16
    with pytest.raises(ValueError, match="cannot override"):
        profiles.spec_from_profile(tiny, hidden_size=8, **LIMITS)
    with pytest.raises(ValueError, match="missing"):
        profiles.spec_from_profile(tiny, max_batch_size=1)


def test_profile_without_bias_maps_to_bias_free_spec(tmp_path):
    source = write_source(tmp_path / "source", bias=False)
    profile = profiles.build_profile("nobias", source, {})
    assert profiles.spec_from_profile(profile, **LIMITS).attention_bias is False


@pytest.mark.parametrize("theta", [1, 0, -5, float("nan"), True, "1e6"])
def test_spec_rejects_unusable_rope_theta(theta):
    with pytest.raises(ValueError, match="rope_theta"):
        InferenceSpec(rope_theta=theta)


def test_spec_rejects_non_boolean_attention_bias():
    with pytest.raises(ValueError, match="attention_bias"):
        InferenceSpec(attention_bias=1)


# --- the emulator running a profile (tiny, CPU) ---
def test_workload_allocates_the_profiled_tensors_and_uses_the_biases(tiny):
    torch = pytest.importorskip("torch")
    spec = profiles.spec_from_profile(tiny, **{**LIMITS, "max_tensor_bytes": 2**20})
    model = InferenceWorkload(spec)
    allocated = [tuple(tensor.shape) for tensor in model.parameters()]
    assert allocated == [shape for _, shape in spec.tensor_shapes()]
    assert (
        sum(tensor.numel() for tensor in model.parameters())
        == tiny["tensors"]["parameters"]
    )
    with torch.inference_mode():
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        full = model.forward(tokens, model.new_caches(1, 5), 0)
        caches = model.new_caches(1, 5)
        model.forward(tokens[:, :4], caches, 0)
        incremental = model.forward(tokens[:, 4:], caches, 4)
        torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-5)
        plain = InferenceWorkload(replace(spec, attention_bias=False))
        without = plain.forward(tokens, plain.new_caches(1, 5), 0)
    assert not torch.equal(full, without)  # the bias terms take part in the result
    result = model.run(1, 8, 3)
    assert result["parameter_bytes"] == tiny["tensors"]["parameters"] * 4


async def test_backend_builds_its_workload_from_a_profile(tiny, tmp_path, monkeypatch):
    pytest.importorskip("radical.asyncflow")
    from types import SimpleNamespace

    from workflows.miniapps_workflow import llm_compute_backend as module

    class FakeWorkload:
        def __init__(self, spec, device="cpu"):
            self.spec, self.device = spec, SimpleNamespace(type=device)

    monkeypatch.setattr(module, "InferenceWorkload", FakeWorkload)
    path = Path(tmp_path, "tiny.json")
    path.write_text(json.dumps(tiny))
    backend = await module.ComputeLLMBackend.create(
        {"profile": str(path), "spec": LIMITS, "new_tokens": 2}
    )
    try:
        description = backend.description()
        assert description["profile"] == {
            "name": "tiny",
            "sha256": profiles.profile_sha256(tiny),
        }
        assert description["spec"]["attention_bias"] is True
        assert description["spec"]["hidden_size"] == 16
    finally:
        await backend.shutdown()
    with pytest.raises(ValueError, match="cannot override"):
        await module.ComputeLLMBackend.create(
            {"profile": str(path), "spec": {**LIMITS, "num_layers": 1}}
        )


# --- real weights through the emulator's tensor program ---
def write_safetensors(path, tensors):
    """Minimal safetensors writer: 8-byte header length, JSON header, raw data."""
    import struct

    names = {"torch.float32": "F32", "torch.bfloat16": "BF16", "torch.float16": "F16"}
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(-1).cpu().numpy().tobytes()
        header[name] = {
            "dtype": names[str(tensor.dtype)],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def model_weights(spec, torch):
    """Random stand-ins for a model's weights, under their Qwen2 names."""
    generator = torch.Generator().manual_seed(11)
    names = profiles.model_tensor_names(spec)
    return {
        names[name]: torch.randn(shape, generator=generator) * 0.1
        for name, shape in spec.tensor_shapes()
    }


def test_real_weights_load_by_name_and_drive_the_same_program(tiny, tmp_path):
    torch = pytest.importorskip("torch")
    limits = {**LIMITS, "max_tensor_bytes": 2**20}
    spec = profiles.spec_from_profile(tiny, **limits)
    real = model_weights(spec, torch)
    items = list(real.items())
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_safetensors(model_dir / "model-00001-of-00002.safetensors", dict(items[:9]))
    write_safetensors(model_dir / "model-00002-of-00002.safetensors", dict(items[9:]))

    loaded = profiles.read_model_weights(model_dir, spec)
    names = profiles.model_tensor_names(spec)
    assert set(loaded) == {name for name, _ in spec.tensor_shapes()}
    assert all(torch.equal(loaded[name], real[names[name]]) for name in loaded)

    workload = profiles.workload_from_model(tiny, model_dir, "cpu", **limits)
    assert torch.equal(workload.layers[1]["q_bias"], real[names["layers.1.q_bias"]])
    reference = InferenceWorkload(spec, weights=loaded)
    with torch.inference_mode():
        tokens = torch.tensor([[3, 1, 4, 1, 5]])
        a = workload.forward(tokens, workload.new_caches(1, 5), 0)
        b = reference.forward(tokens, reference.new_caches(1, 5), 0)
        random = InferenceWorkload(spec)
        c = random.forward(tokens, random.new_caches(1, 5), 0)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_supplied_weights_must_match_the_spec_exactly(tiny):
    torch = pytest.importorskip("torch")
    spec = profiles.spec_from_profile(tiny, **{**LIMITS, "max_tensor_bytes": 2**20})
    good = {name: torch.zeros(shape) for name, shape in spec.tensor_shapes()}
    InferenceWorkload(spec, weights=good)
    with pytest.raises(ValueError, match="missing"):
        InferenceWorkload(spec, weights={k: v for k, v in good.items() if k != "head"})
    with pytest.raises(ValueError, match="unexpected"):
        InferenceWorkload(spec, weights={**good, "extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="shape"):
        InferenceWorkload(spec, weights={**good, "head": torch.zeros(2, 2)})


def test_missing_model_tensor_is_reported(tiny, tmp_path):
    torch = pytest.importorskip("torch")
    spec = profiles.spec_from_profile(tiny, **{**LIMITS, "max_tensor_bytes": 2**20})
    real = model_weights(spec, torch)
    real.pop("lm_head.weight")
    write_safetensors(tmp_path / "model-00001-of-00001.safetensors", real)
    with pytest.raises(ValueError, match=r"no tensor lm_head\.weight"):
        profiles.read_model_weights(tmp_path, spec)


def test_emulator_program_reproduces_a_transformers_qwen2_model(tmp_path):
    """With a real model's weights, the emulator computes that model's logits."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import importlib.util

    config = transformers.Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=80,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(5)
    model = transformers.Qwen2ForCausalLM(config).eval()
    source = tmp_path / "source"
    source.mkdir()
    write_safetensors(source / "model-00001-of-00001.safetensors", model.state_dict())
    (source / profiles.CONFIG_FILE).write_text(json.dumps(config.to_dict()))
    script = profiles.PROFILE_DIR / "read_safetensors_headers.py"
    module_spec = importlib.util.spec_from_file_location("read_headers", script)
    reader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reader)
    (source / profiles.HEADERS_FILE).write_text(json.dumps(reader.read_headers(source)))

    profile = profiles.build_profile("tiny-qwen2", source, {"model": "random"})
    workload = profiles.workload_from_model(
        profile,
        source,
        "cpu",
        max_batch_size=2,
        max_prompt_tokens=16,
        max_new_tokens=8,
        max_tensor_bytes=2**24,
    )
    tokens = torch.tensor([[5, 17, 3, 42, 8, 96, 0, 23], [9, 9, 61, 2, 77, 30, 4, 11]])
    with torch.inference_mode():
        expected = model(tokens).logits[:, -1]
        actual = workload.forward(tokens, workload.new_caches(2, 8), 0)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        # Greedy decoding through each implementation's own KV cache. The model is
        # stepped directly: generate() would mask prompt positions equal to the pad
        # token, which is a different computation.
        caches, step, offset, ours = workload.new_caches(2, 13), tokens, 0, []
        past, reference_step, generated = None, tokens, []
        for _ in range(6):
            logits = workload.forward(step, caches, offset)
            output = model(reference_step, past_key_values=past, use_cache=True)
            past = output.past_key_values
            torch.testing.assert_close(
                logits, output.logits[:, -1], rtol=1e-4, atol=1e-5
            )
            ours.append(logits.argmax(-1))
            generated.append(output.logits[:, -1].argmax(-1))
            offset += step.shape[1]
            step = ours[-1][:, None]
            reference_step = generated[-1][:, None]
    assert torch.equal(torch.stack(ours, dim=1), torch.stack(generated, dim=1))
