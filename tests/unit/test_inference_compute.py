"""Tiny CPU-only functional checks; CUDA tests are explicitly opt-in."""

import os
from dataclasses import replace

import pytest

from workflows.miniapps_workflow.inference_compute import (
    FIDELITY,
    InferenceSpec,
    InferenceWorkload,
)


@pytest.fixture
def spec():
    return InferenceSpec(
        hidden_size=16,
        intermediate_size=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=32,
        max_batch_size=2,
        max_prompt_tokens=16,
        max_new_tokens=4,
        max_tensor_bytes=1024 * 1024,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="optional inference-compute dependency")


@pytest.mark.parametrize(
    "settings",
    [
        {"hidden_size": 15},
        {"num_heads": 3},
        {"num_kv_heads": 3},
        {"hidden_size": 12},
        {"num_layers": 0},
        {"max_new_tokens": -1},
        {"dtype": "int8"},
        {"seed": -1},
        {"max_batch_size": True},
        {"max_tensor_bytes": 1},
    ],
)
def test_invalid_spec_fails_before_allocation(settings):
    with pytest.raises(ValueError):
        InferenceSpec(**settings)


def test_compute_is_deterministic_inference_with_real_cache(spec, torch):
    rng_before = torch.get_rng_state().clone()
    model = InferenceWorkload(spec)
    assert torch.equal(torch.get_rng_state(), rng_before)
    tensors = model.parameters()
    assert all(not x.requires_grad for x in tensors)
    assert sum(x.numel() for x in tensors) == spec.parameter_count
    first = model.run(2, 8, 3)
    second = model.run(2, 8, 3)
    assert first["token_ids"] == second["token_ids"]
    assert len(first["token_ids"]) == 2
    assert all(len(row) == 3 for row in first["token_ids"])
    assert first["prefill_calls"] == 1
    assert first["decode_calls"] == 2
    assert first["forward_calls"] == 3
    assert first["cache_tokens"] == 10
    assert first["kv_cache_bytes"] == 2 * 2 * 2 * 2 * 10 * 4 * 4
    assert first["parameter_bytes"] == sum(
        x.numel() * x.element_size() for x in tensors
    )
    assert first["elapsed_s"] > 0
    assert first["device"] == "cpu"
    assert first["fidelity"] == FIDELITY == "synthetic_decoder_not_measured_trace"
    assert all(x.grad is None for x in tensors)


def test_observer_sees_every_forward_pass_without_changing_the_result(spec, torch):
    model = InferenceWorkload(spec)
    seen = []

    def observe(tokens, caches, offset, logits):
        seen.append((tuple(tokens.shape), offset, tuple(logits.shape)))
        assert not torch.is_grad_enabled()
        assert len(caches) == spec.num_layers

    observed = model.run(2, 8, 3, observer=observe)
    assert seen == [((2, 8), 0, (2, 32)), ((2, 1), 8, (2, 32)), ((2, 1), 9, (2, 32))]
    assert observed["token_ids"] == model.run(2, 8, 3)["token_ids"]


def test_cached_decode_matches_causal_full_prefix(spec, torch):
    model = InferenceWorkload(spec)
    with torch.inference_mode():
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        expected = model.forward(tokens, model.new_caches(1, 5), 0)
        state = model.new_caches(1, 5)
        model.forward(tokens[:, :4], state, 0)
        actual = model.forward(tokens[:, 4:], state, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "args", [(0, 8, 2), (3, 8, 2), (1, 17, 2), (1, 8, 5), (1, 8, 0), (1, 2.5, 1)]
)
def test_request_bounds(spec, torch, args):
    model = InferenceWorkload(spec)
    with pytest.raises(ValueError):
        model.run(*args)


def test_one_generated_token_needs_only_prefill(spec, torch):
    result = InferenceWorkload(spec).run(1, 8, 1)
    assert result["decode_calls"] == 0
    assert result["cache_tokens"] == 8


def test_cpu_bfloat16_is_explicit_not_silently_substituted(spec, torch):
    result = InferenceWorkload(replace(spec, dtype="bfloat16")).run(1, 4, 2)
    assert result["dtype"] == "bfloat16"
    assert result["parameter_bytes"] == spec.parameter_count * 2


def test_cuda_request_never_falls_back_to_cpu(spec, torch, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        InferenceWorkload(spec, "cuda")


@pytest.mark.skipif(
    os.environ.get("DDSIM_TEST_CUDA") != "1",
    reason="requires explicit DDSIM_TEST_CUDA=1 hardware approval",
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_same_program_and_dimensions_on_cuda(spec, torch, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spec = replace(spec, dtype=dtype)
    cpu = InferenceWorkload(spec, "cpu")
    gpu = InferenceWorkload(spec, "cuda")
    for a_weights, b_weights in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(a_weights, b_weights.cpu(), rtol=0, atol=0)

    def logits_with_cache_check(model):
        with torch.inference_mode():
            tokens = torch.tensor([[1, 2, 3, 4]], device=model.device)
            full = model.forward(tokens, model.new_caches(1, 4), 0)
            cache = model.new_caches(1, 4)
            model.forward(tokens[:, :3], cache, 0)
            incremental = model.forward(tokens[:, 3:], cache, 3)
            tolerance = 0.05 if dtype == "bfloat16" else 1e-4
            torch.testing.assert_close(
                incremental, full, rtol=tolerance, atol=tolerance
            )
            assert torch.isfinite(full).all()
            return full.cpu()

    a_logits, b_logits = logits_with_cache_check(cpu), logits_with_cache_check(gpu)
    tolerance = 0.05 if dtype == "bfloat16" else 1e-4
    torch.testing.assert_close(a_logits, b_logits, rtol=tolerance, atol=tolerance)
    a, b = cpu.run(1, 4, 2), gpu.run(1, 4, 2)
    for key in (
        "batch_size",
        "prompt_tokens",
        "new_tokens",
        "forward_calls",
        "cache_tokens",
        "parameter_bytes",
        "kv_cache_bytes",
        "dtype",
    ):
        assert a[key] == b[key]
    # Floating-point implementation differences can change greedy token choices.
    # Neither a timing ratio nor bitwise output equivalence is asserted.
