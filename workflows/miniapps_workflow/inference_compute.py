"""Portable, synthetic decoder inference; not a measured Qwen trace.

The same tensor program executes on CPU/CUDA: embeddings, RMSNorm, rotary
positions, grouped-query causal attention, a KV cache, gated SiLU MLPs and a
vocabulary projection. Weights are random, fixed and never trained. Device
execution, not a latency model or utilization target, determines runtime.

Keep this module independent of AsyncFlow. Importing it does not import torch or
initialize CUDA (torch is an optional dependency, imported when a workload is
built). Defaults are deliberately tiny; no full-model preset is implied.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

    #: One layer's (keys, values) cache tensors.
    LayerCache = tuple[torch.Tensor, torch.Tensor]
    #: Called after every forward pass: (tokens, caches, offset, logits).
    ForwardObserver = Callable[
        [torch.Tensor, list[LayerCache], int, torch.Tensor], None
    ]

#: Recorded with every result so synthetic runs are never mistaken for a trace.
FIDELITY = "synthetic_decoder_not_measured_trace"

_DTYPES = ("float32", "float16", "bfloat16")
_FLOAT32_BYTES = 4
#: Epsilon of the RMSNorm layers.
RMS_NORM_EPS = 1e-6
#: Spec fields that are not positive integer sizes.
_NON_COUNT_FIELDS = frozenset(("dtype", "seed", "attention_bias", "rope_theta"))


def _is_int(value: object) -> bool:
    """True for a real ``int``; ``bool`` is rejected although it subclasses int."""
    return type(value) is int


@dataclass(frozen=True)
class InferenceSpec:
    """Dimensions, precision and bounds of one synthetic decoder workload.

    All sizes are validated at construction, before any tensor is allocated.
    ``max_tensor_bytes`` guards the estimated tensor storage only; it is not a
    limit on process RSS, allocator overhead or library workspaces.
    """

    hidden_size: int = 64
    intermediate_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    vocab_size: int = 128
    max_batch_size: int = 4
    max_prompt_tokens: int = 512
    max_new_tokens: int = 32
    dtype: str = "float32"
    seed: int = 0
    max_tensor_bytes: int = 256 * 1024 * 1024
    #: Add bias terms to the query/key/value projections, as Qwen2 does.
    attention_bias: bool = False
    #: Base of the rotary position frequencies.
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        """Validate every field; raise ``ValueError`` for an unusable spec."""
        for name, value in asdict(self).items():
            if name in _NON_COUNT_FIELDS:
                continue
            if not _is_int(value) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if not isinstance(self.attention_bias, bool):
            raise ValueError("attention_bias must be true or false")
        theta = self.rope_theta
        if (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float))
            or not math.isfinite(theta)
            or theta <= 1
        ):
            raise ValueError("rope_theta must be a finite number greater than 1")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("rotary positions require an even head dimension")
        self.check_storage(
            self.max_batch_size, self.max_prompt_tokens, self.max_new_tokens
        )

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.hidden_size // self.num_heads

    @property
    def element_bytes(self) -> int:
        """Bytes per stored scalar for the configured dtype."""
        return _FLOAT32_BYTES if self.dtype == "float32" else 2

    @property
    def parameter_count(self) -> int:
        """Number of weights, summed over :meth:`tensor_shapes`."""
        return sum(math.prod(shape) for _, shape in self.tensor_shapes())

    def tensor_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        """Name and shape of every weight tensor, in allocation order.

        This is the inventory an :class:`InferenceWorkload` allocates, computed
        without allocating anything, so it can be checked against a real
        model's tensor list at any size.
        """
        h, m = self.hidden_size, self.intermediate_size
        kv = self.num_kv_heads * self.head_dim
        shapes: list[tuple[str, tuple[int, ...]]] = [
            ("embedding", (self.vocab_size, h)),
            ("head", (self.vocab_size, h)),
            ("final_norm", (h,)),
        ]
        per_layer: list[tuple[str, tuple[int, ...]]] = [
            ("q", (h, h)),
            ("k", (kv, h)),
            ("v", (kv, h)),
            ("o", (h, h)),
            ("gate", (m, h)),
            ("up", (m, h)),
            ("down", (h, m)),
            ("norm1", (h,)),
            ("norm2", (h,)),
        ]
        if self.attention_bias:
            per_layer += [("q_bias", (h,)), ("k_bias", (kv,)), ("v_bias", (kv,))]
        for layer in range(self.num_layers):
            shapes += [(f"layers.{layer}.{name}", shape) for name, shape in per_layer]
        return shapes

    def cache_tokens(self, prompt_tokens: int, new_tokens: int) -> int:
        """Positions held in the KV cache for one request.

        The last generated token is returned, not fed through another decode
        step, so it never enters the cache.
        """
        return prompt_tokens + new_tokens - 1

    def kv_bytes(self, batch_size: int, prompt_tokens: int, new_tokens: int) -> int:
        """Bytes of key/value cache storage for one batch."""
        capacity = self.cache_tokens(prompt_tokens, new_tokens)
        elements = (
            2 * self.num_layers * batch_size * self.num_kv_heads * capacity
        ) * self.head_dim
        return elements * self.element_bytes

    def check_storage(
        self, batch_size: int, prompt_tokens: int, new_tokens: int
    ) -> None:
        """Raise ``ValueError`` if the estimated tensors exceed ``max_tensor_bytes``.

        The estimate covers weights, the KV cache and a conservative allowance
        for eager attention and activation temporaries. It is not a promise that
        an arbitrary allocator or backend fits within that number.
        """
        known = self.parameter_count * self.element_bytes + self.kv_bytes(
            batch_size, prompt_tokens, new_tokens
        )
        h, m = self.hidden_size, self.intermediate_size
        seq = self.cache_tokens(prompt_tokens, new_tokens)
        scratch = batch_size * (
            self.num_heads * seq * seq * _FLOAT32_BYTES
            + seq * (12 * h + 4 * m) * _FLOAT32_BYTES
            + self.vocab_size * _FLOAT32_BYTES
        )
        if known + scratch > self.max_tensor_bytes:
            raise ValueError(
                f"estimated tensor budget {known + scratch} exceeds "
                f"max_tensor_bytes={self.max_tensor_bytes}; use smaller dimensions"
            )


class InferenceWorkload:
    """Synchronous inference-only tensor workload on one device.

    Not thread-safe: a single owner must serialize calls to :meth:`run`.
    A CUDA request never falls back to CPU, and the configured dtype is never
    substituted; unsupported combinations raise instead.
    """

    def __init__(self, spec: InferenceSpec, device: str | torch.device = "cpu") -> None:
        """Build the fixed synthetic weights for ``spec`` on ``device``.

        Raises:
            ValueError: ``device`` is neither ``cpu`` nor ``cuda[:index]``.
            RuntimeError: CUDA was requested but is unavailable.
        """
        # The spec has already validated every bound; torch is imported only now
        # so the module itself stays importable without the optional dependency.
        import torch

        self.spec = spec
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda[:index]")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        self.dtype = getattr(torch, spec.dtype)
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)

        def weight(rows: int, cols: int) -> torch.Tensor:
            # Drawn on the CPU from a private generator: the same values on any
            # device, and no mutation of the global RNG state.
            initial = torch.randn(
                rows, cols, generator=generator, dtype=torch.float32
            ) / math.sqrt(cols)
            return initial.to(device=self.device, dtype=self.dtype)

        def norm() -> torch.Tensor:
            return torch.ones(spec.hidden_size, device=self.device, dtype=self.dtype)

        def bias(size: int) -> torch.Tensor:
            initial = torch.randn(size, generator=generator, dtype=torch.float32)
            return (initial * 0.02).to(device=self.device, dtype=self.dtype)

        def layer() -> dict[str, torch.Tensor]:
            # Insertion order is the allocation order of ``spec.tensor_shapes()``.
            tensors = {
                "q": weight(h, h),
                "k": weight(kv, h),
                "v": weight(kv, h),
                "o": weight(h, h),
                "gate": weight(m, h),
                "up": weight(m, h),
                "down": weight(h, m),
                "norm1": norm(),
                "norm2": norm(),
            }
            if spec.attention_bias:
                tensors.update(q_bias=bias(h), k_bias=bias(kv), v_bias=bias(kv))
            return tensors

        h, m = spec.hidden_size, spec.intermediate_size
        kv = spec.num_kv_heads * spec.head_dim
        self.embedding = weight(spec.vocab_size, h)
        self.head = weight(spec.vocab_size, h)
        self.final_norm = norm()
        self.layers: list[dict[str, torch.Tensor]] = [
            layer() for _ in range(spec.num_layers)
        ]
        exponent = torch.arange(0, spec.head_dim, 2, device=self.device).float()
        self.inv_freq = 1.0 / (spec.rope_theta ** (exponent / spec.head_dim))
        self._synchronize()

    def parameters(self) -> list[torch.Tensor]:
        """Every weight tensor, in a stable order (embedding, head, norm, layers)."""
        tensors = [self.embedding, self.head, self.final_norm]
        for layer in self.layers:
            tensors.extend(layer.values())
        return tensors

    def new_caches(self, batch_size: int, capacity: int) -> list[LayerCache]:
        """Allocate one uninitialized (keys, values) cache pair per layer."""
        torch, spec = self._torch, self.spec
        shape = (batch_size, spec.num_kv_heads, capacity, spec.head_dim)
        return [
            (
                torch.empty(shape, device=self.device, dtype=self.dtype),
                torch.empty(shape, device=self.device, dtype=self.dtype),
            )
            for _ in self.layers
        ]

    def _synchronize(self) -> None:
        """Wait for queued CUDA work so wall-clock timing is meaningful."""
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm, computed in float32 and cast back to the working dtype."""
        wide = x.float()
        scale = self._torch.rsqrt(wide.square().mean(-1, keepdim=True) + RMS_NORM_EPS)
        return (wide * scale).to(x.dtype) * weight

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings to ``x`` at absolute ``positions``."""
        torch = self._torch
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)

    def forward(
        self, tokens: torch.Tensor, caches: list[LayerCache], offset: int
    ) -> torch.Tensor:
        """Run the decoder over ``tokens`` and return next-token logits.

        Args:
            tokens: Token IDs, shape ``(batch, length)``.
            caches: Per-layer caches from :meth:`new_caches`; positions
                ``offset .. offset + length`` are written in place.
            offset: Number of positions already in the cache. Zero means a
                causal prefill; otherwise the tokens attend to the whole prefix.

        Returns:
            Logits for the last position, shape ``(batch, vocab_size)``.
        """
        torch, spec = self._torch, self.spec
        functional = torch.nn.functional
        batch, length = tokens.shape
        end = offset + length
        positions = torch.arange(offset, end, device=self.device)
        repeats = spec.num_heads // spec.num_kv_heads
        x = self.embedding[tokens]
        for layer, (keys, values) in zip(self.layers, caches):
            z = self._norm(x, layer["norm1"])
            q, k, v = z @ layer["q"].T, z @ layer["k"].T, z @ layer["v"].T
            if spec.attention_bias:
                q = q + layer["q_bias"]
                k = k + layer["k_bias"]
                v = v + layer["v_bias"]
            q = self._split_heads(q, spec.num_heads)
            k = self._split_heads(k, spec.num_kv_heads)
            v = self._split_heads(v, spec.num_kv_heads)
            q, k = self._rope(q, positions), self._rope(k, positions)
            keys[:, :, offset:end].copy_(k)
            values[:, :, offset:end].copy_(v)
            k = keys[:, :, :end].repeat_interleave(repeats, dim=1)
            v = values[:, :, :end].repeat_interleave(repeats, dim=1)
            # Prefill is causal. A one-token decode attends to the entire prefix;
            # is_causal=True on this non-square attention would mask incorrectly.
            attended = functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=(offset == 0)
            )
            x = x + attended.transpose(1, 2).reshape(batch, length, -1) @ layer["o"].T
            z = self._norm(x, layer["norm2"])
            gated = functional.silu(z @ layer["gate"].T) * (z @ layer["up"].T)
            x = x + gated @ layer["down"].T
        # Only the last position is needed to choose the next token.
        return self._norm(x[:, -1], self.final_norm) @ self.head.T

    def _split_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        """Reshape ``(batch, length, heads * head_dim)`` to per-head layout."""
        batch, length, _ = x.shape
        return x.view(batch, length, heads, self.spec.head_dim).transpose(1, 2)

    def run(
        self,
        batch_size: int,
        prompt_tokens: int,
        new_tokens: int,
        observer: ForwardObserver | None = None,
    ) -> dict[str, Any]:
        """Execute one prefill plus autoregressive decoding.

        This models shapes, not tokenization or content: the input IDs are a
        deterministic numeric sequence. No sleeps, backward passes or training.

        Args:
            batch_size: Requests in the batch, at most ``spec.max_batch_size``.
            prompt_tokens: Prompt length, at most ``spec.max_prompt_tokens``.
            new_tokens: Tokens to generate, at most ``spec.max_new_tokens``.
            observer: Optional callback invoked after each forward pass, inside
                inference mode, for validation. It is part of the timed region.

        Returns:
            Generated ``token_ids`` plus timing, shape and storage accounting.
            ``elapsed_s`` covers request setup, computation, device
            synchronization and the copy of the IDs to the host; it excludes
            weight construction.

        Raises:
            ValueError: A request dimension is out of bounds, or the request
                would exceed the tensor budget.
        """
        spec, torch = self.spec, self._torch
        for name, value, limit in (
            ("batch_size", batch_size, spec.max_batch_size),
            ("prompt_tokens", prompt_tokens, spec.max_prompt_tokens),
            ("new_tokens", new_tokens, spec.max_new_tokens),
        ):
            if not _is_int(value) or not 1 <= value <= limit:
                raise ValueError(f"{name} must be in [1, {limit}]")
        spec.check_storage(batch_size, prompt_tokens, new_tokens)
        capacity = spec.cache_tokens(prompt_tokens, new_tokens)
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            caches = self.new_caches(batch_size, capacity)
            tokens = (
                torch.arange(batch_size * prompt_tokens, device=self.device).reshape(
                    batch_size, prompt_tokens
                )
                % spec.vocab_size
            )
            generated = []
            offset = 0
            for _ in range(new_tokens):
                logits = self.forward(tokens, caches, offset)
                if observer is not None:
                    observer(tokens, caches, offset, logits)
                next_token = logits.argmax(dim=-1)
                generated.append(next_token)
                offset += tokens.shape[1]
                tokens = next_token[:, None]
            token_ids = torch.stack(generated, dim=1).cpu().tolist()
            self._synchronize()
        return {
            "token_ids": token_ids,
            "elapsed_s": time.perf_counter() - start,
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens,
            "forward_calls": new_tokens,
            "prefill_calls": 1,
            "decode_calls": new_tokens - 1,
            "cache_tokens": capacity,
            "parameter_bytes": spec.parameter_count * spec.element_bytes,
            "kv_cache_bytes": spec.kv_bytes(batch_size, prompt_tokens, new_tokens),
            "device": str(self.device),
            "dtype": spec.dtype,
            "fidelity": FIDELITY,
        }


def main() -> None:
    """Tiny standalone CPU/CUDA smoke check; no workflow dependencies needed."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument(
        "--spec-json", help="optional JSON object of InferenceSpec fields"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=3)
    args = parser.parse_args()
    if args.spec_json:
        spec = InferenceSpec(**json.loads(Path(args.spec_json).read_text()))
    else:
        spec = InferenceSpec()
    result = InferenceWorkload(spec, args.device).run(
        args.batch_size, args.prompt_tokens, args.new_tokens
    )
    print(json.dumps({"spec": asdict(spec), "result": result}, indent=2))


if __name__ == "__main__":
    main()
