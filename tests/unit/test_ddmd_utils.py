"""Unit tests for pipelines.ddmd_pipeline.utils module."""

import math

import numpy as np
import pytest

from pipelines.ddmd_pipeline.utils import (
    bestk,
    hash2intarray,
    intarray2hash,
    t1Dto2D,
    t2Dto1D,
)


# ---------------------------------------------------------------------------
# bestk
# ---------------------------------------------------------------------------
class TestBestk:
    def test_smallest_k(self):
        """Return the k smallest values and their indices."""
        a = [5.0, 1.0, 3.0, 0.5, 4.0]
        values, indices = bestk(a, k=2, smallest=True)
        # Smallest 2: 0.5 (idx 3), 1.0 (idx 1)
        np.testing.assert_array_almost_equal(values, [0.5, 1.0])
        np.testing.assert_array_equal(indices, [3, 1])

    def test_largest_k(self):
        """Return the k largest values when smallest=False."""
        a = [5.0, 1.0, 3.0, 0.5, 4.0]
        values, indices = bestk(a, k=2, smallest=False)
        # values are negated internally, so returned "values" are negative
        # Largest 2: 5.0 (idx 0), 4.0 (idx 4) -> negated: -5.0, -4.0
        assert indices[0] == 0  # 5.0
        assert indices[1] == 4  # 4.0

    def test_k_equals_length(self):
        """When k >= len(a), return all elements sorted."""
        a = [3.0, 1.0, 2.0]
        values, indices = bestk(a, k=3, smallest=True)
        assert len(values) == 3
        np.testing.assert_array_almost_equal(values, [1.0, 2.0, 3.0])

    def test_k_greater_than_length(self):
        """When k > len(a), return all elements."""
        a = [3.0, 1.0]
        values, indices = bestk(a, k=10, smallest=True)
        assert len(values) == 2

    def test_single_element(self):
        a = [42.0]
        values, indices = bestk(a, k=1, smallest=True)
        np.testing.assert_array_almost_equal(values, [42.0])
        np.testing.assert_array_equal(indices, [0])

    def test_returns_sorted_values(self):
        """Returned values should be in sorted order."""
        a = np.random.rand(20)
        values, indices = bestk(a, k=5, smallest=True)
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------------------------
# t2Dto1D / t1Dto2D
# ---------------------------------------------------------------------------
class TestTriangularConversions:
    def test_t2Dto1D_basic(self):
        """Convert upper triangle of symmetric matrix to 1D array."""
        A = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        B = t2Dto1D(A)
        # Upper triangle (i<j): (0,1)=0, (0,2)=1, (1,2)=0
        assert len(B) == 3
        np.testing.assert_array_equal(B, [0, 1, 0])

    def test_t1Dto2D_basic(self):
        """Convert 1D array back to symmetric matrix."""
        B = np.array([0, 1, 0], dtype=np.uint8)
        A = t1Dto2D(B)
        assert A.shape == (3, 3)
        # Diagonal should be 1
        np.testing.assert_array_equal(np.diag(A), [1, 1, 1])
        # Symmetric
        np.testing.assert_array_equal(A, A.T)
        # Off-diagonal values
        assert A[0, 1] == 0
        assert A[0, 2] == 1
        assert A[1, 2] == 0

    def test_roundtrip(self):
        """t1Dto2D(t2Dto1D(A)) should recover the upper triangle."""
        n = 5
        A = np.random.randint(0, 2, size=(n, n), dtype=np.uint8)
        # Make symmetric with 1s on diagonal
        A = np.triu(A, 1)
        A = A + A.T
        np.fill_diagonal(A, 1)

        B = t2Dto1D(A)
        A_recovered = t1Dto2D(B)
        np.testing.assert_array_equal(A, A_recovered)

    def test_1d_length_formula(self):
        """1D array length should be n*(n-1)/2."""
        for n in [3, 4, 5, 10]:
            A = np.zeros((n, n), dtype=np.uint8)
            B = t2Dto1D(A)
            assert len(B) == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# hash2intarray / intarray2hash
# ---------------------------------------------------------------------------
class TestHashConversions:
    def test_roundtrip(self):
        """intarray2hash(hash2intarray(h)) == h for valid hex strings."""
        h = "0000ffff0001"
        ia = hash2intarray(h)
        result = intarray2hash(ia)
        assert result == h

    def test_hash2intarray_values(self):
        """Check specific hex-to-int conversion."""
        h = "0001000a"  # 1, 10 in 4-hex-digit chunks
        ia = hash2intarray(h)
        np.testing.assert_array_equal(ia, [1, 10])

    def test_intarray2hash_values(self):
        """Check specific int-to-hex conversion."""
        ia = np.array([1, 10], dtype=np.int64)
        result = intarray2hash(ia)
        assert result == "0001000a"

    def test_empty_hash(self):
        """Empty string produces empty array."""
        ia = hash2intarray("")
        assert len(ia) == 0


# ---------------------------------------------------------------------------
# parse_args (basic check)
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_parse_args_requires_config(self):
        """parse_args should fail without -c/--config."""
        from pipelines.ddmd_pipeline.utils import parse_args

        with pytest.raises(SystemExit):
            import sys

            old_argv = sys.argv
            sys.argv = ["prog"]
            try:
                parse_args()
            finally:
                sys.argv = old_argv

    def test_parse_args_with_config(self):
        """parse_args should accept -c flag."""
        from pipelines.ddmd_pipeline.utils import parse_args

        import sys

        old_argv = sys.argv
        sys.argv = ["prog", "-c", "test.yaml"]
        try:
            args = parse_args()
            assert args.config == "test.yaml"
        finally:
            sys.argv = old_argv
