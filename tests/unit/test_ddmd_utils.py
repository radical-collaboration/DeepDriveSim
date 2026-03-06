"""Unit tests for workflows.ddmd_workflow.utils module."""

import sys

import numpy as np
import pytest

from workflows.ddmd_workflow.utils import (
    bestk,
    hash2intarray,
    intarray2hash,
    parse_args,
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
    def test_t2d_to_1d_basic(self):
        """Convert upper triangle of symmetric matrix to 1D array."""
        mat = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        vec = t2Dto1D(mat)
        # Upper triangle (i<j): (0,1)=0, (0,2)=1, (1,2)=0
        assert len(vec) == 3
        np.testing.assert_array_equal(vec, [0, 1, 0])

    def test_t1d_to_2d_basic(self):
        """Convert 1D array back to symmetric matrix."""
        vec = np.array([0, 1, 0], dtype=np.uint8)
        mat = t1Dto2D(vec)
        assert mat.shape == (3, 3)
        # Diagonal should be 1
        np.testing.assert_array_equal(np.diag(mat), [1, 1, 1])
        # Symmetric
        np.testing.assert_array_equal(mat, mat.T)
        # Off-diagonal values
        assert mat[0, 1] == 0
        assert mat[0, 2] == 1
        assert mat[1, 2] == 0

    def test_roundtrip(self):
        """t1Dto2D(t2Dto1D(A)) should recover the upper triangle."""
        n = 5
        mat = np.random.randint(0, 2, size=(n, n), dtype=np.uint8)
        # Make symmetric with 1s on diagonal
        mat = np.triu(mat, 1)
        mat = mat + mat.T
        np.fill_diagonal(mat, 1)

        vec = t2Dto1D(mat)
        mat_recovered = t1Dto2D(vec)
        np.testing.assert_array_equal(mat, mat_recovered)

    def test_1d_length_formula(self):
        """1D array length should be n*(n-1)/2."""
        for n in [3, 4, 5, 10]:
            mat = np.zeros((n, n), dtype=np.uint8)
            vec = t2Dto1D(mat)
            assert len(vec) == n * (n - 1) // 2


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
        old_argv = sys.argv
        sys.argv = ["prog"]
        try:
            with pytest.raises(SystemExit):
                parse_args()
        finally:
            sys.argv = old_argv

    def test_parse_args_with_config(self):
        """parse_args should accept -c flag."""
        old_argv = sys.argv
        sys.argv = ["prog", "-c", "test.yaml"]
        try:
            args = parse_args()
            assert args.config == "test.yaml"
        finally:
            sys.argv = old_argv
