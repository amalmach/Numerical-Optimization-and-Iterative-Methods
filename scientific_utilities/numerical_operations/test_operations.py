"""Automated tests for elementary numerical operations."""

import numpy as np
import pytest

from scientific_utilities.numerical_operations.operations import (
    closest,
    norm_two,
    AxB,
    matvec,
    scalar,
    vecvec,
    matT,
    vec_asarray,
)


def test_closest():
    """Find the nearest value in a numerical sequence."""
    assert closest([0.1, 0.4, 0.8], 0.7) == 0.8


def test_closest_tie():
    """Preserve the original last-occurrence tie-breaking behavior."""
    assert closest([1.0, 3.0], 2.0) == 3.0


def test_closest_invalid_input():
    """Reject empty sequences."""
    with pytest.raises(ValueError):
        closest([], 1.0)


def test_norm_two():
    """Compare the Euclidean norm with NumPy."""
    v = np.array([3.0, 4.0, 12.0])

    assert np.isclose(norm_two(v), np.linalg.norm(v))


def test_norm_two_invalid_input():
    """Reject nonfinite vector entries."""
    with pytest.raises(ValueError):
        norm_two([1.0, np.nan])


def test_matrix_multiplication():
    """Compare matrix multiplication with NumPy."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    assert np.allclose(AxB(A, B), A @ B)


def test_rectangular_matrix_multiplication():
    """Support compatible rectangular matrices."""
    rng = np.random.default_rng(42)

    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))

    assert np.allclose(AxB(A, B), A @ B)


def test_incompatible_matrix_dimensions():
    """Reject incompatible matrix dimensions."""
    with pytest.raises(ValueError):
        AxB(np.ones((2, 3)), np.ones((4, 2)))


def test_matrix_vector_product():
    """Compare matrix-vector multiplication with NumPy."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([2.0, 3.0])

    assert np.allclose(matvec(A, v), A @ v)


def test_rectangular_matrix_vector_product():
    """Support rectangular matrix-vector multiplication."""
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = np.array([1.0, 2.0, 3.0])

    assert np.allclose(matvec(A, v), A @ v)


def test_scalar_product():
    """Compare the dot product with NumPy."""
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])

    assert np.isclose(scalar(u, v), np.dot(u, v))


def test_scalar_product_invalid_dimensions():
    """Reject vectors with different lengths."""
    with pytest.raises(ValueError):
        scalar([1.0, 2.0], [1.0, 2.0, 3.0])


def test_outer_product():
    """Compare the outer product with NumPy."""
    u = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0, 5.0])

    assert np.allclose(vecvec(u, v), np.outer(u, v))


def test_matrix_transpose():
    """Compare matrix transposition with NumPy."""
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert np.array_equal(matT(A), A.T)


def test_vec_asarray():
    """Convert a rectangular collection of vectors into an array."""
    vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    result = vec_asarray(vectors)

    assert np.array_equal(result, np.asarray(vectors))
    assert result.shape == (3, 2)


def test_vec_asarray_invalid_shape():
    """Reject irregular nested vectors."""
    with pytest.raises(ValueError):
        vec_asarray([[1.0, 2.0], [3.0]])
