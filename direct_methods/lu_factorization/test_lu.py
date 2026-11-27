"""
Automated tests for LU factorization with partial pivoting.
"""

import numpy as np
import pytest

from scipy.linalg import lu

from direct_methods.lu_factorization.lu import lu_factorization


def test_lu_factorization_basic():
    """Verify PA = LU for a standard nonsingular matrix."""

    A = np.array([
        [4.0, 3.0],
        [6.0, 3.0],
    ])

    P, L, U = lu_factorization(A)

    assert np.allclose(P @ A, L @ U)
    assert np.allclose(L, np.tril(L))
    assert np.allclose(U, np.triu(U))
    assert np.allclose(np.diag(L), 1.0)


def test_lu_factorization_with_pivoting():
    """Verify factorization when the first pivot is zero."""

    A = np.array([
        [0.0, 2.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0],
    ])

    P, L, U = lu_factorization(A)

    assert np.allclose(P @ A, L @ U)
    assert not np.allclose(P, np.eye(3))


def test_lu_factorization_random_matrix():
    """Verify LU factorization on a reproducible random matrix."""

    rng = np.random.default_rng(42)

    A = rng.standard_normal((5, 5))

    P, L, U = lu_factorization(A)

    residual = np.linalg.norm(P @ A - L @ U)

    assert residual < 1e-10


def test_lu_factorization_scipy():
    """Compare the factorization residual with SciPy's LU."""

    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])

    P, L, U = lu_factorization(A)

    # scipy.linalg.lu uses A = P_scipy @ L_scipy @ U_scipy.
    P_scipy, L_scipy, U_scipy = lu(A)

    assert np.allclose(P @ A, L @ U)
    assert np.allclose(A, P_scipy @ L_scipy @ U_scipy)

    assert np.allclose(A, P.T @ L @ U)


def test_lu_factorization_singular():
    """Singular matrices must raise an exception."""

    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(np.linalg.LinAlgError):
        lu_factorization(A)


def test_lu_factorization_invalid_shape():
    """Reject nonsquare matrices."""

    A = np.ones((2, 3))

    with pytest.raises(ValueError):
        lu_factorization(A)


def test_lu_factorization_nonfinite():
    """Reject matrices containing NaN or infinity."""

    A = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
    ])

    with pytest.raises(ValueError):
        lu_factorization(A)


def test_lu_factorization_preserves_input():
    """Factorization must not modify the input matrix."""

    A = np.array([
        [3.0, 4.0],
        [1.0, 2.0],
    ])

    original = A.copy()

    lu_factorization(A)

    assert np.array_equal(A, original)


from direct_methods.lu_factorization.lu import (
    forward_substitution,
    backward_substitution,
    lu_solve,
)


def test_forward_substitution():
    """Verify solving a lower triangular system."""

    L = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 3.0, 0.0],
        [4.0, -1.0, 2.0],
    ])

    b = np.array([2.0, 7.0, 8.0])

    y = forward_substitution(L, b)

    assert np.allclose(L @ y, b)
    assert np.allclose(y, np.linalg.solve(L, b))


def test_backward_substitution():
    """Verify solving an upper triangular system."""

    U = np.array([
        [2.0, 1.0, -1.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 4.0],
    ])

    b = np.array([2.0, 7.0, 8.0])

    x = backward_substitution(U, b)

    assert np.allclose(U @ x, b)
    assert np.allclose(x, np.linalg.solve(U, b))


def test_lu_solve_with_pivoting():
    """Verify LU solving when row exchanges are required."""

    A = np.array([
        [0.0, 2.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0],
    ])

    b = np.array([3.0, 2.0, 9.0])

    x = lu_solve(A, b)

    assert np.allclose(A @ x, b)
    assert np.allclose(x, np.linalg.solve(A, b))


def test_lu_solve_random_matrix():
    """Compare LU solving against NumPy for a random system."""

    rng = np.random.default_rng(42)

    A = rng.standard_normal((6, 6))
    b = rng.standard_normal(6)

    x = lu_solve(A, b)
    reference = np.linalg.solve(A, b)

    residual = np.linalg.norm(A @ x - b)

    assert np.allclose(x, reference)
    assert residual < 1e-10


def test_lu_solve_singular_matrix():
    """LU solving must reject singular matrices."""

    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(np.linalg.LinAlgError):
        lu_solve(A, b)


def test_lu_solve_invalid_rhs():
    """Reject incompatible right-hand-side dimensions."""

    A = np.eye(3)
    b = np.ones(2)

    with pytest.raises(ValueError):
        lu_solve(A, b)
