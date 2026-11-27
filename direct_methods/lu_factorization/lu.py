"""
LU factorization with partial pivoting.

Computes P, L, U such that P @ A = L @ U.
"""

import numpy as np


def lu_factorization(A, pivot_tol=None):
    """
    Compute the LU factorization of a square matrix.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Input matrix.
    pivot_tol : float, optional
        Absolute pivot threshold used to detect singularity.
        Defaults to a floating-point roundoff-based tolerance.

    Returns
    -------
    P : ndarray
        Permutation matrix.
    L : ndarray
        Unit lower triangular matrix.
    U : ndarray
        Upper triangular matrix.

    Raises
    ------
    ValueError
        If the input is not a finite square matrix.
    numpy.linalg.LinAlgError
        If a zero or numerically negligible pivot is encountered.
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    if A.shape[0] == 0:
        raise ValueError("A must be nonempty.")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values.")

    n = A.shape[0]

    if pivot_tol is None:
        pivot_tol = (
            np.finfo(float).eps
            * n
            * np.linalg.norm(A, ord=np.inf)
        )

    if not np.isfinite(pivot_tol) or pivot_tol < 0:
        raise ValueError("pivot_tol must be finite and nonnegative.")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n):

        # Select the largest available pivot in column k.
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        pivot_value = abs(U[pivot_row, k])

        if pivot_value <= pivot_tol:
            raise np.linalg.LinAlgError(
                f"Matrix is singular or numerically singular at column {k}."
            )

        # Exchange rows when necessary.
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]

            # Exchange previously computed L multipliers.
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Gaussian elimination.
        for i in range(k + 1, n):
            multiplier = U[i, k] / U[k, k]

            L[i, k] = multiplier

            U[i, k:] -= multiplier * U[k, k:]

            # Enforce the exact triangular structure.
            U[i, k] = 0.0

    return P, L, U
    
def forward_substitution(L, b):
    """
    Solve Ly = b for a nonsingular lower triangular matrix L.

    Parameters
    ----------
    L : array_like, shape (n, n)
        Lower triangular matrix.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    ndarray
        Solution vector y.
    """
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] == 0:
        raise ValueError("L must be a nonempty square matrix.")

    n = L.shape[0]

    if b.shape != (n,):
        raise ValueError("b must be a vector with length matching L.")

    if not np.all(np.isfinite(L)) or not np.all(np.isfinite(b)):
        raise ValueError("L and b must contain finite values.")

    if not np.allclose(L, np.tril(L)):
        raise ValueError("L must be lower triangular.")

    y = np.zeros(n, dtype=float)

    for i in range(n):
        diagonal = L[i, i]

        if abs(diagonal) <= np.finfo(float).tiny:
            raise np.linalg.LinAlgError(
                f"Zero diagonal encountered at row {i}."
            )

        y[i] = (
            b[i] - np.dot(L[i, :i], y[:i])
        ) / diagonal

    return y


def backward_substitution(U, b):
    """
    Solve Ux = b for a nonsingular upper triangular matrix U.

    Parameters
    ----------
    U : array_like, shape (n, n)
        Upper triangular matrix.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    ndarray
        Solution vector x.
    """
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)

    if U.ndim != 2 or U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError("U must be a nonempty square matrix.")

    n = U.shape[0]

    if b.shape != (n,):
        raise ValueError("b must be a vector with length matching U.")

    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(b)):
        raise ValueError("U and b must contain finite values.")

    if not np.allclose(U, np.triu(U)):
        raise ValueError("U must be upper triangular.")

    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        diagonal = U[i, i]

        if abs(diagonal) <= np.finfo(float).tiny:
            raise np.linalg.LinAlgError(
                f"Zero diagonal encountered at row {i}."
            )

        x[i] = (
            b[i] - np.dot(U[i, i + 1:], x[i + 1:])
        ) / diagonal

    return x

def lu_solve(A, b, pivot_tol=None):
    """
    Solve Ax = b using LU factorization with partial pivoting.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Nonsingular coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side vector.
    pivot_tol : float, optional
        Pivot threshold passed to lu_factorization.

    Returns
    -------
    ndarray, shape (n,)
        Solution vector.

    Raises
    ------
    ValueError
        If A and b have incompatible dimensions.
    numpy.linalg.LinAlgError
        If A is singular or numerically singular.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix.")

    if b.shape != (A.shape[0],):
        raise ValueError("b must be a vector matching the dimension of A.")

    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values.")

    P, L, U = lu_factorization(A, pivot_tol=pivot_tol)

    y = forward_substitution(L, P @ b)
    x = backward_substitution(U, y)

    return x
