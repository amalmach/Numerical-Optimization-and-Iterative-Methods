"""
Elementary numerical operations.

Educational implementations of basic numerical linear algebra
operations, adapted from the repository's original scripts.
"""

import numpy as np


def _vector(v):
    """Convert input to a finite, nonempty real vector."""
    v = np.asarray(v, dtype=float)

    if v.ndim != 1 or v.size == 0:
        raise ValueError("Input must be a nonempty 1D vector.")

    if not np.all(np.isfinite(v)):
        raise ValueError("Vector must contain finite values.")

    return v


def _matrix(A):
    """Convert input to a finite, nonempty real matrix."""
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or 0 in A.shape:
        raise ValueError("Input must be a nonempty 2D matrix.")

    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix must contain finite values.")

    return A


def closest(lst, val):
    """
    Find the element of lst nearest to val.

    As in the original implementation, ties are resolved
    by selecting the last equally close element.
    """
    values = _vector(lst)

    if not np.isfinite(val):
        raise ValueError("val must be finite.")

    nearest = values[0]

    for value in values[1:]:
        if abs(value - val) <= abs(nearest - val):
            nearest = value

    return float(nearest)


def norm_two(v):
    """Compute the Euclidean norm of a vector."""
    v = _vector(v)

    return float(np.linalg.norm(v))


def AxB(X, Y):
    """Compute the product of two matrices using explicit loops."""
    X = _matrix(X)
    Y = _matrix(Y)

    if X.shape[1] != Y.shape[0]:
        raise ValueError("Incompatible matrix dimensions.")

    m, n = X.shape
    p = Y.shape[1]

    result = np.zeros((m, p), dtype=float)

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += X[i, k] * Y[k, j]

    return result


def matvec(A, v):
    """Compute a matrix-vector product."""
    A = _matrix(A)
    v = _vector(v)

    if A.shape[1] != v.size:
        raise ValueError("Incompatible matrix-vector dimensions.")

    result = np.zeros(A.shape[0], dtype=float)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i] += A[i, j] * v[j]

    return result


def scalar(u, v):
    """Compute the Euclidean dot product."""
    u = _vector(u)
    v = _vector(v)

    if u.size != v.size:
        raise ValueError("Vectors must have the same length.")

    result = 0.0

    for i in range(u.size):
        result += u[i] * v[i]

    return float(result)


def vecvec(u, v):
    """Compute the outer product of two vectors."""
    u = _vector(u)
    v = _vector(v)

    result = np.zeros((u.size, v.size), dtype=float)

    for i in range(u.size):
        for j in range(v.size):
            result[i, j] = u[i] * v[j]

    return result


def matT(A):
    """Compute the transpose of a matrix."""
    A = _matrix(A)

    m, n = A.shape
    result = np.zeros((n, m), dtype=float)

    for i in range(m):
        for j in range(n):
            result[j, i] = A[i, j]

    return result


def vec_asarray(v):
    """
    Convert a rectangular collection of vectors into a 2D array.

    Each input vector becomes one row of the resulting matrix.
    """
    return _matrix(v).copy()
