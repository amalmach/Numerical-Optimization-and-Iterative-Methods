"""Stable real and complex Givens rotations for incremental QR."""

import numpy as np


def compute_givens_rotation(a, b):
    """Return c (real), s, r such that [[c,s],[-conj(s),c]]@[a,b]=[r,0]."""
    if b == 0:
        return 1.0, 0.0, a
    if a == 0:
        return 0.0, np.conj(b) / abs(b), abs(b)
    scale = max(abs(a), abs(b))
    norm = scale * np.sqrt((abs(a) / scale) ** 2 + (abs(b) / scale) ** 2)
    alpha = a / abs(a)
    c = abs(a) / norm
    s = alpha * np.conj(b) / norm
    return c, s, alpha * norm


def apply_givens_rotation(column, cosines, sines, k):
    """Apply the k previously computed rotations to a Hessenberg column."""
    for i in range(k):
        a, b = column[i], column[i + 1]
        column[i] = cosines[i] * a + sines[i] * b
        column[i + 1] = -np.conj(sines[i]) * a + cosines[i] * b
    return column


def rotate_rhs(g, k, c, s):
    a, b = g[k], g[k + 1]
    g[k] = c * a + s * b
    g[k + 1] = -np.conj(s) * a + c * b
