"""Storage and inspection helpers for Arnoldi Hessenberg matrices."""

import numpy as np


def initialize_hessenberg(krylov_dim, dtype=float):
    """Create the (m+1)-by-m upper Hessenberg workspace."""
    return np.zeros((krylov_dim + 1, krylov_dim), dtype=dtype)


def projected_hessenberg(H, steps):
    """Return the leading square projected matrix (read-only copy)."""
    return H[:steps, :steps].copy()
