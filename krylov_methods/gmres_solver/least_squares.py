"""Solve the GMRES reduced least-squares problem after QR rotations."""

import numpy as np
import scipy.linalg as la


def solve_reduced_system(H, g, steps):
    """Triangular back substitution, falling back to least squares if singular."""
    R = H[:steps, :steps]
    rhs = g[:steps]
    if steps == 0:
        return np.empty(0, dtype=g.dtype)
    scale = max(1.0, np.linalg.norm(R, ord=np.inf))
    if np.min(np.abs(np.diag(R))) <= 10 * np.finfo(float).eps * scale:
        return np.linalg.lstsq(R, rhs, rcond=None)[0]
    return la.solve_triangular(R, rhs, lower=False, check_finite=False)


def my_upper_triangular_linalg_solve(A, b):
    A, b = np.asarray(A), np.asarray(b)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
        raise ValueError("Incompatible triangular system dimensions")
    return la.solve_triangular(A, b, lower=False)
