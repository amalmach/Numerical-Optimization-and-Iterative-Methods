"""Modified Gram-Schmidt Arnoldi orthogonalization."""

import numpy as np


def arnoldi_step(operator, V, H, k, *, reorthogonalize=True):
    """Build column k of H and (when possible) basis vector k+1.

    The returned bool reports Arnoldi breakdown, i.e. no independent
    vector could be added.  A breakdown can be a happy breakdown.
    """
    w = np.asarray(operator(V[:, k])).reshape(-1).copy()
    for i in range(k + 1):
        h = np.vdot(V[:, i], w)
        H[i, k] += h
        w -= h * V[:, i]
    if reorthogonalize:
        for i in range(k + 1):
            h = np.vdot(V[:, i], w)
            H[i, k] += h
            w -= h * V[:, i]
    norm = np.linalg.norm(w)
    H[k + 1, k] = norm
    # Relative scale guards against false new directions in finite precision.
    scale = max(1.0, np.linalg.norm(H[:k + 2, k]))
    breakdown = norm <= 20 * np.finfo(float).eps * scale
    if not breakdown:
        V[:, k + 1] = w / norm
    return breakdown
