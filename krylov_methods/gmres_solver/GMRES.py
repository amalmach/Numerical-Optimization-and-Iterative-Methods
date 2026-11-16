#!/usr/bin/env python3
"""GMRES: unrestarted/restarted, with optional left preconditioning.
From repository parent:
    from gmres_solver.GMRES import gmres
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .arnoldi import arnoldi_step
from .hessenberg import initialize_hessenberg
from .givens import apply_givens_rotation, compute_givens_rotation, rotate_rhs
from .least_squares import solve_reduced_system
from .preconditioners import select_preconditioner


@dataclass
class GMRESResult:
    x: np.ndarray
    converged: bool
    iterations: int
    cycles: int
    residual_norm: float
    residual_history: list[float]
    preconditioned_history: list[float]
    message: str


def _operator(A, n, dtype):
    if callable(A) and not isinstance(A, spl.LinearOperator):
        return lambda x: np.asarray(A(x), dtype=dtype).reshape(-1)
    if not hasattr(A, "shape") or A.shape != (n, n):
        raise ValueError("A must be square and compatible with b")
    return lambda x: np.asarray(A @ x, dtype=dtype).reshape(-1)


def gmres(A, b, x0=None, *, restart=None, maxiter=None, rtol=1e-8,
          atol=0.0, preconditioner=None, M=None, preconditioner_options=None,
          reorthogonalize=True, callback: Callable | None = None,
          verbose=False):
    """Solve Ax=b using real/complex left-preconditioned GMRES.

    Parameters
    ----------
    A : ndarray, sparse matrix, LinearOperator, or callable v -> A@v
        Callable operators must accept and return one-dimensional vectors.
    b : array-like, shape (n,) or (n,1)
    x0 : initial guess (default zero)
    restart : int or None
        Maximum Arnoldi dimension per cycle. None means *unrestarted*.
    maxiter : int or None
        Maximum TOTAL inner Arnoldi iterations; default n if unrestarted,
        otherwise 10*n. Unrestarted is capped at dimension n.
    rtol, atol : nonnegative floats
        Convergence uses TRUE ||b-Ax|| <= max(atol, rtol*||b||).
    preconditioner : none, 'jacobi', 'lu', 'ilu', 'ic0', 'ssor', 'custom'
    M : matrix, inverse-action callable, or LinearOperator with 'custom'
    preconditioner_options : dict of constructor parameters (e.g. drop_tol)
    callback : optional function called with (iteration, true_residual,
        estimated_preconditioned_residual). Intermediate true residuals
        require reconstructing a temporary approximation.

    Returns
    -------
    GMRESResult with solution, convergence metadata and histories.
    The preconditioned history contains Givens estimates; the true residual
    history contains the initial norm, one value per completed cycle and
    optional intermediate values when a callback is given.
    """
    b = np.asarray(b).reshape(-1)
    if b.size == 0 or not np.issubdtype(b.dtype, np.number):
        raise ValueError("b must be a nonempty numeric vector")
    n = len(b)
    dtype = np.result_type(b.dtype, getattr(A, "dtype", float),
                           getattr(M, "dtype", float), np.float64)
    b = b.astype(dtype)
    x = np.zeros(n, dtype=dtype) if x0 is None else np.asarray(x0, dtype=dtype).reshape(-1).copy()
    if x.size != n:
        raise ValueError("x0 has incompatible length")
    if rtol < 0 or atol < 0 or not np.isfinite(rtol + atol):
        raise ValueError("rtol and atol must be finite and nonnegative")
    if restart is not None and (not isinstance(restart, (int, np.integer)) or restart < 1):
        raise ValueError("restart must be None or a positive integer")
    if maxiter is None:
        maxiter = n if restart is None else 10 * n
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    matvec = _operator(A, n, dtype)
    options = {} if preconditioner_options is None else dict(preconditioner_options)
    psolve = select_preconditioner(A, preconditioner, M=M, **options)

    def apply_m(v):
        result = np.asarray(psolve(v), dtype=dtype).reshape(-1)
        if result.size != n or not np.all(np.isfinite(result)):
            raise ValueError("Preconditioner returned invalid vector")
        return result

    threshold = max(atol, rtol * np.linalg.norm(b))
    iterations = 0
    cycles = 0
    true_norm = float(np.linalg.norm(b - matvec(x)))
    true_history = [true_norm]
    pre_history = []
    if true_norm <= threshold:
        return GMRESResult(x, True, 0, 0, true_norm, true_history,
                           pre_history, "initial guess converged")

    limit = min(n, maxiter) if restart is None else min(n, restart)
    message = "maximum iteration count reached"
    while iterations < maxiter:
        cycles += 1
        r = b - matvec(x)
        z = apply_m(r)
        beta = np.linalg.norm(z)
        if beta == 0:
            message = "preconditioner annihilated a nonzero residual"
            break
        dim = min(limit, maxiter - iterations)
        V = np.zeros((n, dim + 1), dtype=dtype)
        H = initialize_hessenberg(dim, dtype=dtype)
        V[:, 0] = z / beta
        g = np.zeros(dim + 1, dtype=dtype)
        g[0] = beta
        cs = np.zeros(dim, dtype=float)
        sn = np.zeros(dim, dtype=dtype)
        steps = 0
        breakdown = False
        estimated_converged = False
        for k in range(dim):
            breakdown = arnoldi_step(lambda v: apply_m(matvec(v)), V, H, k,
                                     reorthogonalize=reorthogonalize)
            # Arnoldi subdiagonal is used even on happy breakdown.
            if breakdown:
                H[k + 1, k] = 0
            apply_givens_rotation(H[:, k], cs, sn, k)
            cs[k], sn[k], H[k, k] = compute_givens_rotation(H[k, k], H[k + 1, k])
            H[k + 1, k] = 0
            rotate_rhs(g, k, cs[k], sn[k])
            iterations += 1
            steps = k + 1
            estimate = float(abs(g[k + 1]))
            pre_history.append(estimate)
            if callback is not None:
                coeff = solve_reduced_system(H, g, steps)
                candidate = x + V[:, :steps] @ coeff
                current_true = float(np.linalg.norm(b - matvec(candidate)))
                true_history.append(current_true)
                callback(iterations, current_true, estimate)
            if verbose:
                print(f"iteration {iterations}: preconditioned residual estimate={estimate:.6e}")
            # A preconditioned norm is not directly comparable to the true
            # tolerance; candidate's TRUE norm is checked after this cycle.
            estimated_converged = estimate <= max(atol, rtol * beta)
            if estimated_converged or breakdown:
                break

        y = solve_reduced_system(H, g, steps)
        x = x + V[:, :steps] @ y
        true_norm = float(np.linalg.norm(b - matvec(x)))
        true_history.append(true_norm)
        if true_norm <= threshold:
            return GMRESResult(x, True, iterations, cycles, true_norm,
                               true_history, pre_history, "converged")
        if breakdown:
            message = "Arnoldi breakdown without true-residual convergence"
            break
        if restart is None:
            message = "unrestarted Krylov dimension exhausted"
            break
        # Left preconditioned residual can be deceptively small; after a
        # restart, recalculate everything from the true residual.
    return GMRESResult(x, False, iterations, cycles, true_norm,
                       true_history, pre_history, message)


def my_gmres_solver(A, b, x0=None, M=None, tol=1e-5, max_iter=None):
    """Old-style compatibility entry point returning only x.

    M is understood as an inverse-action function/operator or a matrix
    representing the preconditioner itself.
    """
    result = gmres(A, b, x0=x0, M=M,
                   preconditioner="custom" if M is not None else None,
                   atol=tol, rtol=0.0, maxiter=max_iter)
    if not result.converged:
        raise RuntimeError(result.message)
    return result.x


def Restart_GMRES_solver(A, b, x0=None, M=None, rtol=1e-8, atol=1e-8,
                         restart=None, max_iter_per_restart=30):
    """Compatibility wrapper: restart denotes cycle COUNT (old convention).

    Prefer gmres(restart=KRYLOV_DIM, maxiter=TOTAL_INNER_ITERATIONS).
    """
    cycles = 1 if restart is None else int(restart)
    if cycles < 1:
        raise ValueError("restart cycle count must be positive")
    result = gmres(A, b, x0=x0, M=M,
                   preconditioner="custom" if M is not None else None,
                   rtol=rtol, atol=atol,
                   restart=None if restart is None else max_iter_per_restart,
                   maxiter=cycles * max_iter_per_restart)
    if not result.converged:
        raise RuntimeError(result.message)
    return result.x


GMRES_solver = my_gmres_solver
