"""Newton--Krylov nonlinear solvers

Right preconditioners are inverse actions P^{-1} v, never dense inverses.
SciPy >= 1.11 is recommended for gmres(rtol=, atol=, callback_type=).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

Vector = NDArray[np.float64]
Function = Callable[[Vector], Vector]
Preconditioner = Callable[[Vector, Vector], Vector]


@dataclass
class NewtonKrylovResult:
    x: Vector
    converged: bool
    iterations: int
    residual_norms: list[float]
    gmres_iterations: list[int]
    forcing_terms: list[float]
    step_lengths: list[float]
    message: str


def newton_krylov(
    fun: Function,
    x0: Vector,
    *,
    jacobian: Callable[[Vector], NDArray[np.float64]] | None = None,
    inverse_preconditioner: Preconditioner | None = None,
    tol: float = 1e-10,
    max_newton: int = 50,
    linear_rtol: float = 1e-3,
    adaptive_forcing: bool = False,
    restart: int = 30,
    max_linear_cycles: int = 20,
    fd_step: float | None = None,
    line_search: bool = True,
) -> NewtonKrylovResult:
    """Solve F(x)=0 using (potentially Jacobian-free) right-preconditioned GMRES.

    A supplied ``jacobian`` uses an explicit dense matrix and exact matrix-vector
    multiplication. Otherwise Jv is approximated by a forward difference.

    ``inverse_preconditioner(x, v)`` computes P(x)^{-1} v for *right*
    preconditioning: (J P^{-1}) y = -F and dx = P^{-1} y. For an inexact
    inner solve, the true (unpreconditioned) linear residual is checked.

    Adaptive forcing uses the simple clipped residual-ratio rule
    eta_k = clip(0.9 * (||F_k|| / ||F_{k-1}||)^1.5, 1e-8, 0.5)
    for k > 0, with the initial eta_0 = ``linear_rtol``.
    """
    if tol <= 0 or not (0 < linear_rtol < 1) or max_newton < 1:
        raise ValueError("Require tol > 0, 0 < linear_rtol < 1, max_newton >= 1")
    if restart < 1 or max_linear_cycles < 1 or (fd_step is not None and fd_step <= 0):
        raise ValueError("restart and max_linear_cycles >= 1, fd_step > 0")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite nonempty one-dimensional array")
    n = x.size

    def evaluate(z: Vector) -> Vector:
        val = np.asarray(fun(z), dtype=float)
        if val.shape != (n,) or not np.all(np.isfinite(val)):
            raise ValueError("fun must return a finite vector of length len(x0)")
        return val

    f = evaluate(x)
    norms = [float(np.linalg.norm(f))]
    counts: list[int] = []
    etas: list[float] = []
    alphas: list[float] = []

    def done(converged: bool, message: str) -> NewtonKrylovResult:
        return NewtonKrylovResult(x.copy(), converged, len(alphas), norms, counts, etas, alphas, message)

    if norms[-1] <= tol:
        return done(True, "Initial guess satisfies tolerance")

    for k in range(max_newton):
        if adaptive_forcing and k > 0:
            eta = float(np.clip(0.9 * (norms[-1] / norms[-2]) ** 1.5, 1e-8, 0.5))
        else:
            eta = linear_rtol
        etas.append(eta)

        if jacobian is not None:
            jac = np.asarray(jacobian(x), dtype=float)
            if jac.shape != (n, n) or not np.all(np.isfinite(jac)):
                raise ValueError("jacobian must return a finite (n, n) array")

            def jvp(v: Vector) -> Vector:
                return jac @ v
        else:
            def jvp(v: Vector) -> Vector:
                vnorm = np.linalg.norm(v)
                if vnorm == 0:
                    return np.zeros(n)
                h = (fd_step if fd_step is not None else np.sqrt(np.finfo(float).eps))
                h *= (1 + np.linalg.norm(x)) / vnorm
                return (evaluate(x + h * v) - f) / h

        def pinv(v: Vector) -> Vector:
            value = v if inverse_preconditioner is None else inverse_preconditioner(x, v)
            value = np.asarray(value, dtype=float)
            if value.shape != (n,) or not np.all(np.isfinite(value)):
                raise ValueError("inverse_preconditioner must return a finite n-vector")
            return value

        op = LinearOperator((n, n), matvec=lambda v: jvp(pinv(v)), dtype=float)
        count = 0

        def count_callback(_):
            nonlocal count
            count += 1

        correction_y, info = gmres(
            op, -f, rtol=eta, atol=0.0,
            restart=min(n, restart), maxiter=max_linear_cycles,
            callback=count_callback, callback_type="pr_norm",
        )
        counts.append(count)
        if info != 0 or not np.all(np.isfinite(correction_y)):
            return done(False, f"GMRES did not converge at Newton iteration {k + 1} (info={info})")
        step = pinv(correction_y)
        # Confirm that the returned right-preconditioned step actually solves
        # the *original* linearized equation to requested relative accuracy.
        linear_residual = float(np.linalg.norm(jvp(step) + f))
        if linear_residual > max(1.5 * eta * norms[-1], 1e-12):
            return done(False, f"True linear residual too large at Newton iteration {k + 1}")

        phi = 0.5 * norms[-1] ** 2
        accepted = False
        alpha = 1.0
        for _ in range(35 if line_search else 1):
            trial_x = x + alpha * step
            try:
                trial_f = evaluate(trial_x)
            except ValueError:
                alpha *= 0.5
                continue
            trial_norm = float(np.linalg.norm(trial_f))
            if not line_search or 0.5 * trial_norm**2 <= (1 - 1e-4 * alpha) * phi:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return done(False, f"Backtracking failed at Newton iteration {k + 1}")
        x, f = trial_x, trial_f
        norms.append(trial_norm)
        alphas.append(alpha)
        if norms[-1] <= tol:
            return done(True, "Converged")
    return done(False, "Maximum Newton iterations reached")


def newton_gmres(fun: Function, x0: Vector, *, jacobian, **kwargs) -> NewtonKrylovResult:
    """Newton--GMRES with a supplied dense analytic Jacobian."""
    return newton_krylov(fun, x0, jacobian=jacobian, **kwargs)


def jfnk(fun: Function, x0: Vector, **kwargs) -> NewtonKrylovResult:
    """Jacobian-free Newton--GMRES."""
    return newton_krylov(fun, x0, **kwargs)


def preconditioned_jfnk(
    fun: Function, x0: Vector, *, inverse_preconditioner: Preconditioner, **kwargs
) -> NewtonKrylovResult:
    """Jacobian-free Newton--GMRES with right preconditioning."""
    return newton_krylov(fun, x0, inverse_preconditioner=inverse_preconditioner, **kwargs)


def inexact_jfnk(fun: Function, x0: Vector, **kwargs) -> NewtonKrylovResult:
    """JFNK with adaptive forcing terms for inner GMRES solves."""
    return newton_krylov(fun, x0, adaptive_forcing=True, **kwargs)
