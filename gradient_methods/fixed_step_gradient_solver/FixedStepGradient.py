"""Fixed-step gradient descent for quadratic optimization problems."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedStepGradientResult:
    """Numerical result and convergence diagnostics."""

    solution: np.ndarray
    iterations: int
    converged: bool
    cost_history: list[float]
    gradient_norms: list[float]
    message: str


def J1(u):
    """J1(u) = sum((u_i - 1)^2), with minimizer (1, ..., 1)."""
    u = np.asarray(u, dtype=float)
    target = np.ones_like(u)
    difference = u - target
    return float(np.dot(difference, difference)), 2.0 * difference


def J2(u):
    """J2(u) = sum((u_i - i)^2), with i = 1, ..., N."""
    u = np.asarray(u, dtype=float)
    target = np.arange(1, u.size + 1, dtype=float)
    difference = u - target
    return float(np.dot(difference, difference)), 2.0 * difference


def cost(u, k):
    """Evaluate one of the two original quadratic objectives."""
    if k == 1:
        return J1(u)
    if k == 2:
        return J2(u)
    raise ValueError("Objective k must be 1 (J1) or 2 (J2).")


def solve_fixed_step_gradient(
    u0,
    rho=0.25,
    max_iter=1500,
    tol=1e-10,
    objective=1,
):
    """
    Minimize J1 or J2 using fixed-step gradient descent.

    Iteration:
        u_{k+1} = u_k - rho * grad(J(u_k))

    Stopping condition:
        ||grad(J(u_k))||_2 <= tol

    For these quadratic objectives, convergence from a nonoptimal
    starting point requires 0 < rho < 1.
    """
    u = np.asarray(u0, dtype=float)

    if u.ndim != 1 or u.size == 0:
        raise ValueError("u0 must be a nonempty one-dimensional array.")

    if not np.all(np.isfinite(u)):
        raise ValueError("u0 must contain only finite values.")

    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be a finite positive number.")

    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, (bool, np.bool_)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer.")

    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")

    if objective not in (1, 2):
        raise ValueError("objective must be 1 (J1) or 2 (J2).")

    u = u.copy()
    cost_history = []
    gradient_norms = []

    for iteration in range(max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            value, gradient = cost(u, objective)
            gradient_norm = float(np.linalg.norm(gradient))

        cost_history.append(value)
        gradient_norms.append(gradient_norm)

        if not np.isfinite(value) or not np.isfinite(gradient_norm):
            return FixedStepGradientResult(
                solution=u,
                iterations=iteration,
                converged=False,
                cost_history=cost_history,
                gradient_norms=gradient_norms,
                message="Numerical divergence: nonfinite cost or gradient.",
            )

        if gradient_norm <= tol:
            return FixedStepGradientResult(
                solution=u,
                iterations=iteration,
                converged=True,
                cost_history=cost_history,
                gradient_norms=gradient_norms,
                message="Gradient tolerance satisfied.",
            )

        if iteration == max_iter:
            break

        with np.errstate(over="ignore", invalid="ignore"):
            u = u - rho * gradient

    return FixedStepGradientResult(
        solution=u,
        iterations=max_iter,
        converged=False,
        cost_history=cost_history,
        gradient_norms=gradient_norms,
        message="Maximum number of iterations reached.",
    )


def GF(u, rho, NbrIter, epsJ, k):
    """
    Compatibility wrapper for the original notebook function.

    Returns:
        solution, iterations, cost_history
    """
    result = solve_fixed_step_gradient(
        u0=u,
        rho=rho,
        max_iter=NbrIter,
        tol=epsJ,
        objective=k,
    )

    return result.solution, result.iterations, result.cost_history
