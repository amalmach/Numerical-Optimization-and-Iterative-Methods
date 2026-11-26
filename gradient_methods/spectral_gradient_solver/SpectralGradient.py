"""Barzilai-Borwein spectral gradient methods (BB1 and BB2)."""

from dataclasses import dataclass

import numpy as np

from .objectives import cost


@dataclass
class SpectralGradientResult:
    """Solution and convergence diagnostics."""

    solution: np.ndarray
    iterations: int
    converged: bool
    cost_history: list[float]
    gradient_norms: list[float]
    step_history: list[float]
    message: str


def solve_spectral_gradient(
    x0,
    *,
    variant="BB1",
    objective=1,
    initial_step=0.25,
    max_iter=1000,
    tol=1e-10,
    step_min=1e-12,
    step_max=1e6,
):
    """Minimize a supported objective using BB1 or BB2.

    The initial update uses ``initial_step``.

    Subsequent step sizes use:

        BB1: alpha = (s.T @ s) / (s.T @ y)
        BB2: alpha = (s.T @ y) / (y.T @ y)

    where s = x_new - x and
          y = grad_new - grad.

    Invalid curvature triggers a fallback to the initial step.
    """

    x = np.asarray(x0, dtype=float)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("x0 must be a nonempty 1D vector.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain finite values.")

    if variant not in ("BB1", "BB2"):
        raise ValueError("variant must be 'BB1' or 'BB2'.")

    if objective not in (1, 2, 5, 6, "R"):
        raise ValueError("Unsupported objective.")

    if objective == "R" and x.size < 2:
        raise ValueError("Rosenbrock requires dimension >= 2.")

    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step must be finite and positive.")

    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 0
    ):
        raise ValueError("max_iter must be a nonnegative integer.")

    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")

    if (
        not np.isfinite(step_min)
        or not np.isfinite(step_max)
        or step_min <= 0
        or step_max < step_min
    ):
        raise ValueError("Invalid step-size bounds.")

    if not step_min <= initial_step <= step_max:
        raise ValueError("initial_step must lie within step bounds.")

    x = x.copy()
    alpha = float(initial_step)

    cost_history = []
    gradient_norms = []
    step_history = []

    for iteration in range(max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            value, gradient = cost(objective, x)
            gradient_norm = float(np.linalg.norm(gradient))

        cost_history.append(value)
        gradient_norms.append(gradient_norm)

        if not np.isfinite(value) or not np.isfinite(gradient_norm):
            return SpectralGradientResult(
                x, iteration, False,
                cost_history, gradient_norms, step_history,
                "Numerical divergence: nonfinite objective or gradient.",
            )

        if gradient_norm <= tol:
            return SpectralGradientResult(
                x, iteration, True,
                cost_history, gradient_norms, step_history,
                "Gradient tolerance satisfied.",
            )

        if iteration == max_iter:
            break

        with np.errstate(over="ignore", invalid="ignore"):
            new_x = x - alpha * gradient
            step = new_x - x

        step_history.append(alpha)

        if not np.all(np.isfinite(new_x)):
            return SpectralGradientResult(
                x, iteration, False,
                cost_history, gradient_norms, step_history,
                "Numerical divergence: nonfinite iterate.",
            )

        with np.errstate(over="ignore", invalid="ignore"):
            _, new_gradient = cost(objective, new_x)
            y = new_gradient - gradient

            sy = float(np.dot(step, y))
            ss = float(np.dot(step, step))
            yy = float(np.dot(y, y))

        if variant == "BB1":
            numerator, denominator = ss, sy
        else:
            numerator, denominator = sy, yy

        if (
            np.isfinite(numerator)
            and np.isfinite(denominator)
            and numerator > 0
            and denominator > 0
        ):
            next_alpha = numerator / denominator
            if not np.isfinite(next_alpha) or next_alpha <= 0:
                next_alpha = initial_step
        else:
            next_alpha = initial_step

        alpha = float(np.clip(next_alpha, step_min, step_max))
        x = new_x

    return SpectralGradientResult(
        x, max_iter, False,
        cost_history, gradient_norms, step_history,
        "Maximum number of iterations reached.",
    )


def BB1(x0, rho=0.25, max_iter=1000, tol=1e-10, objective=1):
    """Convenience wrapper for the BB1 variant."""
    return solve_spectral_gradient(
        x0,
        variant="BB1",
        objective=objective,
        initial_step=rho,
        max_iter=max_iter,
        tol=tol,
    )


def BB2(x0, rho=0.25, max_iter=1000, tol=1e-10, objective=1):
    """Convenience wrapper for the BB2 variant."""
    return solve_spectral_gradient(
        x0,
        variant="BB2",
        objective=objective,
        initial_step=rho,
        max_iter=max_iter,
        tol=tol,
    )
