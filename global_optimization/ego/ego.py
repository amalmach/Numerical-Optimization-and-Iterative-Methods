"""
Efficient Global Optimization (EGO).

Sequential global optimization using a Gaussian process surrogate
and the Expected Improvement acquisition function.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .gaussian_process import GaussianProcess
from .acquisition import expected_improvement
from .hyperparameters import estimate_hyperparameters


@dataclass
class EGOResult:
    """Results from an EGO optimization run."""

    x_best: np.ndarray
    f_best: float
    X: np.ndarray
    y: np.ndarray
    best_history: np.ndarray
    n_evaluations: int


def efficient_global_optimization(
    objective,
    bounds,
    n_initial=9,
    max_evaluations=30,
    theta=0.2,
    sigma=1.0,
    nugget=1e-8,
    xi=0.0,
    n_restarts=10,
    random_state=42,
    estimate_parameters=False,
):
    """
    Minimize a black-box objective using Efficient Global Optimization.

    Parameters
    ----------
    objective : callable
        Objective function accepting a 1D NumPy array.
    bounds : array_like, shape (n_dimensions, 2)
        Lower and upper bounds for each variable.
    n_initial : int
        Number of initial Latin hypercube samples.
    max_evaluations : int
        Maximum number of objective evaluations, including initial samples.
    theta : float
        Gaussian kernel length scale.
    sigma : float
        Gaussian kernel amplitude.
    nugget : float
        Nonnegative covariance regularization.
    xi : float
        Nonnegative exploration parameter for Expected Improvement.
    n_restarts : int
        Number of local acquisition optimization starting points.
    random_state : int
        Random seed for reproducibility.
    estimate_parameters : bool
        If True, re-estimate Gaussian process hyperparameters
        by maximum likelihood at every sequential EGO iteration.

    Returns
    -------
    EGOResult
        Best solution, objective value, and convergence history.
    """

    bounds = np.asarray(bounds, dtype=float)

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dimensions, 2).")

    if not np.all(np.isfinite(bounds)):
        raise ValueError("Bounds must be finite.")

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each lower bound must be less than its upper bound.")

    if not isinstance(n_initial, (int, np.integer)) or n_initial < 2:
        raise ValueError("n_initial must be an integer >= 2.")

    if (
        not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < n_initial
    ):
        raise ValueError("max_evaluations must be >= n_initial.")

    if not isinstance(n_restarts, (int, np.integer)) or n_restarts < 1:
        raise ValueError("n_restarts must be a positive integer.")

    if not np.isfinite(xi) or xi < 0:
        raise ValueError("xi must be nonnegative and finite.")

    n_dimensions = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    width = upper - lower

    rng = np.random.default_rng(random_state)

    # Generate an initial Latin hypercube design on the unit cube.
    sampler = qmc.LatinHypercube(
        d=n_dimensions,
        seed=random_state,
    )

    X_unit = sampler.random(n=n_initial)
    X = qmc.scale(X_unit, lower, upper)

    def evaluate(x):
        value = float(objective(np.asarray(x, dtype=float)))

        if not np.isfinite(value):
            raise ValueError("The objective returned a non-finite value.")

        return value

    y = np.asarray([evaluate(x) for x in X], dtype=float)

    best_history = list(np.minimum.accumulate(y))

    # Sequential EGO iterations.
    while len(y) < max_evaluations:

        # Normalize coordinates to the unit hypercube.
        X_normalized = (X - lower) / width

        current_theta = theta
        current_sigma = sigma

        if estimate_parameters:
            current_theta, current_sigma = estimate_hyperparameters(
                X=X_normalized,
                y=y,
                initial_theta=theta,
                initial_sigma=max(sigma, 1.0),
                nugget=nugget,
            )

        gp = GaussianProcess(
            theta=current_theta,
            sigma=current_sigma,
            nugget=nugget,
        )

        gp.fit(X_normalized, y)
        best_value = float(np.min(y))

        def negative_ei(z):
            mean, std = gp.predict(np.asarray(z).reshape(1, -1))

            ei = expected_improvement(
                mean,
                std,
                best_value=best_value,
                xi=xi,
            )

            return -float(ei[0])

        # Candidate starts for acquisition maximization.
        starts = rng.uniform(
            low=0.0,
            high=1.0,
            size=(n_restarts, n_dimensions),
        )

        best_acquisition = -np.inf
        next_point_unit = None

        for start in starts:
            result = minimize(
                negative_ei,
                x0=start,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * n_dimensions,
            )

            candidate = np.clip(result.x, 0.0, 1.0)
            acquisition_value = -negative_ei(candidate)

            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                next_point_unit = candidate.copy()

        # Avoid wasting evaluations on previously sampled points.
        distances = np.linalg.norm(
            X_normalized - next_point_unit,
            axis=1,
        )

        if np.min(distances) < 1e-6:
            candidates = rng.uniform(
                0.0,
                1.0,
                size=(max(100, 10 * n_restarts), n_dimensions),
            )

            candidate_distances = np.linalg.norm(
                candidates[:, None, :] - X_normalized[None, :, :],
                axis=2,
            )

            valid = np.min(candidate_distances, axis=1) >= 1e-6

            if not np.any(valid):
                raise RuntimeError(
                    "Unable to identify a new evaluation point."
                )

            available = candidates[valid]

            acquisition_values = np.array([
                -negative_ei(candidate)
                for candidate in available
            ])

            next_point_unit = available[np.argmax(acquisition_values)]

        next_point = lower + width * next_point_unit
        next_value = evaluate(next_point)

        X = np.vstack((X, next_point))
        y = np.append(y, next_value)

        best_history.append(min(best_history[-1], next_value))

    best_index = int(np.argmin(y))

    return EGOResult(
        x_best=X[best_index].copy(),
        f_best=float(y[best_index]),
        X=X,
        y=y,
        best_history=np.asarray(best_history),
        n_evaluations=len(y),
    )
