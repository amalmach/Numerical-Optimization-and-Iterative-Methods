"""
Maximum-likelihood hyperparameter estimation for Gaussian processes.

Estimates the squared-exponential kernel length scale and
covariance amplitude by minimizing the negative log marginal
likelihood of a zero-mean Gaussian process.
"""

import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .gaussian_process import gaussian_kernel


def negative_log_likelihood(
    log_parameters,
    X,
    y,
    nugget=1e-8,
):
    """
    Compute the Gaussian process negative log marginal likelihood.

    Parameters
    ----------
    log_parameters : array_like, shape (2,)
        Logarithms of theta and sigma.
    X : ndarray, shape (n_samples, n_features)
        Training coordinates.
    y : ndarray, shape (n_samples,)
        Observed objective values.
    nugget : float
        Nonnegative diagonal regularization.

    Returns
    -------
    float
        Negative log marginal likelihood.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.shape[0] != len(y) or len(y) == 0:
        raise ValueError("X and y must contain matching, nonempty samples.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Training data must be finite.")

    if not np.isfinite(nugget) or nugget < 0:
        raise ValueError("nugget must be nonnegative and finite.")

    parameters = np.asarray(log_parameters, dtype=float)

    if parameters.shape != (2,) or not np.all(np.isfinite(parameters)):
        raise ValueError("log_parameters must contain two finite values.")

    theta, sigma = np.exp(parameters)

    K = gaussian_kernel(
        X,
        X,
        theta=theta,
        sigma=sigma,
    )

    K += nugget * np.eye(len(y))

    try:
        factor = cho_factor(
            K,
            lower=True,
            check_finite=True,
        )

        alpha = cho_solve(factor, y)

        L = factor[0]

        log_determinant = 2.0 * np.sum(
            np.log(np.diag(L))
        )

        n = len(y)

        return float(
            0.5 * np.dot(y, alpha)
            + 0.5 * log_determinant
            + 0.5 * n * np.log(2.0 * np.pi)
        )

    except np.linalg.LinAlgError:
        return np.inf


def estimate_hyperparameters(
    X,
    y,
    initial_theta=0.2,
    initial_sigma=10.0,
    theta_bounds=(0.01, 2.0),
    sigma_bounds=(1e-3, 1e4),
    nugget=1e-8,
):
    """
    Estimate Gaussian process hyperparameters by maximum likelihood.

    Parameters
    ----------
    X : ndarray
        Training coordinates.
    y : ndarray
        Observed function values.
    initial_theta : float
        Initial kernel length scale.
    initial_sigma : float
        Initial covariance amplitude.
    theta_bounds : tuple
        Admissible range for theta.
    sigma_bounds : tuple
        Admissible range for sigma.
    nugget : float
        Covariance regularization.

    Returns
    -------
    tuple
        Estimated (theta, sigma).
    """

    for name, value, limits in (
        ("theta", initial_theta, theta_bounds),
        ("sigma", initial_sigma, sigma_bounds),
    ):
        if (
            not np.isfinite(value)
            or value <= 0
            or len(limits) != 2
            or not np.all(np.isfinite(limits))
            or limits[0] <= 0
            or limits[0] >= limits[1]
            or not limits[0] <= value <= limits[1]
        ):
            raise ValueError(
                f"Invalid initial value or bounds for {name}."
            )

    log_initial = np.log([
        initial_theta,
        initial_sigma,
    ])

    log_bounds = [
        tuple(np.log(theta_bounds)),
        tuple(np.log(sigma_bounds)),
    ]

    result = minimize(
        negative_log_likelihood,
        x0=log_initial,
        args=(X, y, nugget),
        method="L-BFGS-B",
        bounds=log_bounds,
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Hyperparameter optimization failed: {result.message}"
        )

    theta, sigma = np.exp(result.x)

    return float(theta), float(sigma)
