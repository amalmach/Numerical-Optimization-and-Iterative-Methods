"""
Gaussian process regression for Efficient Global Optimization.

Implements a zero-mean Gaussian process with a squared-exponential
covariance kernel and Cholesky-based numerical linear algebra.
"""

import numpy as np

from scipy.linalg import (
    cho_factor,
    cho_solve,
    solve_triangular,
)


def gaussian_kernel(X1, X2, theta=0.2, sigma=1.0):
    """
    Squared-exponential covariance kernel.

    k(x, y) = sigma² * exp(-||x-y||² / theta²)

    Parameters
    ----------
    X1, X2 : ndarray
        Input arrays of shape (n_samples, n_features).
    theta : float
        Positive kernel length-scale parameter.
    sigma : float
        Positive covariance amplitude.

    Returns
    -------
    ndarray
        Covariance matrix of shape (len(X1), len(X2)).
    """
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))

    if X1.shape[1] != X2.shape[1]:
        raise ValueError("Input dimensions must match.")

    if not np.all(np.isfinite(X1)) or not np.all(np.isfinite(X2)):
        raise ValueError("Input coordinates must be finite.")

    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be positive and finite.")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    squared_distances = np.sum(
        (X1[:, None, :] - X2[None, :, :]) ** 2,
        axis=2,
    )

    return sigma**2 * np.exp(-squared_distances / theta**2)


class GaussianProcess:
    """
    Zero-mean Gaussian process regression.

    Parameters
    ----------
    theta : float
        Kernel length-scale parameter.
    sigma : float
        Covariance amplitude.
    nugget : float
        Nonnegative diagonal regularization.
    """

    def __init__(self, theta=0.2, sigma=1.0, nugget=1e-8):
        if not np.isfinite(nugget) or nugget < 0:
            raise ValueError("nugget must be nonnegative and finite.")

        self.theta = theta
        self.sigma = sigma
        self.nugget = nugget

        self.X_train = None
        self.y_train = None
        self.cholesky = None
        self.alpha = None

    def fit(self, X, y):
        """Fit the Gaussian process to observed data."""

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.size:
            raise ValueError("X and y must contain the same number of samples.")

        if X.shape[0] == 0:
            raise ValueError("At least one observation is required.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Training data must be finite.")

        K = gaussian_kernel(
            X,
            X,
            theta=self.theta,
            sigma=self.sigma,
        )

        K += self.nugget * np.eye(len(X))

        self.cholesky = cho_factor(
            K,
            lower=True,
            check_finite=True,
        )

        self.alpha = cho_solve(self.cholesky, y)

        self.X_train = X.copy()
        self.y_train = y.copy()

        return self

    def predict(self, X, return_std=True):
        """
        Predict the Gaussian process mean and standard deviation.

        Returns
        -------
        mean : ndarray
            Predictive mean.
        std : ndarray, optional
            Predictive standard deviation.
        """

        if self.X_train is None:
            raise RuntimeError("Fit the Gaussian process before prediction.")

        X = np.atleast_2d(np.asarray(X, dtype=float))

        K_cross = gaussian_kernel(
            X,
            self.X_train,
            theta=self.theta,
            sigma=self.sigma,
        )

        mean = K_cross @ self.alpha

        if not return_std:
            return mean

        L, lower = self.cholesky

        v = solve_triangular(
            L,
            K_cross.T,
            lower=lower,
            check_finite=True,
        )

        variance = self.sigma**2 - np.sum(v**2, axis=0)

        variance = np.maximum(variance, 0.0)

        std = np.sqrt(variance)

        return mean, std
