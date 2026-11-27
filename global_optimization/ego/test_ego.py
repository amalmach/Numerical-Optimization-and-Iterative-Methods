"""
Automated tests for Efficient Global Optimization components.
"""

import numpy as np

from global_optimization.ego.benchmarks import (
    modified_branin,
    modified_branin_normalized,
)


def test_modified_branin_normalization():
    """Original and normalized coordinates must give identical results."""

    points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ]

    for y1, y2 in points:
        x1 = 15.0 * y1 - 5.0
        x2 = 15.0 * y2

        original = modified_branin(x1, x2)
        normalized = modified_branin_normalized(y1, y2)

        assert np.isclose(original, normalized)


def test_modified_branin_reference_values():
    """Verify independently recorded benchmark values."""

    expected = [
        ((0.0, 0.0), 283.12909601),
        ((0.5, 0.5), 36.62996441),
        ((1.0, 1.0), 195.87219088),
    ]

    for (y1, y2), reference in expected:
        value = modified_branin_normalized(y1, y2)

        assert np.isclose(value, reference, atol=1e-7)


def test_modified_branin_vectorization():
    """Verify evaluation on NumPy arrays."""

    y1 = np.array([0.0, 0.5, 1.0])
    y2 = np.array([0.0, 0.5, 1.0])

    values = modified_branin_normalized(y1, y2)

    assert values.shape == (3,)
    assert np.all(np.isfinite(values))


from global_optimization.ego.gaussian_process import (
    GaussianProcess,
    gaussian_kernel,
)


def test_gaussian_kernel_symmetry():
    """The covariance matrix must be symmetric."""

    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    K = gaussian_kernel(X, X)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)


def test_gaussian_process_training_predictions():
    """The GP should approximately interpolate training observations."""

    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    y = modified_branin_normalized(X[:, 0], X[:, 1])

    gp = GaussianProcess(
        theta=0.2,
        sigma=1.0,
        nugget=1e-10,
    )

    gp.fit(X, y)

    mean, std = gp.predict(X)

    assert np.allclose(mean, y, atol=1e-6)
    assert np.all(std >= 0)
    assert np.all(np.isfinite(std))


def test_gaussian_process_prediction_shapes():
    """Check prediction shapes at previously unseen points."""

    X_train = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    y_train = modified_branin_normalized(
        X_train[:, 0],
        X_train[:, 1],
    )

    gp = GaussianProcess(theta=0.3)
    gp.fit(X_train, y_train)

    X_test = np.array([
        [0.25, 0.25],
        [0.75, 0.75],
    ])

    mean, std = gp.predict(X_test)

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))


from global_optimization.ego.acquisition import expected_improvement


def test_expected_improvement_zero_uncertainty():
    """EI has a well-defined limit when predictive std is zero."""

    mean = np.array([1.0, 3.0, 5.0])
    std = np.zeros(3)

    ei = expected_improvement(
        mean,
        std,
        best_value=3.0,
    )

    assert np.allclose(ei, [2.0, 0.0, 0.0])


def test_expected_improvement_nonnegative():
    """Expected Improvement must be nonnegative."""

    mean = np.array([0.0, 1.0, 2.0])
    std = np.array([0.1, 0.5, 1.0])

    ei = expected_improvement(
        mean,
        std,
        best_value=1.0,
    )

    assert ei.shape == (3,)
    assert np.all(np.isfinite(ei))
    assert np.all(ei >= 0)


def test_expected_improvement_scalar():
    """Verify the analytical EI value when improvement is zero."""

    value = expected_improvement(
        mean=1.0,
        std=2.0,
        best_value=1.0,
    )

    expected = 2.0 / np.sqrt(2.0 * np.pi)

    assert np.isclose(value, expected)


from global_optimization.ego.ego import efficient_global_optimization


def test_ego_evaluation_budget():
    """EGO must respect the prescribed objective evaluation budget."""

    def objective(x):
        return np.sum((x - 0.3) ** 2)

    result = efficient_global_optimization(
        objective=objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        n_initial=5,
        max_evaluations=10,
        n_restarts=3,
        random_state=42,
    )

    assert result.n_evaluations == 10
    assert result.X.shape == (10, 2)
    assert result.y.shape == (10,)
    assert result.best_history.shape == (10,)


def test_ego_convergence_history():
    """The best objective value must never increase."""

    def objective(x):
        return np.sum((x - 0.3) ** 2)

    result = efficient_global_optimization(
        objective=objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        n_initial=5,
        max_evaluations=10,
        n_restarts=3,
        random_state=42,
    )

    assert np.all(np.diff(result.best_history) <= 1e-12)
    assert np.isclose(result.best_history[-1], result.f_best)


def test_ego_reproducibility():
    """Identical random seeds must produce identical results."""

    def objective(x):
        return np.sum((x - 0.3) ** 2)

    parameters = dict(
        objective=objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        n_initial=5,
        max_evaluations=10,
        n_restarts=3,
        random_state=42,
    )

    result1 = efficient_global_optimization(**parameters)
    result2 = efficient_global_optimization(**parameters)

    assert np.allclose(result1.X, result2.X)
    assert np.allclose(result1.y, result2.y)
    assert np.allclose(result1.best_history, result2.best_history)


def test_ego_solution_bounds():
    """All evaluated points must remain within the search domain."""

    def objective(x):
        return np.sum((x - 0.3) ** 2)

    bounds = np.array([
        [-2.0, 2.0],
        [-1.0, 3.0],
    ])

    result = efficient_global_optimization(
        objective=objective,
        bounds=bounds,
        n_initial=5,
        max_evaluations=10,
        n_restarts=3,
        random_state=42,
    )

    assert np.all(result.X >= bounds[:, 0] - 1e-12)
    assert np.all(result.X <= bounds[:, 1] + 1e-12)

    assert np.all(np.isfinite(result.y))
    assert np.isfinite(result.f_best)

    best_index = np.argmin(result.y)

    assert np.allclose(result.x_best, result.X[best_index])
    assert np.isclose(result.f_best, result.y[best_index])


from global_optimization.ego.hyperparameters import (
    negative_log_likelihood,
    estimate_hyperparameters,
)


def test_negative_log_likelihood_finite():
    """The GP negative log marginal likelihood must be finite."""

    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    y = modified_branin_normalized(X[:, 0], X[:, 1])

    value = negative_log_likelihood(
        log_parameters=np.log([0.2, 100.0]),
        X=X,
        y=y,
    )

    assert np.isfinite(value)


def test_hyperparameter_estimation_bounds():
    """Estimated parameters must satisfy their prescribed bounds."""

    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    y = modified_branin_normalized(X[:, 0], X[:, 1])

    theta, sigma = estimate_hyperparameters(
        X,
        y,
        initial_theta=0.2,
        initial_sigma=100.0,
    )

    assert 0.01 <= theta <= 2.0
    assert 1e-3 <= sigma <= 1e4
    assert np.isfinite(theta)
    assert np.isfinite(sigma)


def test_hyperparameter_estimation_improves_likelihood():
    """Optimization should not increase the negative log likelihood."""

    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    y = modified_branin_normalized(X[:, 0], X[:, 1])

    initial_parameters = np.log([0.2, 100.0])

    initial_nll = negative_log_likelihood(
        initial_parameters,
        X,
        y,
    )

    theta, sigma = estimate_hyperparameters(
        X,
        y,
        initial_theta=0.2,
        initial_sigma=100.0,
    )

    optimized_nll = negative_log_likelihood(
        np.log([theta, sigma]),
        X,
        y,
    )

    assert optimized_nll <= initial_nll + 1e-6


def test_ego_adaptive_hyperparameters():
    """Adaptive EGO must complete its prescribed evaluation budget."""

    def objective(x):
        return np.sum((x - 0.3) ** 2)

    result = efficient_global_optimization(
        objective=objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        n_initial=5,
        max_evaluations=10,
        theta=0.2,
        sigma=1.0,
        nugget=1e-8,
        n_restarts=3,
        random_state=42,
        estimate_parameters=True,
    )

    assert result.n_evaluations == 10
    assert result.X.shape == (10, 2)
    assert result.y.shape == (10,)
    assert result.best_history.shape == (10,)

    assert np.all(np.isfinite(result.y))
    assert np.isfinite(result.f_best)

    assert np.all(np.diff(result.best_history) <= 1e-12)
    assert np.isclose(result.f_best, np.min(result.y))

    assert np.all(result.X >= 0.0)
    assert np.all(result.X <= 1.0)
