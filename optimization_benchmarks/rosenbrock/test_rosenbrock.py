"""Automated tests for the Rosenbrock-type benchmark."""

import numpy as np
import pytest

from optimization_benchmarks.rosenbrock.benchmark import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
)


def test_global_minimum_2d():
    """The objective and gradient vanish at (1, 1)."""
    x = np.ones(2)

    assert np.isclose(rosenbrock(x), 0.0)
    assert np.allclose(rosenbrock_gradient(x), 0.0)


def test_global_minimum_nd():
    """The global minimum is attained at the all-ones vector."""
    for n in (3, 5, 10):
        x = np.ones(n)

        assert np.isclose(rosenbrock(x), 0.0)
        assert np.allclose(rosenbrock_gradient(x), 0.0)


def test_known_objective_value():
    """Check a known two-dimensional objective value."""
    x = np.array([0.0, 0.0])

    assert np.isclose(rosenbrock(x), 1.0)


def test_known_gradient():
    """Check the analytical gradient at a known point."""
    x = np.array([0.0, 0.0])

    expected = np.array([-2.0, 0.0])

    assert np.allclose(rosenbrock_gradient(x), expected)


def test_known_hessian():
    """Check the analytical Hessian at the global minimizer."""
    x = np.ones(2)

    expected = np.array([
        [10.0, -4.0],
        [-4.0, 2.0],
    ])

    assert np.allclose(rosenbrock_hessian(x), expected)


def test_hessian_symmetry():
    """The analytical Hessian must be symmetric."""
    x = np.array([-1.2, 1.0, 0.5, 1.5])

    H = rosenbrock_hessian(x)

    assert np.allclose(H, H.T)


def test_gradient_finite_differences():
    """Compare the analytical gradient with central differences."""
    x = np.array([-1.2, 1.0, 0.5, 1.5])
    h = 1e-5

    gradient_fd = np.zeros_like(x)

    for i in range(x.size):
        direction = np.zeros_like(x)
        direction[i] = h

        gradient_fd[i] = (
            rosenbrock(x + direction)
            - rosenbrock(x - direction)
        ) / (2.0 * h)

    assert np.allclose(
        rosenbrock_gradient(x),
        gradient_fd,
        atol=1e-7,
        rtol=1e-7,
    )


def test_hessian_finite_differences():
    """Compare the Hessian with central differences of the gradient."""
    x = np.array([-1.2, 1.0, 0.5, 1.5])
    h = 1e-5

    H_fd = np.zeros((x.size, x.size))

    for i in range(x.size):
        direction = np.zeros_like(x)
        direction[i] = h

        H_fd[:, i] = (
            rosenbrock_gradient(x + direction)
            - rosenbrock_gradient(x - direction)
        ) / (2.0 * h)

    assert np.allclose(
        rosenbrock_hessian(x),
        H_fd,
        atol=1e-7,
        rtol=1e-7,
    )


def test_hessian_shape():
    """The Hessian must have dimensions (n, n)."""
    x = np.array([0.5, -0.5, 1.0, 2.0, 0.0])

    assert rosenbrock_hessian(x).shape == (5, 5)


def test_invalid_dimension():
    """Reject input vectors with fewer than two components."""
    x = np.array([1.0])

    with pytest.raises(ValueError):
        rosenbrock(x)

    with pytest.raises(ValueError):
        rosenbrock_gradient(x)

    with pytest.raises(ValueError):
        rosenbrock_hessian(x)
