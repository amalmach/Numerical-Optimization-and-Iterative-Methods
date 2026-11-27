"""
Reproducible example of the Rosenbrock-type benchmark.

Run from the repository root:

    python -m optimization_benchmarks.rosenbrock.example
"""

import numpy as np

from .benchmark import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
)


def display_results(x, label):
    """Display objective value, derivatives, and Hessian eigenvalues."""

    value = rosenbrock(x)
    gradient = rosenbrock_gradient(x)
    hessian = rosenbrock_hessian(x)

    eigenvalues = np.linalg.eigvalsh(hessian)

    print(f"\n{label}")
    print("-" * 55)

    print("Point:", x)
    print(f"Objective value: {value:.8f}")
    print("Gradient:", gradient)
    print(f"Gradient norm: {np.linalg.norm(gradient):.8e}")

    print("\nHessian:")
    print(hessian)

    print("\nHessian eigenvalues:")
    print(eigenvalues)


def main():
    """Run the Rosenbrock benchmark demonstration."""

    print("=" * 60)
    print("ROSENBROCK OPTIMIZATION BENCHMARK")
    print("=" * 60)

    x0 = np.array([-1.2, 1.0])
    x_star = np.ones(2)

    display_results(x0, "Initial point")
    display_results(x_star, "Global minimizer")

    assert np.isclose(rosenbrock(x_star), 0.0)
    assert np.allclose(rosenbrock_gradient(x_star), 0.0)

    print("\nBenchmark validation passed.")


if __name__ == "__main__":
    main()
