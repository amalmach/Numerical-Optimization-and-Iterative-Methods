"""
Example: Efficient Global Optimization on the Modified Branin function.

Compares fixed and maximum-likelihood-estimated Gaussian process
hyperparameters using identical initial designs and evaluation budgets.

Run from the repository root:

    python -m global_optimization.ego.example
"""

import numpy as np

from .benchmarks import modified_branin_normalized
from .ego import efficient_global_optimization


def objective(x):
    """Modified Branin function on the unit square."""
    return float(modified_branin_normalized(x[0], x[1]))


def main():
    """Compare fixed and adaptive EGO on the Modified Branin benchmark."""

    parameters = dict(
        objective=objective,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        n_initial=9,
        max_evaluations=30,
        theta=0.2,
        sigma=1.0,
        nugget=1e-8,
        xi=0.0,
        n_restarts=10,
        random_state=42,
    )

    print("=" * 65)
    print("EFFICIENT GLOBAL OPTIMIZATION")
    print("Modified Branin benchmark")
    print("=" * 65)

    results = {}

    for name, adaptive in [
        ("Fixed EGO", False),
        ("Adaptive EGO", True),
    ]:
        result = efficient_global_optimization(
            **parameters,
            estimate_parameters=adaptive,
        )

        results[name] = result

        original_coordinates = np.array([
            15.0 * result.x_best[0] - 5.0,
            15.0 * result.x_best[1],
        ])

        print(f"\n{name}")
        print("-" * 65)
        print(f"Evaluations:           {result.n_evaluations}")
        print(f"Initial best value:    {result.best_history[8]:.10f}")
        print(f"Final best value:      {result.f_best:.10f}")
        print(f"Best normalized point: {result.x_best}")
        print(f"Original coordinates:  {original_coordinates}")

    fixed = results["Fixed EGO"]
    adaptive = results["Adaptive EGO"]

    difference = fixed.f_best - adaptive.f_best

    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)

    print(f"Fixed EGO:      {fixed.f_best:.10f}")
    print(f"Adaptive EGO:   {adaptive.f_best:.10f}")
    print(f"Difference:     {difference:.10f}")

    if difference > 1e-8:
        print("\nAdaptive EGO achieved the lower objective value.")
    elif difference < -1e-8:
        print("\nFixed EGO achieved the lower objective value.")
    else:
        print("\nBoth configurations achieved similar results.")


if __name__ == "__main__":
    main()
