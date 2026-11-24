"""Reproducible examples of fixed-step gradient descent."""

import numpy as np

from gradient_methods.fixed_step_gradient_solver import (
    solve_fixed_step_gradient,
)


def run_example():
    """Demonstrate convergence and step-size sensitivity."""

    n = 10
    u0 = np.zeros(n)
    tol = 1e-10
    max_iter = 1500

    print("=" * 65)
    print("FIXED-STEP GRADIENT DESCENT")
    print("=" * 65)

    for objective in (1, 2):
        target = (
            np.ones(n)
            if objective == 1
            else np.arange(1, n + 1, dtype=float)
        )

        print(f"\nObjective J{objective}")
        print("-" * 65)
        print(f"{'Step size':>12} {'Iterations':>12} "
              f"{'Converged':>12} {'Final error':>15}")

        for rho in (0.1, 0.25, 0.5, 0.75, 1.0):
            result = solve_fixed_step_gradient(
                u0=u0,
                rho=rho,
                max_iter=max_iter,
                tol=tol,
                objective=objective,
            )

            error = np.linalg.norm(result.solution - target)

            print(
                f"{rho:12.2f} "
                f"{result.iterations:12d} "
                f"{str(result.converged):>12} "
                f"{error:15.6e}"
            )

    print("\n" + "=" * 65)
    print("Step-size observations")
    print("=" * 65)
    print("0 < rho < 1 : convergence for these quadratic objectives")
    print("rho = 0.5   : exact minimizer after one update")
    print("rho = 1.0   : oscillation from a nonoptimal initial point")


if __name__ == "__main__":
    run_example()
