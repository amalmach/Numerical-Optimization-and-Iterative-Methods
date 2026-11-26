"""Regression tests for the Barzilai–Borwein spectral gradient solver."""

import unittest

import numpy as np

from gradient_methods.spectral_gradient_solver.SpectralGradient import (
    BB1,
    BB2,
    solve_spectral_gradient,
)
from gradient_methods.spectral_gradient_solver.objectives import cost


class TestSpectralGradient(unittest.TestCase):

    def test_quadratic_convergence(self):
        for objective in (1, 2):
            target = (
                np.ones(10)
                if objective == 1
                else np.arange(1, 11, dtype=float)
            )

            for solver in (BB1, BB2):
                with self.subTest(objective=objective, solver=solver.__name__):
                    result = solver(
                        np.zeros(10),
                        rho=0.25,
                        max_iter=100,
                        tol=1e-10,
                        objective=objective,
                    )
                    self.assertTrue(result.converged)
                    np.testing.assert_allclose(
                        result.solution, target, atol=1e-8
                    )

    def test_all_objectives(self):
        for objective in (1, 2, 5, 6, "R"):
            for variant in ("BB1", "BB2"):
                with self.subTest(objective=objective, variant=variant):
                    result = solve_spectral_gradient(
                        np.zeros(10),
                        variant=variant,
                        objective=objective,
                        initial_step=0.25,
                        max_iter=300,
                        tol=1e-8,
                    )
                    self.assertTrue(result.converged)
                    self.assertLessEqual(result.gradient_norms[-1], 1e-8)

    def test_history_lengths(self):
        result = BB1(np.zeros(10), max_iter=100)

        self.assertEqual(
            len(result.cost_history), result.iterations + 1
        )
        self.assertEqual(
            len(result.gradient_norms), result.iterations + 1
        )
        self.assertEqual(
            len(result.step_history), result.iterations
        )

    def test_initial_convergence(self):
        result = BB1(np.ones(5))

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.cost_history), 1)
        self.assertEqual(len(result.step_history), 0)

    def test_zero_iteration_budget(self):
        result = BB2(np.zeros(5), max_iter=0)

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.cost_history), 1)

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            solve_spectral_gradient(np.zeros(5), variant="BB3")

    def test_invalid_step(self):
        for step in (0, -1, np.nan, np.inf):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    BB1(np.zeros(5), rho=step)

    def test_invalid_iteration_budget(self):
        for budget in (-1, 1.5, True):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    BB2(np.zeros(5), max_iter=budget)

    def test_invalid_tolerance(self):
        for tol in (-1, np.nan, np.inf):
            with self.subTest(tol=tol):
                with self.assertRaises(ValueError):
                    BB1(np.zeros(5), tol=tol)

    def test_invalid_dimension(self):
        for x in (np.array([]), np.zeros((2, 2))):
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError):
                    BB1(x)

    def test_rosenbrock_dimension(self):
        with self.assertRaises(ValueError):
            BB1(np.zeros(1), objective="R")

    def test_step_bounds(self):
        result = BB1(
            np.zeros(10),
            rho=0.25,
            max_iter=100,
            step_min=1e-4,
        ) if False else solve_spectral_gradient(
            np.zeros(10),
            variant="BB1",
            objective=5,
            initial_step=0.25,
            max_iter=100,
            step_min=1e-4,
            step_max=0.5,
        )

        self.assertTrue(
            all(1e-4 <= step <= 0.5 for step in result.step_history)
        )

    def test_objective_gradient_consistency(self):
        for objective in (1, 2, 5, 6, "R"):
            with self.subTest(objective=objective):
                value, gradient = cost(objective, np.zeros(10))
                self.assertTrue(np.isfinite(value))
                self.assertEqual(gradient.shape, (10,))
                self.assertTrue(np.all(np.isfinite(gradient)))


if __name__ == "__main__":
    unittest.main()
