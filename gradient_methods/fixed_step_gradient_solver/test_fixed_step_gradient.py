"""Regression tests for fixed-step gradient descent."""

import unittest

import numpy as np

from gradient_methods.fixed_step_gradient_solver.FixedStepGradient import (
    GF,
    J1,
    J2,
    cost,
    solve_fixed_step_gradient,
)


class TestFixedStepGradient(unittest.TestCase):

    def test_j1_and_gradient(self):
        u = np.array([0.0, 2.0, 3.0])
        value, gradient = J1(u)

        self.assertAlmostEqual(value, 6.0)
        np.testing.assert_allclose(gradient, [-2.0, 2.0, 4.0])

    def test_j2_and_gradient(self):
        u = np.zeros(3)
        value, gradient = J2(u)

        self.assertAlmostEqual(value, 14.0)
        np.testing.assert_allclose(gradient, [-2.0, -4.0, -6.0])

    def test_j1_convergence(self):
        result = solve_fixed_step_gradient(
            np.zeros(10),
            rho=0.25,
            max_iter=1500,
            tol=1e-10,
            objective=1,
        )

        self.assertTrue(result.converged)
        np.testing.assert_allclose(
            result.solution,
            np.ones(10),
            atol=1e-9,
        )
        self.assertLessEqual(result.gradient_norms[-1], 1e-10)
        self.assertEqual(
            len(result.cost_history),
            result.iterations + 1,
        )

    def test_j2_convergence(self):
        result = solve_fixed_step_gradient(
            np.zeros(10),
            rho=0.25,
            max_iter=1500,
            tol=1e-10,
            objective=2,
        )

        self.assertTrue(result.converged)
        np.testing.assert_allclose(
            result.solution,
            np.arange(1, 11),
            atol=1e-9,
        )

    def test_optimal_step_for_quadratics(self):
        for objective in (1, 2):
            with self.subTest(objective=objective):
                result = solve_fixed_step_gradient(
                    np.zeros(5),
                    rho=0.5,
                    max_iter=10,
                    tol=1e-12,
                    objective=objective,
                )

                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 1)
                self.assertAlmostEqual(result.cost_history[-1], 0.0)

    def test_oscillating_step(self):
        result = solve_fixed_step_gradient(
            np.zeros(5),
            rho=1.0,
            max_iter=20,
            tol=1e-10,
            objective=1,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 20)

    def test_zero_iteration_budget(self):
        result = solve_fixed_step_gradient(
            np.zeros(3),
            max_iter=0,
            objective=1,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.cost_history), 1)

    def test_initial_point_already_optimal(self):
        result = solve_fixed_step_gradient(
            np.ones(4),
            max_iter=0,
            objective=1,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_invalid_inputs(self):
        invalid_cases = [
            {"u0": []},
            {"u0": [[1.0, 2.0]]},
            {"u0": [np.nan]},
            {"u0": [0.0], "rho": 0.0},
            {"u0": [0.0], "rho": -0.1},
            {"u0": [0.0], "rho": np.inf},
            {"u0": [0.0], "max_iter": -1},
            {"u0": [0.0], "max_iter": 1.5},
            {"u0": [0.0], "tol": -1.0},
            {"u0": [0.0], "objective": 3},
        ]

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    solve_fixed_step_gradient(**parameters)

    def test_objective_selector(self):
        u = np.zeros(3)

        self.assertAlmostEqual(cost(u, 1)[0], 3.0)
        self.assertAlmostEqual(cost(u, 2)[0], 14.0)

        with self.assertRaises(ValueError):
            cost(u, 3)

    def test_original_gf_interface(self):
        solution, iterations, history = GF(
            np.zeros(4),
            0.25,
            100,
            1e-10,
            1,
        )

        np.testing.assert_allclose(
            solution,
            np.ones(4),
            atol=1e-9,
        )
        self.assertGreater(iterations, 0)
        self.assertEqual(len(history), iterations + 1)


if __name__ == "__main__":
    unittest.main()
