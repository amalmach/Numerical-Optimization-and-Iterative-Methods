"""Regression tests for the Steepest Descent linear-system solver."""

import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .SteepestDescent import solve_steepest_descent, steepest_descent


class TestSteepestDescent(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[4., 1.], [1., 3.]])
        self.b = np.array([1., 2.])

    def check(self, A, b=None, **kwargs):
        if b is None:
            b = self.b
        result = solve_steepest_descent(A, b, rtol=1e-10, atol=1e-12,
                                         max_iter=2000, **kwargs)
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.solution, np.linalg.solve(self.A, self.b),
                                   rtol=1e-8, atol=1e-8)
        return result

    def test_dense(self):
        result = self.check(self.A, store_history=True)
        self.assertEqual(result.history.shape[0], len(result.residual_norms))

    def test_sparse(self):
        self.check(sparse.csr_matrix(self.A))

    def test_operator(self):
        self.check(LinearOperator(self.A.shape, matvec=lambda v: self.A @ v))

    def test_callable(self):
        self.check(lambda v: self.A @ v)

    def test_complex_hermitian(self):
        A = np.array([[4., 1.+1j], [1.-1j, 3.]])
        b = np.array([1.+1j, 2.-1j])
        result = solve_steepest_descent(A, b, rtol=1e-10,
                                         atol=1e-12, max_iter=2000)
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.solution, np.linalg.solve(A, b), atol=1e-8)

    def test_complex_callable_real_rhs(self):
        A = np.array([[4., 1.+1j], [1.-1j, 3.]])
        b = np.array([1., 2.])
        result = solve_steepest_descent(lambda v: A @ v, b, max_iter=2000,
                                         rtol=1e-10)
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.solution, np.linalg.solve(A, b), atol=1e-8)

    def test_initial_convergence(self):
        x = np.linalg.solve(self.A, self.b)
        result = solve_steepest_descent(self.A, self.b, x0=x)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.converged)

    def test_zero_budget(self):
        result = solve_steepest_descent(self.A, self.b, max_iter=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_legacy_dict(self):
        data = steepest_descent(self.A, self.b, tol=1e-10)
        self.assertTrue(data["converged"])
        self.assertEqual(data["history"].shape[0], data["iterations"] + 1)

    def test_compare_cg_solution(self):
        ours = self.check(self.A)
        ref, info = cg(self.A, self.b, rtol=1e-10)
        self.assertEqual(info, 0)
        np.testing.assert_allclose(ours.solution, ref, atol=1e-9)

    def test_reject_non_spd(self):
        with self.assertRaises(ValueError):
            solve_steepest_descent(np.array([[1., 2.], [2., 1.]]), self.b)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            solve_steepest_descent(self.A, self.b, rtol=-1)


if __name__ == "__main__":
    unittest.main()
