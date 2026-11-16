"""Run: python -m unittest bicgstab_solver.test_bicgstab -v"""

import unittest

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .BiCGSTAB import bicgstab, bicgstab_solver, solve_bicgstab


class TestBiCGSTAB(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2024)
        self.A = rng.standard_normal((20, 20)) + 16 * np.eye(20)
        self.b = rng.standard_normal(20)

    def assert_solved(self, result, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        self.assertTrue(result.converged, result.message)
        self.assertLess(np.linalg.norm(A @ result.solution - b), 1e-8)

    def test_dense(self):
        self.assert_solved(solve_bicgstab(self.A, self.b, rtol=1e-11))

    def test_sparse(self):
        result = solve_bicgstab(sp.csr_matrix(self.A), self.b, rtol=1e-11)
        self.assert_solved(result)

    def test_preconditioners(self):
        for name in ("jacobi", "ilu", "lu", "ssor", "ic0"):
            with self.subTest(preconditioner=name):
                A = self.A.T @ self.A + 5 * np.eye(20) if name == "ic0" else self.A
                result = solve_bicgstab(sp.csr_matrix(A), self.b,
                                        preconditioner=name, rtol=1e-10)
                self.assert_solved(result, A)

    def test_custom(self):
        result = solve_bicgstab(self.A, self.b,
                                preconditioner="custom",
                                M=lambda v: v / np.diag(self.A), rtol=1e-11)
        self.assert_solved(result)

    def test_matrix_free(self):
        result = solve_bicgstab(lambda v: self.A @ v, self.b, rtol=1e-11)
        self.assert_solved(result)

    def test_complex(self):
        A = self.A + 1j * np.eye(20)
        b = self.b + 1j * self.b[::-1]
        result = solve_bicgstab(A, b, rtol=1e-11)
        self.assert_solved(result, A, b)

    def test_initial_convergence(self):
        x = np.linalg.solve(self.A, self.b)
        result = solve_bicgstab(self.A, self.b, x0=x)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_zero_budget(self):
        result = solve_bicgstab(self.A, self.b, max_iter=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_original_dict_interface(self):
        result = bicgstab(self.A, self.b, tol=1e-10)
        self.assertTrue(result["converged"])
        self.assertEqual(result["history"].shape[1], 20)

    def test_legacy_x_only_interface(self):
        x = bicgstab_solver(self.A, self.b, rtol=1e-10)
        np.testing.assert_allclose(self.A @ x, self.b, atol=1e-8)

    def test_compare_scipy(self):
        x_scipy, info = spl.bicgstab(self.A, self.b, rtol=1e-10, atol=0)
        self.assertEqual(info, 0)
        result = solve_bicgstab(self.A, self.b, rtol=1e-10)
        self.assert_solved(result)
        np.testing.assert_allclose(result.solution, x_scipy, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
