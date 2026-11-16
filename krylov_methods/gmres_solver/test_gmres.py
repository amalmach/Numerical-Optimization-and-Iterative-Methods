"""Regression tests: run `python -m unittest gmres_solver.test_gmres -v`."""

import unittest

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .GMRES import gmres
from .givens import compute_givens_rotation
from .preconditioners import select_preconditioner


class TestGMRES(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2024)
        self.A = rng.standard_normal((12, 12)) + 14 * np.eye(12)
        self.b = rng.standard_normal(12)

    def assert_solution(self, result, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        self.assertTrue(result.converged, result.message)
        self.assertLess(np.linalg.norm(A @ result.x - b), 1e-8)

    def test_unrestarted(self):
        self.assert_solution(gmres(self.A, self.b, rtol=1e-11))

    def test_restarted(self):
        result = gmres(self.A, self.b, restart=3, maxiter=70, rtol=1e-11)
        self.assert_solution(result)
        self.assertGreater(result.cycles, 1)

    def test_preconditioners(self):
        for name in ("jacobi", "lu", "ilu", "ic0", "ssor"):
            with self.subTest(name=name):
                A = self.A.T @ self.A + 5 * np.eye(12) if name == "ic0" else self.A
                result = gmres(sp.csr_matrix(A), self.b, restart=5, maxiter=90,
                               preconditioner=name, rtol=1e-10)
                self.assert_solution(result, A, self.b)

    def test_matrix_free(self):
        self.assert_solution(gmres(lambda v: self.A @ v, self.b, maxiter=12))

    def test_custom(self):
        inverse_diag = lambda v: v / np.diag(self.A)
        result = gmres(self.A, self.b, restart=4, maxiter=80,
                       preconditioner="custom", M=inverse_diag, rtol=1e-11)
        self.assert_solution(result)

    def test_happy_breakdown(self):
        A = np.eye(6)
        b = np.arange(1, 7, dtype=float)
        self.assert_solution(gmres(A, b), A, b)

    def test_complex(self):
        A = self.A.astype(complex) + 1j * np.eye(12)
        b = self.b + 0.5j * self.b[::-1]
        self.assert_solution(gmres(A, b, restart=5, maxiter=70), A, b)

    def test_givens_annihilates(self):
        for a, b in ((3.0, 4.0), (0, 1j), (2 + 3j, 5 - 2j)):
            c, s, r = compute_givens_rotation(a, b)
            self.assertAlmostEqual(abs(-np.conj(s) * a + c * b), 0, places=12)
            self.assertAlmostEqual(abs(c * a + s * b - r), 0, places=12)

    def test_compare_scipy(self):
        expected, info = spl.gmres(self.A, self.b, rtol=1e-10, atol=0,
                                   restart=4, maxiter=30)
        self.assertEqual(info, 0)
        result = gmres(self.A, self.b, restart=4, maxiter=120, rtol=1e-10)
        self.assert_solution(result)
        np.testing.assert_allclose(result.x, expected, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
