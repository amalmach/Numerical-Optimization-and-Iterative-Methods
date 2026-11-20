"""Lanczos numerical regression tests."""

import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from .Lanczos import lanczos_tridiagonalize
from .galerkin import lanczos_galerkin


class TestLanczos(unittest.TestCase):
    def setUp(self):
        n = 12
        self.A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        self.b = np.random.default_rng(42).normal(size=n)

    def test_projection(self):
        result = lanczos_tridiagonalize(self.A, self.b, m=8)
        Q, T = result.basis, result.tridiagonal
        np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-10)
        np.testing.assert_allclose(Q.T @ self.A @ Q, T, atol=1e-10)
        self.assertEqual(T.shape, (8, 8))

    def test_lanczos_relation(self):
        result = lanczos_tridiagonalize(self.A, self.b, m=8)
        Q, T = result.basis, result.tridiagonal
        e = np.zeros(8)
        e[-1] = 1
        np.testing.assert_allclose(self.A @ Q, Q @ T + result.beta_next * np.outer(result.next_vector, e), atol=1e-10)

    def test_ritz_bounds(self):
        result = lanczos_tridiagonalize(self.A, self.b, m=8)
        exact = np.linalg.eigvalsh(self.A)
        self.assertGreaterEqual(result.ritz_values[0], exact[0] - 1e-10)
        self.assertLessEqual(result.ritz_values[-1], exact[-1] + 1e-10)

    def test_sparse(self):
        result = lanczos_tridiagonalize(sparse.csr_matrix(self.A), self.b, m=5)
        self.assertEqual(result.tridiagonal.shape, (5, 5))

    def test_operator_and_reorthogonalization(self):
        op = LinearOperator(self.A.shape, matvec=lambda v: self.A @ v)
        result = lanczos_tridiagonalize(op, self.b, m=8, reorthogonalize=True)
        np.testing.assert_allclose(result.basis.T @ result.basis, np.eye(8), atol=1e-10)

    def test_complex_hermitian(self):
        A = np.array([[4, 1+1j], [1-1j, 3]], dtype=complex)
        result = lanczos_tridiagonalize(A, np.array([1+0j, 2+1j]), m=2)
        np.testing.assert_allclose(np.sort(result.ritz_values), np.linalg.eigvalsh(A), atol=1e-10)

    def test_complex_projection(self):
        A = np.array([[4, 1 + 1j], [1 - 1j, 3]], dtype=complex)
        result = lanczos_tridiagonalize(A, np.array([1+0j, 2+1j]), m=2)
        Q = result.basis
        np.testing.assert_allclose(Q.conj().T @ Q, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(Q.conj().T @ A @ Q, result.tridiagonal, atol=1e-10)

    def test_complex_callable_real_start(self):
        A = np.array([[4, 1 + 1j], [1 - 1j, 3]], dtype=complex)
        result = lanczos_tridiagonalize(lambda v: A @ v, np.array([1., 2.]), m=2)
        self.assertTrue(np.iscomplexobj(result.basis))
        np.testing.assert_allclose(result.ritz_values, np.linalg.eigvalsh(A), atol=1e-10)

    def test_reject_nonsymmetric(self):
        A = np.array([[2., 3.], [0., 2.]])
        with self.assertRaises(ValueError):
            lanczos_tridiagonalize(A, np.array([1., 1.]))

    def test_invalid_starting_vector(self):
        v = np.ones(12)
        v[0] = np.nan
        with self.assertRaises(ValueError):
            lanczos_tridiagonalize(self.A, v)

    def test_bad_galerkin_rhs(self):
        with self.assertRaises(ValueError):
            lanczos_galerkin(self.A, np.full(12, np.inf))

    def test_ritz_pair_residual(self):
        result = lanczos_tridiagonalize(self.A, self.b, m=8)
        eigenvalues, Y = np.linalg.eigh(result.tridiagonal)
        U = result.basis @ Y
        actual = np.linalg.norm(self.A @ U - U * eigenvalues, axis=0)
        predicted = result.beta_next * np.abs(Y[-1, :])
        np.testing.assert_allclose(actual, predicted, atol=1e-10)

    def test_full_galerkin_solution(self):
        result = lanczos_galerkin(self.A, self.b, m=12, tol=1e-10)
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.solution, np.linalg.solve(self.A, self.b), atol=1e-9)

    def test_partial_galerkin(self):
        result = lanczos_galerkin(self.A, self.b, m=5, tol=1e-12)
        self.assertFalse(result.converged)
        self.assertEqual(result.lanczos.iterations, 5)

    def test_initial_solution(self):
        x = np.linalg.solve(self.A, self.b)
        result = lanczos_galerkin(self.A, self.b, x0=x)
        self.assertTrue(result.converged)
        self.assertEqual(result.lanczos.iterations, 0)

    def test_zero_start(self):
        with self.assertRaises(ValueError):
            lanczos_tridiagonalize(self.A, np.zeros(len(self.b)))


if __name__ == "__main__":
    unittest.main()
