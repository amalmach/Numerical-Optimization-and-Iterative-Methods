"""Regression tests against direct solves and SciPy CG."""

import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .ConjugateGradient import solve_conjugate_gradient


class TestConjugateGradient(unittest.TestCase):
    def setUp(self):
        self.A = np.diag([4., 5., 6., 7.]) + np.diag([1., 1., 1.], 1) + np.diag([1., 1., 1.], -1)
        self.b = np.array([1., 2., -1., 3.])

    def check(self, A, **kwargs):
        result = solve_conjugate_gradient(A, self.b, rtol=1e-11, atol=1e-13, **kwargs)
        self.assertTrue(result.converged, result.message)
        self.assertLess(np.linalg.norm(self.b - self.A @ result.solution), 1e-10)
        return result

    def test_dense(self):
        result = self.check(self.A, store_history=True)
        self.assertEqual(result.history.shape[0], len(result.residual_norms))

    def test_sparse(self):
        self.check(sparse.csr_matrix(self.A))

    def test_linear_operator(self):
        self.check(LinearOperator(self.A.shape, matvec=lambda x: self.A @ x))

    def test_callable(self):
        self.check(lambda x: self.A @ x)

    def test_jacobi(self):
        self.check(self.A, preconditioner="jacobi")

    def test_ssor(self):
        self.check(self.A, preconditioner="ssor", preconditioner_options={"omega": 1.1})

    def test_custom(self):
        d = np.diag(self.A)
        self.check(self.A, preconditioner="custom", M=lambda r: r / d)

    def test_complex_hermitian(self):
        A = np.array([[5., 1.+2.j], [1.-2.j, 6.]])
        b = np.array([1.+1.j, -2.j])
        result = solve_conjugate_gradient(A, b, rtol=1e-12)
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.solution, np.linalg.solve(A, b), atol=1e-11)

    def test_compare_scipy(self):
        ours = self.check(self.A)
        reference, info = cg(self.A, self.b, rtol=1e-11, atol=1e-13)
        self.assertEqual(info, 0)
        np.testing.assert_allclose(ours.solution, reference, atol=1e-11)

    def test_zero_budget(self):
        result = solve_conjugate_gradient(self.A, self.b, max_iter=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_initial_convergence(self):
        x = np.linalg.solve(self.A, self.b)
        result = solve_conjugate_gradient(self.A, self.b, x0=x)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()
