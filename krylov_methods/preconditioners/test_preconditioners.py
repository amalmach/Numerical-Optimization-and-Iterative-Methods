"""Run from repo root: python -m unittest krylov_methods.preconditioners.test_preconditioners -v."""

import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from . import make_preconditioner


class PreconditionerTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
        self.r = np.array([2., -1., 3.])

    def test_identity(self):
        np.testing.assert_allclose(make_preconditioner(self.A)(self.r), self.r)

    def test_jacobi(self):
        action = make_preconditioner(self.A, "jacobi", require_spd=True)
        np.testing.assert_allclose(action(self.r), self.r / np.diag(self.A))

    def test_exact_lu(self):
        for matrix in (self.A, sparse.csr_matrix(self.A)):
            action = make_preconditioner(matrix, "lu")
            np.testing.assert_allclose(matrix @ action(self.r), self.r, atol=1e-12)

    def test_exact_cholesky_complex(self):
        A = np.array([[4., 1j], [-1j, 3.]])
        r = np.array([1.+2j, 3.-1j])
        action = make_preconditioner(A, "cholesky", require_spd=True)
        np.testing.assert_allclose(A @ action(r), r, atol=1e-12)

    def test_ilu(self):
        action = make_preconditioner(sparse.csr_matrix(self.A), "ilu", options={"drop_tol": 0.0})
        np.testing.assert_allclose(self.A @ action(self.r), self.r, atol=1e-12)

    def test_ic0(self):
        action = make_preconditioner(sparse.csr_matrix(self.A), "ic0", require_spd=True)
        np.testing.assert_allclose(self.A @ action(self.r), self.r, atol=1e-12)

    def test_ssor_spd_action(self):
        A = np.array([[4., 1j], [-1j, 3.]])
        action = make_preconditioner(A, "ssor", require_spd=True)
        r = np.array([1.+1j, -2.+1j])
        positive = np.vdot(r, action(r))
        self.assertAlmostEqual(positive.imag, 0., places=11)
        self.assertGreater(positive.real, 0.)

    def test_custom_callable_and_operator(self):
        inverse = lambda x: x / np.diag(self.A)
        operator = LinearOperator(self.A.shape, matvec=inverse)
        for M in (inverse, operator):
            action = make_preconditioner(self.A, "custom", M=M, require_spd=True)
            np.testing.assert_allclose(action(self.r), inverse(self.r))

    def test_custom_explicit_matrix(self):
        action = make_preconditioner(self.A, "custom", M=np.diag(np.diag(self.A)), require_spd=True)
        np.testing.assert_allclose(action(self.r), self.r / np.diag(self.A))

    def test_reject_nonsymmetric_pcg_ilu(self):
        with self.assertRaises(ValueError):
            make_preconditioner(self.A, "ilu", require_spd=True)

    def test_reject_bad_options(self):
        with self.assertRaises(ValueError):
            make_preconditioner(self.A, "jacobi", options={"bad": 1})


if __name__ == "__main__":
    unittest.main()
