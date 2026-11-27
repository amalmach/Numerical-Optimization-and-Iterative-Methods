import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from newton_krylov import newton_gmres, jfnk, preconditioned_jfnk, inexact_jfnk


def f(x):
    return np.array([x[0]*np.cos(x[1])-4, x[1]*(x[0]-1)-5])


def jac(x):
    return np.array([[np.cos(x[1]),-x[0]*np.sin(x[1])],[x[1],x[0]-1]])


@pytest.mark.parametrize('method', ['explicit','jacobian_free','preconditioned','inexact'])
def test_2d_problem(method):
    x0 = np.array([1.,1.])
    if method == 'explicit':
        out = newton_gmres(f, x0, jacobian=jac)
    elif method == 'jacobian_free':
        out = jfnk(f, x0)
    elif method == 'preconditioned':
        out = preconditioned_jfnk(f, x0, inverse_preconditioner=lambda x, v: v / np.where(np.abs(np.diag(jac(x))) < 1e-6, 1.0, np.diag(jac(x))))
    else:
        out = inexact_jfnk(f, x0)
    assert out.converged, out.message
    assert np.linalg.norm(f(out.x)) < 1e-10
    np.testing.assert_allclose(out.x, [6.50409711, 0.90841421], atol=1e-7)


def test_medium_size_linear_system_preconditioning():
    n = 40
    a = np.diag(np.linspace(1, 20, n)) + 0.03 * np.eye(n, k=1)
    target = np.ones(n)
    b = a @ target
    out = preconditioned_jfnk(lambda x: a @ x - b, np.zeros(n),
                              inverse_preconditioner=lambda x, v: v / np.diag(a),
                              tol=1e-9)
    assert out.converged, out.message
    np.testing.assert_allclose(out.x, target, atol=1e-8)


def test_initial_solution_and_validation():
    x = np.array([6.50409711, 0.90841421])
    assert jfnk(f, x, tol=1e-7).iterations == 0
    with pytest.raises(ValueError):
        jfnk(f, [1,1], tol=-1)
