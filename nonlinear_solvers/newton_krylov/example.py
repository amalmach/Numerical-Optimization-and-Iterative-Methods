"""Run with: python examples.py from the package directory."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newton_krylov import newton_gmres, jfnk, preconditioned_jfnk, inexact_jfnk


def fun(x):
    return np.array([x[0] * np.cos(x[1]) - 4, x[1] * (x[0] - 1) - 5])


def jac(x):
    return np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]])


def inverse_diagonal(x, v):
    diagonal = np.diag(jac(x))
    return v / np.where(abs(diagonal) < 1e-6, 1.0, diagonal)


if __name__ == '__main__':
    x0 = np.array([1., 1.])
    for name, solve in [
        ('Newton--GMRES', lambda: newton_gmres(fun, x0, jacobian=jac)),
        ('JFNK', lambda: jfnk(fun, x0)),
        ('Right-preconditioned JFNK', lambda: preconditioned_jfnk(fun, x0, inverse_preconditioner=inverse_diagonal)),
        ('Inexact JFNK', lambda: inexact_jfnk(fun, x0)),
    ]:
        result = solve()
        print(f'{name:26} converged={result.converged} outer={result.iterations} '
              f'||F||={result.residual_norms[-1]:.3e} inner={sum(result.gmres_iterations)} '
              f'x={result.x} ({result.message})')
