# Modular GMRES Solver

An educational implementation based on Amal Machtalay's GMRES codes.

| Module | Responsibility |
|---|---|
| `arnoldi.py` | Modified Gram-Schmidt and optional reorthogonalization |
| `hessenberg.py` | Hessenberg workspace and projection helper |
| `givens.py` | Stable Givens rotations and right-hand-side updates |
| `least_squares.py` | Reduced triangular solve with rank-deficient fallback |
| `preconditioners.py` | Identity, Jacobi, exact LU, ILU, IC(0), SSOR, custom |
| `GMRES.py` | Solver options, convergence, restarting, result metadata |
| `example.py` | Executable usage examples |
| `test_gmres.py` | Regression tests against SciPy |

## Run

Install dependencies: `python -m pip install numpy scipy`.

From the **directory containing** the `gmres_solver` folder:

```bash
python -m gmres_solver.example
python -m unittest gmres_solver.test_gmres -v
```

## Usage

```python
import numpy as np
from gmres_solver.GMRES import gmres

A = np.array([[4., 1.], [2., 3.]])
b = np.array([1., 2.])

plain = gmres(A, b, restart=None, maxiter=2)
restarted = gmres(A, b, restart=2, maxiter=20)
preconditioned = gmres(A, b, restart=2, maxiter=20,
                       preconditioner="jacobi")
print(preconditioned.x, preconditioned.converged)
```

`restart=None` means **unrestarted**. Otherwise `restart` is the maximum
**inner Krylov dimension per cycle**, not the number of cycles. `maxiter`
limits **total** Arnoldi iterations. Convergence is checked using
`||b-Ax|| <= max(atol, rtol*||b||)` (the **true** residual).

Use `preconditioner="ilu"` with options such as
`preconditioner_options={"drop_tol": 1e-5, "fill_factor": 15}`.
For a custom inverse action, use `preconditioner="custom", M=my_solve`.
An explicit custom matrix `M` is interpreted as the preconditioner matrix
to factor and solve, **not** as the inverse matrix.

`GMRESResult` provides `x`, `converged`, `iterations`, `cycles`,
`residual_norm`, `residual_history`, `preconditioned_history`, and `message`.

### Notes

- Left preconditioning minimizes the **preconditioned** residual within a
  cycle, but the stopping criterion checks the **true** residual.
- `ic0` is a simple, educational real-SPD IC(0) implementation with dense
  intermediate storage; do not use it on large sparse systems.
- Exact LU is a **direct solve used as a preconditioner** and is primarily
  useful for validation, not for improving GMRES performance.
- The code is intended for research/teaching and has not been optimized for
  large production problems or high-performance parallel computing.
- Existing `my_gmres_solver` and `Restart_GMRES_solver` aliases are provided
  as compatibility functions in `GMRES.py`, but new code should use `gmres`.
