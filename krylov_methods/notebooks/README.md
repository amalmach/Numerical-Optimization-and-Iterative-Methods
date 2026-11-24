<div align="center">
  
# Interactive Krylov Solver Notebooks

These notebooks examine the project's **modular iterative solvers**, compare
them against SciPy where appropriate, and visualize convergence, conditioning,
preconditioning, and spectral approximations.

</div>

---

## Interactive notebooks

| Experiment | GitHub notebook | Launch on Binder |
|:---|:---|:---|
| Conjugate Gradient (CG/PCG) | [CG](conjugate_gradient_demo.ipynb) | [Launch CG](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/HEAD?labpath=krylov_methods%2Fnotebooks%2Fconjugate_gradient_demo.ipynb) |
| GMRES (restarted / unrestarted) | [GMRES](gmres_demo.ipynb) | [Launch GMRES](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/HEAD?labpath=krylov_methods%2Fnotebooks%2Fgmres_demo.ipynb) |
| BiCGSTAB | [BiCGSTAB](bicgstab_demo.ipynb) | [Launch BiCGSTAB](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/HEAD?labpath=krylov_methods%2Fnotebooks%2Fbicgstab_demo.ipynb) |
| Symmetric Lanczos | [Lanczos](symmetric_lanczos_demo.ipynb) | [Launch Lanczos](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/HEAD?labpath=krylov_methods%2Fnotebooks%2Fsymmetric_lanczos_demo.ipynb) |
| **Cross-method comparison** | [Comparison](krylov_comparison.ipynb) | [Launch comparison](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/HEAD?labpath=krylov_methods%2Fnotebooks%2Fkrylov_comparison.ipynb) |

## Cross-method comparison

[`krylov_comparison.ipynb`](krylov_comparison.ipynb) uses the **new modular
solver packages**:

- `gradient_methods.steepest_descent_solver`: Steepest Descent on SPD / HPD systems.
- `krylov_methods.conjugate_gradient_solver`: CG on SPD / HPD systems.
- `krylov_methods.gmres_solver`: restarted and unrestarted GMRES.
- `krylov_methods.bicgstab_solver`: BiCGSTAB for nonsymmetric systems.
- `krylov_methods.symmetric_lanczos_solver`: tridiagonalization and optional
  Galerkin linear-system approximation.

Steepest Descent is a **gradient-based baseline** and lives in
[`gradient_methods/`](../../gradient_methods/).

The comparison notebook includes three interactive experiments:

1. **SPD / complex HPD systems:** compare Steepest Descent, CG, GMRES, and
   BiCGSTAB using known exact solutions and true residuals.
2. **Nonsymmetric systems:** compare restarted/unrestarted GMRES with BiCGSTAB,
   including iteration counts, runtimes, residuals, and solution errors.
3. **Symmetric Lanczos:** inspect orthogonality, projection error, Ritz values,
   Ritz-pair residuals, and the Galerkin approximation to $Ax=b$.

The methods are intentionally only applied to compatible matrices: CG and
Steepest Descent require positive definiteness, whereas GMRES and BiCGSTAB
also accommodate nonsymmetric systems. Lanczos is primarily a basis / spectral
procedure, not a general-purpose iterative solver by itself.

## Shared utilities and setup

The notebooks share [`notebook_utils.py`](notebook_utils.py), including
reproducible system generators, plotting helpers, and fixed preconditioner
examples.

Launch from the **repository root**:

```bash
python3 -m pip install numpy scipy matplotlib ipywidgets ipykernel jupyterlab
jupyter lab
```

The notebooks locate the repository root and import sibling packages from
there. To run on Binder, commit **both** `gradient_methods/` and
`krylov_methods/` plus the notebooks and the Binder dependencies (for example,
`.binder/requirements.txt`) to the default branch.

Run each notebook's setup cell, select an experiment, and click its **Run**
button. Binder is a limited-resource public service: start with small matrices,
moderate Krylov dimensions, and reasonable iteration budgets.

## Interpretation notes

- The stopping rule is normally
  $\|b-Ax\|_2\leq\max(\mathrm{atol},\mathrm{rtol}\|b\|_2)$.
- **True residual** $\|b-Ax_k\|_2$ differs from a left-preconditioned
  residual or a QR/Givens estimate. Compare like with like.
- A small residual does not necessarily imply a small **solution error** on
  an ill-conditioned system.
- GMRES true-residual callbacks add computation, so notebook timings are
  illustrative, not a controlled benchmark.
- Compare individual preconditioners in the CG, GMRES, and BiCGSTAB notebooks.
  For formulas and implementation restrictions, see the shared
  [Preconditioning Methods](../preconditioners/README.md) documentation.


---

<div align="center">

[⬅ Back to Krylov Methods](../README.md)

</div>
