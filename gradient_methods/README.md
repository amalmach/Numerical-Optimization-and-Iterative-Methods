<div align="center">

# Gradient Methods

Numerical optimization algorithms for minimizing differentiable objective functions.

This directory contains reusable Python solvers, numerical tests, mathematical documentation, and interactive Jupyter/Voilà dashboards.

</div>

---

## Main idea

Gradient methods minimize differentiable objective functions by iteratively moving in the direction of the negative gradient. Consider

<p align="center">$\displaystyle
\min_{x \in \mathbb{R}^n} f(x).
$</p>

Starting from an initial point \(x_0\), the general iteration is

<p align="center">$\displaystyle
x_{k+1}=x_k-\alpha_k\nabla f(x_k),
$</p>

where \(\nabla f(x_k)\) is the gradient and \(\alpha_k>0\) is the step size.

Different gradient methods are distinguished by how they choose \(\alpha_k\):

- **Fixed-Step Gradient Descent:** Uses a constant step size.
- **Steepest Descent:** Determines the step size using a line-search procedure.
- **Spectral Gradient (BB1/BB2):** Estimates the step size from differences between successive iterates and gradients.

Convergence depends on the objective function and the step-size strategy. In particular, spectral gradient methods may exhibit nonmonotonic objective values.

## Implemented methods

| Method | Implementation | Documentation | Interactive Notebook | Voilà Dashboard |
|---|---|---|---|---|
| **Fixed-Step Gradient Descent** | [Solver](fixed_step_gradient_solver/FixedStepGradient.py) | [README](fixed_step_gradient_solver/README.md) | [Notebook](notebooks/fixed_step_gradient_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Ffixed_step_gradient_demo.ipynb) |
| **Steepest Descent** | [Solver](steepest_descent_solver/SteepestDescent.py) | [README](steepest_descent_solver/README.md) | [Notebook](notebooks/steepest_descent_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fsteepest_descent_demo.ipynb) |
| **Gradient Methods Comparison** | Fixed-Step vs Steepest Descent | — | [Notebook](notebooks/gradient_methods_comparison_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fgradient_methods_comparison_demo.ipynb) |
| **Spectral Gradient (BB1/BB2)** | [Solver](spectral_gradient_solver/SpectralGradient.py) | [README](spectral_gradient_solver/README.md) | [Notebook](notebooks/spectral_gradient_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fspectral_gradient_demo.ipynb) |

> Binder may require some time to initialize the environment on the first launch.

## Interactive Dashboards

The Voilà dashboards let you run experiments directly in your browser without installing Python locally.

### Gradient Methods Comparison Dashboard

Compare **Fixed-Step Gradient Descent** and **Steepest Descent with Exact Line Search** on identical quadratic optimization problems.

The interactive dashboard provides:

- **Shared experiment controls:** Select the objective function, problem dimension, fixed step size, convergence tolerance, and iteration budget.
- **Three convergence plots:** Compare objective values, gradient norms, and solution errors.
- **Numerical comparison table:** Examine iteration counts, convergence status, final objective values, gradient norms, and solution errors.
- **Interactive status indicators:** Identify when results are up to date or need to be recalculated.

[![Launch Gradient Methods Comparison Dashboard](https://img.shields.io/badge/LAUNCH-GRADIENT%20COMPARISON-2ea44f?style=for-the-badge)](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fgradient_methods_comparison_demo.ipynb)

![Gradient Methods Comparison Dashboard](../docs/images/gradient_methods_comparison_dashboard.png)

**Notebook:** [gradient_methods_comparison_demo.ipynb](notebooks/gradient_methods_comparison_demo.ipynb)

> **Numerical note:** For the quadratic objectives used in this comparison, the Hessian is \(2I\). Consequently, exact-line-search Steepest Descent reaches the minimizer in one iteration (up to floating-point precision).

## Run the Tests

From the repository root:

```bash
python3 -m unittest gradient_methods.fixed_step_gradient_solver.test_fixed_step_gradient -v
python3 -m unittest gradient_methods.steepest_descent_solver.test_steepest_descent -v
python3 -m unittest gradient_methods.spectral_gradient_solver.test_spectral_gradient -v
```

## References

1. J. Nocedal and S. J. Wright, *Numerical Optimization*, Springer, 1999.

2. S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.

3. J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, 1994.
   
---

<div align="center">

[⬅ Back to Main Repository](../README.md)

</div>
