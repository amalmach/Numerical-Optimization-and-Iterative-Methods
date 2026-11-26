<div align="center">

# Interactive Gradient Methods Notebooks

These notebooks explore the project's **modular gradient-based optimization solvers**, visualize convergence behavior, and compare different step-size strategies through interactive Jupyter/Voilà dashboards.

</div>

---

## Interactive notebooks

| Experiment | GitHub notebook | Launch on Binder |
|:---|:---|:---|
| Fixed-Step Gradient Descent | [Fixed-Step](fixed_step_gradient_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Ffixed_step_gradient_demo.ipynb) |
| Steepest Descent | [Steepest Descent](steepest_descent_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fsteepest_descent_demo.ipynb) |
| Spectral Gradient (BB1/BB2) | [Spectral Gradient](spectral_gradient_demo.ipynb) | [Launch Dashboard](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fspectral_gradient_demo.ipynb) |
| **Cross-method comparison** | [Comparison](gradient_methods_comparison_demo.ipynb) | [Launch Comparison](https://mybinder.org/v2/gh/amalmach/Numerical-Optimization-and-Iterative-Methods/main?urlpath=voila%2Frender%2Fgradient_methods%2Fnotebooks%2Fgradient_methods_comparison_demo.ipynb) |

## Cross-method comparison

[`gradient_methods_comparison_demo.ipynb`](gradient_methods_comparison_demo.ipynb) uses the **modular gradient solver packages**:

- `gradient_methods.fixed_step_gradient_solver`: Fixed-Step Gradient Descent.
- `gradient_methods.steepest_descent_solver`: Steepest Descent with Exact Line Search.
- `gradient_methods.spectral_gradient_solver`: Barzilai–Borwein spectral gradient methods (BB1 and BB2).

The comparison dashboard evaluates all four methods on identical quadratic optimization problems, using common initial points, convergence tolerances, and iteration budgets.

It includes:

1. **Objective-value convergence:** Compare the evolution of the objective function.
2. **Gradient-norm convergence:** Monitor the decrease in gradient magnitude.
3. **Solution error:** Measure the distance to the known minimizer.
4. **Numerical results:** Compare iteration counts, convergence status, final objective values, gradient norms, and solution errors.

The dashboard also provides interactive parameter controls and a status indicator reminding users to rerun experiments when parameters change.

For the quadratic objectives used in the comparison, the Hessian is \(2I\). Consequently, Steepest Descent with Exact Line Search reaches the minimizer in one iteration, up to floating-point precision.

## Individual method dashboards

### Fixed-Step Gradient Descent

[`fixed_step_gradient_demo.ipynb`](fixed_step_gradient_demo.ipynb) illustrates gradient descent with a constant step size.

The dashboard allows users to investigate how the learning rate, problem dimension, convergence tolerance, and iteration budget affect convergence.

**Solver documentation:** [Fixed-Step Gradient Descent](../fixed_step_gradient_solver/README.md)

### Steepest Descent

[`steepest_descent_demo.ipynb`](steepest_descent_demo.ipynb) demonstrates Steepest Descent with Exact Line Search.

It illustrates the influence of the objective function and stopping criteria on convergence.

**Solver documentation:** [Steepest Descent](../steepest_descent_solver/README.md)

### Spectral Gradient — BB1 and BB2

[`spectral_gradient_demo.ipynb`](spectral_gradient_demo.ipynb) compares the two Barzilai–Borwein spectral gradient variants.

The dashboard supports five objective functions:

- **J1 and J2:** Quadratic objectives.
- **J5:** Tridiagonal quadratic objective.
- **J6:** Quartic-regularized objective.
- **JR:** Rosenbrock-type objective.

Users can adjust the problem dimension, initial step size, convergence tolerance, and maximum iteration count.

The dashboard displays objective-value histories, gradient norms, spectral step sizes, and numerical comparison results.

**Solver documentation:** [Spectral Gradient Methods](../spectral_gradient_solver/README.md)

## Shared setup and execution

The notebooks use the numerical solvers implemented in the `gradient_methods/` directory.

Launch JupyterLab from the **repository root**:

```bash
python3 -m pip install numpy scipy matplotlib ipywidgets ipykernel jupyterlab voila
jupyter lab
```

Alternatively, launch the comparison dashboard locally:

```bash
voila gradient_methods/notebooks/gradient_methods_comparison_demo.ipynb
```

The notebooks locate the repository root automatically and import the corresponding solver packages.

For online execution, the Binder links use the dependencies defined in `.binder/requirements.txt`.

### Interactive execution

Open a notebook or its Voilà dashboard, configure the experiment parameters, and click **Run Comparison** or the corresponding execution button.

Binder is a public, resource-limited service. Start with moderate problem dimensions and iteration budgets.

## Interpretation notes

- Gradient methods use updates of the form $x_{k+1}=x_k-\alpha_k\nabla J(x_k)$, where the methods differ in their choice of step size.
- **Fixed-Step Gradient Descent** uses a constant step size, making convergence sensitive to its selection.
- **Steepest Descent** determines the step length through exact line search for the supported quadratic problems.
- **BB1 and BB2** use differences between successive iterates and gradients to estimate spectral step sizes.
- Spectral gradient methods are not necessarily monotonic: the objective value may increase at some iterations.
- A small gradient norm indicates approximate stationarity but does not, by itself, guarantee proximity to a global minimizer for arbitrary nonconvex problems.
- Iteration counts depend on the objective function, initialization, step-size strategy, tolerance, and problem dimension.

For mathematical formulations, implementation details, and numerical safeguards, consult the individual solver READMEs.

---

<div align="center">

[⬅ Back to Gradient Methods](../README.md)

</div>
