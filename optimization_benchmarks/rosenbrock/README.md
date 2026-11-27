# Rosenbrock Optimization Benchmark

A reusable implementation of a multidimensional Rosenbrock-type test function, including its analytical gradient and Hessian.

This benchmark is intended for studying numerical optimization algorithms, derivative accuracy, and the geometry of nonlinear objective functions.

The implementation reuses the Rosenbrock objective and gradient already available in the Spectral Gradient Solver.

## 1. Mathematical Formulation

For a vector \(x\in\mathbb{R}^n\), with \(n\geq2\), consider

\[
J_R(x)=
\sum_{i=1}^{n-1}
\left[
(x_{i+1}-x_i^2)^2+(x_i-1)^2
\right].
\]

The function is nonnegative:

\[
J_R(x)\geq0.
\]

Its global minimum is attained at

\[
\boxed{x^*=(1,1,\ldots,1)^T},
\]

with

\[
\boxed{J_R(x^*)=0}.
\]

### Relationship to the Conventional Rosenbrock Function

The conventional Rosenbrock benchmark is often written as

\[
f(x)=
\sum_{i=1}^{n-1}
\left[
100(x_{i+1}-x_i^2)^2+(1-x_i)^2
\right].
\]

Our implementation uses a coefficient of **1**, rather than **100**, for the first squared term.

This preserves the global minimizer but changes the function's curvature and conditioning.

Results obtained with these two formulations should therefore not be compared without accounting for that difference.

## 2. Two-Dimensional Formulation

For \(x=(x_1,x_2)^T\),

\[
J_R(x_1,x_2)
=
(x_2-x_1^2)^2+(x_1-1)^2.
\]

The analytical gradient is

\[
\nabla J_R(x_1,x_2)=
\begin{pmatrix}
2(x_1-1)-4x_1(x_2-x_1^2)\\
2(x_2-x_1^2)
\end{pmatrix}.
\]

The analytical Hessian is

\[
\nabla^2 J_R(x_1,x_2)=
\begin{pmatrix}
12x_1^2-4x_2+2 & -4x_1\\
-4x_1 & 2
\end{pmatrix}.
\]

At the global minimizer,

\[
\nabla J_R(1,1)=
\begin{pmatrix}
0\\
0
\end{pmatrix},
\]

and

\[
\nabla^2 J_R(1,1)=
\begin{pmatrix}
10 & -4\\
-4 & 2
\end{pmatrix}.
\]

The Hessian is positive definite at the minimizer.

However, the Rosenbrock function is not globally convex.

## 3. Multidimensional Derivatives

Define

\[
w_i=x_{i+1}-x_i^2,
\qquad i=1,\ldots,n-1.
\]

The analytical gradient is assembled from neighboring terms:

\[
\nabla J_R(x)
=
\nabla
\sum_{i=1}^{n-1}
\left[w_i^2+(x_i-1)^2\right].
\]

For \(1<i<n\),

\[
\frac{\partial J_R}{\partial x_i}
=
2w_{i-1}
-4x_iw_i
+2(x_i-1).
\]

The boundary components are

\[
\frac{\partial J_R}{\partial x_1}
=
-4x_1w_1+2(x_1-1),
\]

and

\[
\frac{\partial J_R}{\partial x_n}
=
2w_{n-1}.
\]

The Hessian is symmetric and tridiagonal.

For \(i=1,\ldots,n-1\), each term contributes

\[
H_{ii}
\mathrel{+}=
12x_i^2-4x_{i+1}+2,
\]

\[
H_{i+1,i+1}
\mathrel{+}=2,
\]

and

\[
H_{i,i+1}
=
H_{i+1,i}
=
-4x_i.
\]

These expressions are implemented analytically rather than approximated by finite differences.

## 4. Python API

The package exposes three functions:

| Function | Description |
|---|---|
| `rosenbrock(x)` | Evaluate the objective |
| `rosenbrock_gradient(x)` | Evaluate the analytical gradient |
| `rosenbrock_hessian(x)` | Evaluate the analytical Hessian |

### Example

```python
import numpy as np

from optimization_benchmarks.rosenbrock import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
)

x = np.array([-1.2, 1.0])

print("Objective:", rosenbrock(x))
print("Gradient:", rosenbrock_gradient(x))
print("Hessian:")
print(rosenbrock_hessian(x))
```

The objective and gradient are reused from:

```python
gradient_methods.spectral_gradient_solver.objectives.JR
```

This avoids maintaining duplicate implementations of the same numerical function.

## 5. Numerical Validation

The implementation is validated at the global minimizer and at nontrivial multidimensional points.

For example, at

\[
x_0=(-1.2,1)^T,
\]

the objective is

\[
J_R(x_0)=5.0336.
\]

The analytical gradient is

\[
\nabla J_R(x_0)=
\begin{pmatrix}
-6.512\\
-0.88
\end{pmatrix}.
\]

The Hessian is

\[
\nabla^2 J_R(x_0)=
\begin{pmatrix}
15.28 & 4.8\\
4.8 & 2
\end{pmatrix}.
\]

### Finite-Difference Verification

For the gradient, central differences approximate

\[
\frac{\partial J_R}{\partial x_i}
\approx
\frac{J_R(x+he_i)-J_R(x-he_i)}{2h}.
\]

For the Hessian,

\[
H_{:,i}(x)
\approx
\frac{\nabla J_R(x+he_i)-\nabla J_R(x-he_i)}{2h}.
\]

At the four-dimensional point

\[
x=(-1.2,1,0.5,1.5)^T,
\]

with \(h=10^{-5}\), the numerical derivative checks produced:

| Metric | Error |
|---|---:|
| Gradient error | \(7.093\times10^{-10}\) |
| Hessian error | \(7.827\times10^{-10}\) |

These results support the correctness of the analytical derivative implementations.

## 6. Computational Complexity

For an input vector of dimension \(n\):

| Operation | Complexity |
|---|---|
| Objective evaluation | \(O(n)\) |
| Gradient evaluation | \(O(n)\) |
| Dense Hessian construction | \(O(n^2)\) |
| Hessian storage | \(O(n^2)\) |

Although the Hessian has a tridiagonal structure, the current implementation returns a dense NumPy array.

A sparse representation could reduce storage requirements for large-dimensional problems.

## 7. Automated Tests

Run the tests from the repository root:

```bash
python -m pytest -v optimization_benchmarks/rosenbrock/test_rosenbrock.py
```

The current test suite contains ten tests covering objective values, the known minimizer, gradient correctness, Hessian symmetry, multidimensional behavior, and finite-difference verification.

## 8. Reproducible Example

Run:

```bash
python -m optimization_benchmarks.rosenbrock.example
```

The example evaluates the objective, gradient, Hessian, and Hessian eigenvalues at an initial point and at the known global minimizer.

## 9. Visualization

The original `RosenbrockFunction.ipynb` contains contour and three-dimensional surface visualizations.

The benchmark package will also include a modern interactive notebook for exploring the function's geometry, including its curved valley and global minimizer.

## 10. Applications

The Rosenbrock function is widely used for testing and comparing numerical optimization methods.

It is useful for investigating gradient-based algorithms, Newton-type methods, line-search strategies, convergence behavior, and sensitivity to initial conditions.

In this repository, the benchmark complements the existing Spectral Gradient Solver.

## 11. Limitations

The implementation:

- Uses the coefficient-1 Rosenbrock formulation.
- Supports real-valued vectors with dimension at least two.
- Returns the Hessian as a dense matrix.
- Does not itself perform optimization.
- Depends on the existing Spectral Gradient objective implementation.

The benchmark is intended for reproducible numerical experiments and educational use.
