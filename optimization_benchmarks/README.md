# Optimization Benchmarks

This directory contains mathematical test problems for evaluating numerical optimization algorithms.

Optimization benchmarks provide reproducible objective functions with known mathematical properties. They are useful for studying convergence, derivative accuracy, numerical stability, and sensitivity to initial conditions.

## Available Benchmarks

| Benchmark | Description | Documentation |
|---|---|---|
| Rosenbrock-type Function | Multidimensional nonconvex objective with analytical gradient and Hessian. | [Explore](rosenbrock/README.md) |

## Rosenbrock Benchmark

The Rosenbrock-type function is defined by

\[
J_R(x)=
\sum_{i=1}^{n-1}
\left[
(x_{i+1}-x_i^2)^2+(x_i-1)^2
\right],
\qquad n\geq2.
\]

Its known global minimizer is

\[
x^*=(1,\ldots,1)^T,
\]

with

\[
J_R(x^*)=0.
\]

This repository uses the coefficient-1 formulation rather than the conventional coefficient-100 Rosenbrock function.

The benchmark provides analytical objective, gradient, and Hessian evaluations.

## Repository Structure

```text
optimization_benchmarks/
├── __init__.py
├── README.md
├── rosenbrock/
│   ├── __init__.py
│   ├── README.md
│   ├── benchmark.py
│   ├── example.py
│   └── test_rosenbrock.py
└── notebooks/
```

## Quick Start

Import the benchmark:

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

Run the reproducible example:

```bash
python -m optimization_benchmarks.rosenbrock.example
```

Run the automated tests:

```bash
python -m pytest -v optimization_benchmarks/rosenbrock/test_rosenbrock.py
```

## Numerical Validation

The Rosenbrock implementation is tested using known objective values, the global minimizer, Hessian symmetry, and comparisons of analytical derivatives with central finite differences.

The current automated test suite contains **10 passing tests**.

## Relationship to Optimization Algorithms

The Rosenbrock benchmark complements the numerical optimization algorithms implemented elsewhere in this repository.

In particular, its objective and gradient are reused from:

```text
gradient_methods/spectral_gradient_solver/objectives.py
```

This avoids duplicating the same mathematical implementation.

The benchmark itself does not perform optimization. Instead, it supplies a reproducible test problem for optimization algorithms.

## Documentation

For mathematical derivations, numerical experiments, computational complexity, and implementation details, see the [Rosenbrock README](rosenbrock/README.md).
