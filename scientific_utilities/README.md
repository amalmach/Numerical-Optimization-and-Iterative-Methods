# Scientific Utilities

A collection of reusable numerical tools for scientific computing, numerical linear algebra, and mathematical experimentation.

This directory modernizes elementary operations from the repository's original Python scripts into documented, tested, and reusable modules.

## Overview

Scientific Utilities provides foundational numerical operations used throughout scientific computing.

The primary goals are:

- Demonstrate mathematical operations through readable Python implementations.
- Provide reusable functions with clearly defined inputs and outputs.
- Validate numerical results against NumPy.
- Support reproducible scientific experiments.
- Preserve the repository's original educational examples.

## Available Modules

| Module | Description | Documentation |
|---|---|---|
| **Numerical Operations** | Vector norms, matrix products, dot products, outer products, transposition, array conversion, and nearest-value search | [Explore](numerical_operations/README.md) |

## Numerical Operations

The `numerical_operations` package provides eight functions:

| Function | Description |
|---|---|
| `closest()` | Find the nearest numerical value |
| `norm_two()` | Compute the Euclidean vector norm |
| `AxB()` | Multiply two matrices |
| `matvec()` | Multiply a matrix by a vector |
| `scalar()` | Compute a dot product |
| `vecvec()` | Compute an outer product |
| `matT()` | Transpose a matrix |
| `vec_asarray()` | Convert nested vectors into a NumPy array |

These functions emphasize mathematical clarity and input validation.

For mathematical definitions, implementation details, and complexity analysis, see the [Numerical Operations documentation](numerical_operations/README.md).

## Quick Start

From the repository root:

```python
import numpy as np

from scientific_utilities.numerical_operations import (
    closest,
    norm_two,
    AxB,
    matvec,
)

A = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

v = np.array([1.0, 2.0])

print("Nearest value:", closest([0.1, 0.4, 0.8], 0.7))
print("Euclidean norm:", norm_two(v))
print("Matrix-vector product:", matvec(A, v))
print("Matrix product:", AxB(A, A))
```

## Reproducible Example

Run:

```bash
python -m scientific_utilities.numerical_operations.example
```

The example demonstrates all eight numerical operations and compares their results against NumPy.

## Automated Tests

Run:

```bash
python -m pytest -q scientific_utilities/numerical_operations/test_operations.py
```

The test suite includes **16 automated tests** covering numerical correctness, rectangular matrices, dimension compatibility, input validation, and nearest-value tie-breaking.

## Project Structure

```text
scientific_utilities/
├── __init__.py
├── README.md
├── numerical_operations/
│   ├── __init__.py
│   ├── README.md
│   ├── operations.py
│   ├── example.py
│   └── test_operations.py
└── notebooks/
```

## Design Principles

**Mathematical transparency:** Explicit loops are used where they help demonstrate the underlying mathematics.

**Numerical verification:** Results are compared against NumPy reference implementations.

**Reusable interfaces:** Functions are available through the `numerical_operations` package.

**Input validation:** Invalid shapes, incompatible dimensions, and nonfinite numerical inputs are rejected.

**Reproducibility:** Examples and tests can be executed directly from the repository root.

## Relationship to Other Modules

Scientific Utilities focuses on foundational numerical operations rather than complete numerical algorithms.

More advanced methods are organized separately:

- [Direct Methods](../direct_methods/) — LU factorization and direct linear-system solvers.
- [Gradient Methods](../gradient_methods/) — Gradient-based optimization algorithms.
- [Krylov Methods](../krylov_methods/) — Iterative methods for linear systems.
- [Optimization Benchmarks](../optimization_benchmarks/) — Mathematical benchmark functions.

LU factorization is intentionally not duplicated in Scientific Utilities.

## Legacy Sources

This package modernizes operations demonstrated in:

- [`closest.py`](../closest.py)
- [`numpy_scipy_func.py`](../numpy_scipy_func.py)

Both original scripts are preserved.

## Dependencies

The numerical operations package requires:

- Python 3.10 or later
- NumPy
- pytest (for automated testing)

For performance-critical scientific computing, prefer NumPy's optimized implementations.
