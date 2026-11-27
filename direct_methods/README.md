<div align="center">

# Direct Methods

This directory contains implementations of **direct numerical methods** for solving linear algebra problems.

Direct methods obtain a solution through a finite sequence of algebraic operations, in contrast to iterative methods that progressively approximate a solution.

</div>

---

## Available Methods

| Method | Description | Implementation |
|---|---|---|
| LU Factorization with Partial Pivoting | Factorizes a dense matrix as $PA=LU$ and solves linear systems using triangular substitution. | [LU Factorization](lu_factorization/) |

## Quick Start

From the repository root:

```bash
python -m direct_methods.lu_factorization.example
```

Run the automated tests:

```bash
python -m pytest -v direct_methods/lu_factorization/test_lu.py
```

## Relationship to Iterative Methods

Direct methods complement the iterative methods implemented elsewhere in this repository.

For dense systems of moderate size, LU factorization provides a standard solution technique.

For large sparse systems, iterative Krylov methods can be preferable depending on the matrix properties and available preconditioners.

LU factorization is also useful when constructing preconditioners or solving linear systems inside Newton-type algorithms.

