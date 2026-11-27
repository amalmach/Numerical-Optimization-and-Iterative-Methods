<div align="center">

# Nonlinear Solvers

Python implementations of iterative methods for solving nonlinear systems of equations.

</div>

---

## Available Methods

| Method | Description | Documentation |
|---|---|---|
| **Newton–Krylov Methods** | Newton–GMRES, JFNK, right-preconditioned JFNK, and inexact JFNK | [README](newton_krylov/README.md) |

## Interactive Dashboards

Explore the convergence behavior of nonlinear solvers through interactive experiments.

| Dashboard | Notebook |
|---|---|
| Newton–Krylov | [Open notebook](notebooks/newton_krylov_demo.ipynb) |

## Installation

Install shared dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Running Tests

From the repository root:

```bash
python -m pytest -q nonlinear_solvers/
```

---

<div align="center">

[⬅ Back to Main Repository](../README.md)

</div>
