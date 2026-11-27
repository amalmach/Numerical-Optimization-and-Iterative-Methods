<div align="center">

# Global Optimization Methods

**Numerical algorithms for optimizing nonlinear, multimodal,
and computationally expensive objective functions.**

</div>

---

## Overview

Global optimization aims to identify the best solution over a
prescribed search domain, even when the objective function contains
multiple local minima.

Unlike local optimization methods, global optimization algorithms
seek to explore the search space while progressively refining
promising regions.

This category focuses on surrogate-based optimization and
sequential experimental design.

---

## Available Methods

| Method | Description | Implementation |
|:---|:---|:---|
| **Efficient Global Optimization (EGO)** | Gaussian process-based optimization using Expected Improvement, with fixed or adaptively estimated kernel hyperparameters. | [EGO](ego/) |

---

## Interactive Notebooks

| Method | Notebook |
|:---|:---|
| Efficient Global Optimization | [EGO Interactive Dashboard](notebooks/ego_demo.ipynb) |

The notebook provides an interactive comparison of Fixed EGO
and Adaptive EGO.

---

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

## Run the Examples

```bash
python -m global_optimization.ego.example
```

## Run the Tests

```bash
python -m pytest -v global_optimization/ego/test_ego.py
```

---

<div align="center">

[⬅ Back to Main Repository](../README.md)

</div>
