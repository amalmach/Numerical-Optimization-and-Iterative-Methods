"""
Rosenbrock-type optimization benchmark.

Provides the objective function, analytical gradient,
and analytical Hessian.

The objective and gradient reuse the existing
Spectral Gradient Solver implementation.
"""

from .benchmark import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
)

__all__ = [
    "rosenbrock",
    "rosenbrock_gradient",
    "rosenbrock_hessian",
]
