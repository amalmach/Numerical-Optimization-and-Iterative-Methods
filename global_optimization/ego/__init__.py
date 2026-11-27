"""
Efficient Global Optimization (EGO).

Gaussian process-based global optimization using Expected Improvement,
with optional maximum-likelihood hyperparameter estimation.
"""

from .acquisition import expected_improvement
from .benchmarks import modified_branin, modified_branin_normalized
from .ego import EGOResult, efficient_global_optimization
from .gaussian_process import GaussianProcess, gaussian_kernel
from .hyperparameters import (
    estimate_hyperparameters,
    negative_log_likelihood,
)

__all__ = [
    "EGOResult",
    "GaussianProcess",
    "efficient_global_optimization",
    "expected_improvement",
    "gaussian_kernel",
    "estimate_hyperparameters",
    "negative_log_likelihood",
    "modified_branin",
    "modified_branin_normalized",
]
