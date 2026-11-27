"""
Acceptance-rejection sampling methods.

Provides one- and two-dimensional rejection sampling algorithms
for bounded, nonnegative target densities.
"""

from .rejection_sampling import sample_1d, sample_2d

__all__ = [
    "sample_1d",
    "sample_2d",
]
