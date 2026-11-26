"""Barzilai-Borwein spectral gradient optimization methods."""

from .SpectralGradient import (
    SpectralGradientResult,
    solve_spectral_gradient,
    BB1,
    BB2,
)

from .objectives import (
    J1,
    J2,
    J5,
    J6,
    JR,
    cost,
    tridiag,
)

__all__ = [
    "SpectralGradientResult",
    "solve_spectral_gradient",
    "BB1",
    "BB2",
    "J1",
    "J2",
    "J5",
    "J6",
    "JR",
    "cost",
    "tridiag",
]
