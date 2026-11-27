"""
Newton–Krylov methods for solving nonlinear systems.

Available methods
-----------------
- Newton–GMRES
- Jacobian-Free Newton–Krylov (JFNK)
- Right-preconditioned JFNK
- Inexact Newton–Krylov
"""

from .newton_krylov import (
    NewtonKrylovResult,
    newton_krylov,
    newton_gmres,
    jfnk,
    preconditioned_jfnk,
    inexact_jfnk,
)

__all__ = [
    "NewtonKrylovResult",
    "newton_krylov",
    "newton_gmres",
    "jfnk",
    "preconditioned_jfnk",
    "inexact_jfnk",
]
