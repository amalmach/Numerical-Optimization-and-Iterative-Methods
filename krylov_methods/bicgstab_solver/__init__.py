"""Modular BiCGSTAB solver, with optional right preconditioning."""

from .BiCGSTAB import bicgstab, bicgstab_solver, solve_bicgstab, BiCGSTABResult

__all__ = ["bicgstab", "bicgstab_solver", "solve_bicgstab", "BiCGSTABResult"]
