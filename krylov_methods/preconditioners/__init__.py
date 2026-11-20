"""Shared inverse-action preconditioners for Krylov solvers."""

from .preconditioners import build_preconditioner, make_preconditioner

__all__ = ["build_preconditioner", "make_preconditioner"]
