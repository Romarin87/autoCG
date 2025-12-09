"""
AutoCG package initializer.

Exposes common modules for convenience when installed via pip.
"""

from . import chem  # noqa: F401
from . import generate  # noqa: F401

__all__ = ["chem", "generate"]
