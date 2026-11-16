"""Minimal, self-contained benchmark interface for the anonymous release.

This package deliberately does not import the private research codebase.  It
implements only what an external researcher needs to load the frozen panels,
resolve their sequences, evaluate a new model and reproduce the reported tables.
"""

from __future__ import annotations

__all__ = ["metrics", "panels", "predictions"]

__version__ = "1.0.0"
