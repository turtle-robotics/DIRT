"""LM models package for DIRT — lightweight loader and backend wrappers.

Expose: `get_model()` helper returning a model-like callable with `generate(prompt)`.
"""

from .loader import get_model

__all__ = ["get_model"]
