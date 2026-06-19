"""socketcad -- parametric transtibial socket rectification engine (MVP).

A research/demo prototype that turns a residual-limb mesh plus clinical-intent
parameters into a print-ready socket STL. NOT a cleared medical device.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .spec import RectificationSpec, CustomReliefRegion, spec_from_dict  # noqa: F401
