"""Compatibility wrapper for the spectrometer simulation.

This file mirrors the minimal interface earlier sims expect: it exposes a
`SimulatedSpectrometer` class via the top-level `simulation` package so scripts
that import `from simulation.spectrometer_sim import SimulatedSpectrometer` will
continue to work after the move.
"""

from __future__ import annotations

from spectrometer.spectrometer import SimulatedSpectrometer

__all__ = ["SimulatedSpectrometer"]
