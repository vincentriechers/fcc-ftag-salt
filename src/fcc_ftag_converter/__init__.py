"""FCC-ee EDM4hep → Salt-shaped HDF5 converter."""

from __future__ import annotations

from fcc_ftag_converter.cli import build_parser, convert_fcc_edm4hep_to_h5, main

__all__ = ["build_parser", "convert_fcc_edm4hep_to_h5", "main"]
__version__ = "0.2.0"
