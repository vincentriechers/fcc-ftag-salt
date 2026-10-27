#!/usr/bin/env python
"""Dump the Salt H5 schema into the JSON config used by the converter.

Run this once whenever the reference Salt H5 changes. The converter then
reads only the JSON and no longer needs access to the reference H5.

Usage:
    python scripts/dump_schema.py <reference.h5> [-o <path>]

By default it overwrites the copy shipped inside the package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from fcc_ftag_converter.schema import default_schema_path

_DATASETS = ("jets", "tracks_ghost", "tracks_ghost_rich_ip", "flows", "calo", "electrons", "muons")


def _serialise_fillvalue(fv, dtype: np.dtype) -> dict | None:
    if fv is None or not dtype.names:
        return None
    result: dict[str, object] = {}
    for name in dtype.names:
        # `dtype.names` being non-empty is what makes `.fields` non-None.
        sub = dtype.fields[name][0]  # type: ignore[index]
        value = fv[name]
        if np.issubdtype(sub, np.floating):
            result[name] = float(value) if np.isfinite(value) else None
        elif np.issubdtype(sub, np.bool_):
            result[name] = bool(value)
        elif np.issubdtype(sub, np.integer):
            result[name] = int(value)
        else:
            result[name] = str(value)
    return result


def main(argv: list[str] | None = None) -> int:
    """Write a JSON schema snapshot from a Salt reference H5.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line, defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Salt reference H5 to read the schema from")
    parser.add_argument("-o", "--output", default=str(default_schema_path()))
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {"_source": str(args.reference), "datasets": {}}
    with h5py.File(args.reference, "r") as f:
        for name in _DATASETS:
            if name not in f:
                print(f"warning: {name!r} not in {args.reference}", file=sys.stderr)
                continue
            ds = f[name]
            entry = {
                "ndim": int(ds.ndim),
                "n_slots": int(ds.shape[1]) if ds.ndim == 2 else None,
                "dtype_descr": [list(t) for t in ds.dtype.descr],
                "fillvalue": _serialise_fillvalue(ds.fillvalue, ds.dtype),
            }
            data["datasets"][name] = entry

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
