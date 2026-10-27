"""Clip pfcand input features in a preprocessed UPP H5 to mean +- N*std, in place.

The FCC covariance/ToF features have extreme tails (|z| up to ~1e3-1e4) that
destabilise bf16 training in salt (NaN weights mid-epoch). weaver -- the
framework the reference ParT tagger was trained with -- standardises AND clips
every input to a fixed range during preprocessing, so the reference model never
saw these tails. Salt applies no clipping, so we clip the stored features once,
here. Means/stds come from norm_dict.yaml, so post-clip normalised values are
bounded by +-N (default 5, matching weaver's default clip range).

Only the pfcand INPUT features (variables yaml `pfcand.inputs`) are touched;
labels, valid mask and jets fields are left as-is.

Usage: python clip_pfcand_features.py <file.h5> <norm_dict.yaml> <variables.yaml> [nsigma]
"""

import contextlib
import sys

import h5py
import numpy as np
import yaml

with contextlib.suppress(ImportError):
    import hdf5plugin  # noqa: F401  (UPP files use blosc compression)


def main() -> int:
    """Clip pfcand features to +-nsigma of the UPP norm_dict, in place.

    Returns
    -------
    int
        Process exit status.
    """
    path, norm_path, vars_path = sys.argv[1:4]
    nsigma = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

    norm = yaml.safe_load(open(norm_path))["pfcand"]
    features = yaml.safe_load(open(vars_path))["pfcand"]["inputs"]

    with h5py.File(path, "r+") as f:
        ds = f["pfcand"]
        n = ds.shape[0]
        step = 200_000  # rows per chunk, bounds memory at ~60 MB/field
        for name in features:
            if name not in norm or name not in ds.dtype.names:
                continue
            lo = norm[name]["mean"] - nsigma * norm[name]["std"]
            hi = norm[name]["mean"] + nsigma * norm[name]["std"]
            n_clipped = 0
            for start in range(0, n, step):
                sl = slice(start, min(start + step, n))
                x = ds.fields(name)[sl]
                mask = (x < lo) | (x > hi)
                if not mask.any():
                    continue
                n_clipped += int(mask.sum())
                ds[name, sl] = np.clip(x, lo, hi)
            print(f"  {name:28s} clipped {n_clipped:>9,d} values", flush=True)
    print(f"done: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
