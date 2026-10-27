"""Interleave the unshuffled taujet block at the head of a UPP train split, in place.

UPP's countup resampling ran out of taujets (7.33M available vs the 8.96M
per-class target), and train splits 005-030 each begin with ~64k rows of pure
taujets. Salt draws batches as contiguous 500-row slices, so those rows become
single-class batches (~2.7% of every epoch) with structurally biased gradients.

Fix: mix every 500-row block of the pure-tau prefix with 7 random 500-row
blocks from the rest of the file, applying the same row permutation to every
per-jet dataset. Any contiguous 500-row window then holds <~1/4 taujets.
Deterministic (seeded by filename) and idempotent: exits when no pure-tau
prefix is found. Only row ORDER changes; content is untouched.

Usage: fix_tau_blocks.py <pp_output_train_split_*.h5>
"""

import contextlib
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np

with contextlib.suppress(ImportError):
    import hdf5plugin  # noqa: F401  (UPP files use blosc compression)

BLOCK = 500  # = salt batch_size; keep aligned with the training config
MIX = 7  # partner blocks mixed into each tau block
TAU = 6  # flavour_label index of taujets


def main(path: str) -> int:
    """Mix partner blocks into the pure-tau head of a UPP-shuffled file.

    Parameters
    ----------
    path : str
        H5 file to rewrite in place.

    Returns
    -------
    int
        Process exit status.
    """
    with h5py.File(path, "r+") as f:
        jets = f["jets"]
        n = jets.shape[0]
        head = jets.fields("flavour_label")[: min(n, 200_000)]
        n_pure = int(np.argmax(head != TAU)) if (head != TAU).any() else len(head)
        if n_pure < BLOCK:
            print(f"no pure-tau prefix, nothing to do: {path}")
            return 0
        tau_blocks = (n_pure + BLOCK - 1) // BLOCK
        n_blocks = n // BLOCK
        print(f"{path}: {n_pure:,d} pure-tau rows ({tau_blocks} blocks of {BLOCK})")

        datasets = [d for d in f.values() if isinstance(d, h5py.Dataset) and d.shape[0] == n]
        rng = np.random.default_rng(zlib.crc32(Path(path).name.encode()))
        partners = rng.choice(np.arange(tau_blocks, n_blocks), size=MIX * tau_blocks, replace=False)
        for i in range(tau_blocks):
            blocks = [i, *partners[MIX * i : MIX * (i + 1)]]
            perm = rng.permutation(len(blocks) * BLOCK)
            for ds in datasets:
                rows = np.concatenate([ds[b * BLOCK : (b + 1) * BLOCK] for b in blocks])
                rows = rows[perm]
                for j, b in enumerate(blocks):
                    ds[b * BLOCK : (b + 1) * BLOCK] = rows[j * BLOCK : (j + 1) * BLOCK]

        # verify: no aligned 500-row window in the former block region stays pure
        lab = jets.fields("flavour_label")[: tau_blocks * BLOCK]
        purity = max(
            np.bincount(lab[s : s + BLOCK]).max() / BLOCK for s in range(0, len(lab), BLOCK)
        )
        print(f"done: {path} (max single-class fraction in former prefix: {purity:.2f})")
        return 0 if purity < 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
