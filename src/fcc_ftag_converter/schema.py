"""Salt H5 schema: dataset specs, reference helpers, append-oriented writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parent


def default_schema_path() -> Path:
    """Path to the salt_schema.json shipped inside the package.

    Returns
    -------
    Path
        Location of the packaged schema snapshot.
    """
    return _PACKAGE_ROOT / "configs" / "salt_schema.json"


@dataclass(frozen=True)
class DatasetSpec:
    """One output dataset: its name and, for per-jet datasets, its slot count."""

    name: str
    max_items: int | None = None


DEFAULT_DATASETS = (
    DatasetSpec("jets", None),
    DatasetSpec("tracks_ghost", 50),
    DatasetSpec("tracks_ghost_rich_ip", 50),
    DatasetSpec("flows", 50),
    DatasetSpec("calo", 50),
    DatasetSpec("electrons", 10),
    DatasetSpec("muons", 5),
)
GLOBAL_DTYPE = np.dtype([("pt_btagJes", "f4"), ("eta_btagJes", "f4")])
FCC_TRAINING_DATASETS = (DatasetSpec("global", None), *DEFAULT_DATASETS)

# Fields the converter actively computes and writes. Used by
# ``_ensure_finite_training_defaults`` to sanitise NaN that may come out of
# divisions (e.g. zero jet pt). Placeholder fields are deliberately left out
# so their NaN/sentinel fillvalues survive into the output as a "missing"
# signal instead of being silently turned into 0.0.
TARGET_VARIABLES: dict[str, tuple[str, ...]] = {
    "tracks": (
        "pt",
        "d0",
        "z0SinTheta",
        "dphi",
        "deta",
        "qOverP",
        "lifetimeSignedD0Significance",
        "lifetimeSignedZ0SinThetaSignificance",
        "phiUncertainty",
        "thetaUncertainty",
        "qOverPUncertainty",
        "d0Uncertainty",
        "z0SinThetaUncertainty",
        "leptonID",
        "ftagTruthTypeLabel",
    ),
    "flows": ("pt", "energy", "deta", "dphi", "isCharged"),
    "electrons": (
        "pt",
        "ptfrac",
        "ptrel",
        "dr",
        "abs_eta",
        "eta",
        "phi",
        "ftag_et",
        "qOverP",
        "d0RelativeToBeamspot",
        "d0RelativeToBeamspotSignificance",
        "ftag_ptVarCone30OverPt",
    ),
    "muons": ("pt", "eta", "phi", "ptfrac", "ptrel", "dr", "qOverP"),
}

TRACK_TARGET_FIELDS = TARGET_VARIABLES["tracks"]
FLOW_TARGET_FIELDS = TARGET_VARIABLES["flows"]
ELECTRON_TARGET_FIELDS = TARGET_VARIABLES["electrons"]


def _fillvalue_array(
    fillvalue: np.void | np.generic | None, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.empty(shape, dtype=dtype)
    if fillvalue is None:
        array.fill(0)
    else:
        array[...] = fillvalue
    if dtype.names and "valid" in dtype.names:
        array["valid"] = False
    return array


def _dtype_from_descr(descr: list) -> np.dtype:
    """Reconstruct a numpy structured dtype from a JSON-serialisable ``descr``.

    Parameters
    ----------
    descr : list
        ``np.dtype.descr`` as loaded from JSON, so shapes are lists not tuples.

    Returns
    -------
    np.dtype
        The reconstructed structured dtype.
    """
    fields: list[tuple[Any, ...]] = []
    for entry in descr:
        if len(entry) == 2:
            fields.append((entry[0], entry[1]))
        else:
            fields.append((entry[0], entry[1], tuple(entry[2])))
    return np.dtype(fields)


def _load_schema_from_json(
    json_path: str | Path, dataset_names: tuple[str, ...]
) -> dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]]:
    """Load a frozen Salt schema snapshot from JSON. No H5 access required.

    Parameters
    ----------
    json_path : str | Path
        Schema snapshot, normally :func:`default_schema_path`.
    dataset_names : tuple[str, ...]
        Datasets to extract, in output order.

    Returns
    -------
    dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]]
        Per dataset: dtype, reference shape and fillvalue.

    Raises
    ------
    KeyError
        If a requested dataset is not in the snapshot.
    """
    with open(json_path) as handle:
        data = json.load(handle)
    schema: dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]] = {}
    for name in dataset_names:
        if name not in data["datasets"]:
            raise KeyError(f"Dataset {name!r} not found in {json_path}")
        entry = data["datasets"][name]
        dtype = _dtype_from_descr(entry["dtype_descr"])
        ref_shape = (1,) if entry["ndim"] == 1 else (1, int(entry["n_slots"]))
        fillvalue: np.void | None = None
        fv_dict = entry.get("fillvalue")
        if fv_dict and dtype.names:
            void = np.zeros(1, dtype=dtype)
            for field in dtype.names:
                if field not in fv_dict:
                    continue
                value = fv_dict[field]
                sub_dtype = dtype.fields[field][0]  # type: ignore[index]
                if value is None:
                    # JSON serialises NaN as null; restore it for float fields so
                    # masking models see NaN, not a real 0.0.
                    if np.issubdtype(sub_dtype, np.floating):
                        void[field] = np.nan
                    continue
                void[field] = value
            fillvalue = void[0]
        schema[name] = (dtype, ref_shape, fillvalue)
    return schema


def _load_schema_from_h5(
    reference: str | Path, dataset_names: tuple[str, ...]
) -> dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]]:
    """Open a reference H5 once and snapshot dtype/shape/fillvalue per dataset.

    Parameters
    ----------
    reference : str | Path
        An existing Salt-shaped H5 file.
    dataset_names : tuple[str, ...]
        Datasets to snapshot, in output order.

    Returns
    -------
    dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]]
        Per dataset: dtype, reference shape and fillvalue.
    """
    schema: dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]] = {}
    with h5py.File(reference, "r") as handle:
        for name in dataset_names:
            ds = handle[name]
            schema[name] = (ds.dtype, tuple(ds.shape), ds.fillvalue)
    return schema


def _load_reference_schema(
    reference: str | Path, dataset_names: tuple[str, ...]
) -> dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]]:
    """Dispatch to JSON or H5 loader based on file extension.

    Parameters
    ----------
    reference : str | Path
        A ``.json`` snapshot or a Salt-shaped H5 file.
    dataset_names : tuple[str, ...]
        Datasets to load, in output order.

    Returns
    -------
    dict[str, tuple[np.dtype, tuple[int, ...], np.void | np.generic | None]]
        Per dataset: dtype, reference shape and fillvalue.
    """
    path = Path(reference)
    if path.suffix.lower() == ".json":
        return _load_schema_from_json(path, dataset_names)
    return _load_schema_from_h5(path, dataset_names)


def empty_rows_from_reference(
    reference: str | Path | dict, dataset: str, num_jets: int
) -> np.ndarray:
    """Allocate placeholder rows matching the reference schema.

    Parameters
    ----------
    reference : str | Path | dict
        A path (opens the file) or a pre-loaded schema dict returned by
        :func:`_load_reference_schema` (no I/O).
    dataset : str
        Which dataset to allocate.
    num_jets : int
        Number of rows.

    Returns
    -------
    np.ndarray
        Rows pre-filled with the dataset's fillvalue.
    """
    if isinstance(reference, dict):
        dtype, ref_shape, fillvalue = reference[dataset]
    else:
        with h5py.File(reference, "r") as handle:
            ds = handle[dataset]
            dtype, ref_shape, fillvalue = ds.dtype, tuple(ds.shape), ds.fillvalue
    shape = (num_jets,) if len(ref_shape) == 1 else (num_jets, ref_shape[1])
    return _fillvalue_array(fillvalue, dtype, shape)


def _global_rows_from_jets(jet_rows: np.ndarray) -> np.ndarray:
    rows = np.zeros(len(jet_rows), dtype=GLOBAL_DTYPE)
    rows["pt_btagJes"] = jet_rows["pt_btagJes"]
    rows["eta_btagJes"] = jet_rows["eta_btagJes"]
    return rows


class SaltH5Writer:
    """Append-oriented writer for Salt-style structured HDF5 arrays."""

    def __init__(
        self,
        output: str | Path,
        reference: str | Path,
        *,
        datasets: tuple[DatasetSpec, ...],
        dtype_overrides: dict[str, np.dtype] | None = None,
        compression: str = "gzip",
        compression_opts: int = 7,
    ) -> None:
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.output, "w")
        self.handle.attrs["producer"] = "fcc-ftag-convert"
        self.handle.attrs["reference_schema"] = str(reference)
        self._sizes = {spec.name: 0 for spec in datasets}
        dtype_overrides = dtype_overrides or {}

        ref_needed = tuple(spec.name for spec in datasets if spec.name not in dtype_overrides)
        ref_schema = _load_reference_schema(reference, ref_needed) if ref_needed else {}

        for spec in datasets:
            if spec.name in dtype_overrides:
                dtype = dtype_overrides[spec.name]
                ref_shape: tuple[int, ...] = (0,) if spec.max_items is None else (0, spec.max_items)
                fillvalue = None
            else:
                dtype, ref_shape, fillvalue = ref_schema[spec.name]

            # Ensure the on-disk fillvalue carries valid=False so resize()
            # auto-fills placeholder rows correctly without a payload write.
            if dtype.names and "valid" in dtype.names:
                void = np.zeros(1, dtype=dtype)
                if fillvalue is not None:
                    void[0] = fillvalue
                void["valid"] = False
                fillvalue = void[0]

            shape: tuple[int, ...] = (0,)
            maxshape: tuple[int | None, ...] = (None,)
            chunks: tuple[int, ...] = (2048,)
            if len(ref_shape) > 1:
                shape = (0, ref_shape[1])
                maxshape = (None, ref_shape[1])
                chunks = (2048, ref_shape[1])

            self.handle.create_dataset(
                spec.name,
                shape=shape,
                maxshape=maxshape,
                chunks=chunks,
                dtype=dtype,
                compression=compression,
                compression_opts=compression_opts,
                fillvalue=fillvalue,
            )

    def append(self, name: str, rows: np.ndarray) -> None:
        """Append ``rows`` to the end of dataset ``name``."""
        dataset = self.handle[name]
        start = self._sizes[name]
        stop = start + len(rows)
        dataset.resize((stop, *dataset.shape[1:]))
        dataset[start:stop] = rows
        self._sizes[name] = stop

    def extend_with_fill(self, name: str, n_rows: int) -> None:
        """Extend a dataset by ``n_rows`` empty rows, no payload written.

        h5py auto-fills the newly resized region with the dataset's fillvalue
        (and for structured dtypes with a ``valid`` field that fillvalue has
        ``valid=False``). Skipping the materialised numpy buffer saves a lot
        of memory and CPU for placeholder datasets like ``calo``.
        """
        if n_rows <= 0:
            return
        dataset = self.handle[name]
        start = self._sizes[name]
        stop = start + n_rows
        dataset.resize((stop, *dataset.shape[1:]))
        self._sizes[name] = stop

    def close(self) -> None:
        """Close the underlying H5 file."""
        self.handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
