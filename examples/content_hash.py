# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Container-independent SHA-256 of the values stored in a NetCDF file.

A file checksum identifies bytes: recompressing a file, or writing it with another HDF5
or zlib version, changes it although every value is the same. ``content_sha256`` hashes
what the file *says* instead: for each variable (sorted by name) its name, dimensions,
stored dtype, shape and values in little-endian C order, plus the attributes that decide
how stored values are decoded (``scale_factor``, ``add_offset``, ``_FillValue``,
``missing_value``, ``units``, ``calendar``).

The hash uses the values as stored, before CF decoding: packed integers are hashed as
integers. Compression, chunking, creation time and descriptive attributes (``history``,
``source``...) do not affect it; a changed dtype or packing does, because that is a
different representation of the data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

SOURCES = Path(__file__).parent / "sources"
VERSION = "xhw-content-v1"
DECODING_ATTRS = ("scale_factor", "add_offset", "_FillValue", "missing_value", "units", "calendar")
BLOCK_BYTES = 64 << 20  # read large variables in slabs of about 64 MiB


def registered(name: str) -> dict:
    """The registered manifest `examples/sources/<name>.json` (needs no xHW)."""
    return json.loads((SOURCES / f"{name}.json").read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of the file bytes (needs no xHW)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    """JSON-safe, exact representation of an attribute value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return np.asarray(value).tolist()


def _update_values(digest, var: xr.Variable) -> None:
    dtype = var.dtype
    if dtype.kind in "OSU":
        digest.update(json.dumps(np.asarray(var.values).tolist(), ensure_ascii=True).encode())
        return
    little = dtype.newbyteorder("<")
    if var.ndim == 0 or var.size == 0:
        digest.update(np.ascontiguousarray(np.asarray(var.values), dtype=little).tobytes())
        return
    row_bytes = max(1, var.size // var.shape[0]) * dtype.itemsize
    rows = max(1, BLOCK_BYTES // row_bytes)
    for start in range(0, var.shape[0], rows):
        block = np.asarray(var[start : start + rows].values)
        digest.update(np.ascontiguousarray(block, dtype=little).tobytes())


def content_sha256(path: str | Path) -> str:
    """Hex SHA-256 of the stored values and decoding attributes of a NetCDF file."""
    digest = hashlib.sha256(VERSION.encode() + b"\n")
    opened = xr.open_dataset(
        path,
        engine="h5netcdf",
        decode_cf=False,
        mask_and_scale=False,
        decode_times=False,
        decode_coords=False,
        decode_timedelta=False,
    )
    with opened as ds:
        for name in sorted(ds.variables):
            var = ds.variables[name]
            header = {
                "name": str(name),
                "dims": [str(d) for d in var.dims],
                "dtype": "object" if var.dtype.kind in "OSU" else var.dtype.newbyteorder("<").str,
                "shape": [int(n) for n in var.shape],
                "attrs": {k: _plain(var.attrs[k]) for k in DECODING_ATTRS if k in var.attrs},
            }
            digest.update(json.dumps(header, sort_keys=True, allow_nan=True).encode() + b"\n")
            _update_values(digest, var)
    return digest.hexdigest()
