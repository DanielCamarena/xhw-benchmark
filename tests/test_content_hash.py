# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""content_sha256 ignores compression and descriptive metadata, and notices real changes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5netcdf")
import xarray as xr  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import content_hash  # noqa: E402
import verify_data  # noqa: E402


def packed_dataset() -> xr.Dataset:
    rng = np.random.default_rng(3)
    ints = rng.integers(-2000, 2000, size=(40, 6, 5)).astype("int16")
    ints[0, 0, 0] = -32768  # fill value
    ds = xr.Dataset(
        {
            "sst": (
                ("time", "lat", "lon"),
                ints,
                {
                    "scale_factor": 0.01,
                    "add_offset": 273.15,
                    "_FillValue": np.int16(-32768),
                    "units": "kelvin",
                },
            ),
            "counts": (("time", "lat", "lon"), np.full((40, 6, 5), 24, dtype="int32")),
        },
        coords={
            "time": np.arange(40, dtype="int32"),
            "lat": np.arange(6, dtype="float32"),
            "lon": np.arange(5, dtype="float32"),
        },
    )
    ds["time"].attrs["units"] = "days since 1991-01-01"
    return ds


def write(ds: xr.Dataset, path: Path, level: int | None = None) -> Path:
    encoding = (
        {v: {"zlib": True, "complevel": level, "shuffle": True} for v in ds.data_vars}
        if level
        else {}
    )
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    return path


def test_recompression_keeps_content_but_changes_bytes(tmp_path):
    ds = packed_dataset()
    plain, packed = write(ds, tmp_path / "a.nc"), write(ds, tmp_path / "b.nc", level=9)
    assert content_hash.file_sha256(plain) != content_hash.file_sha256(packed)
    assert content_hash.content_sha256(plain) == content_hash.content_sha256(packed)


def test_hash_follows_block_size_independence(tmp_path, monkeypatch):
    path = write(packed_dataset(), tmp_path / "a.nc")
    whole = content_hash.content_sha256(path)
    monkeypatch.setattr(content_hash, "BLOCK_BYTES", 100)  # force many small slabs
    assert content_hash.content_sha256(path) == whole


def test_descriptive_attributes_do_not_matter_but_decoding_ones_do(tmp_path):
    base = content_hash.content_sha256(write(packed_dataset(), tmp_path / "base.nc"))
    described = packed_dataset()
    described.attrs["history"] = "recompressed on another day"
    described["sst"].attrs["long_name"] = "SST"
    assert content_hash.content_sha256(write(described, tmp_path / "d.nc")) == base
    rescaled = packed_dataset()
    rescaled["sst"].attrs["scale_factor"] = 0.02
    assert content_hash.content_sha256(write(rescaled, tmp_path / "r.nc")) != base


def test_a_changed_value_or_dtype_changes_the_hash(tmp_path):
    base = content_hash.content_sha256(write(packed_dataset(), tmp_path / "base.nc"))
    edited = packed_dataset()
    edited["sst"].values[3, 2, 1] += 1
    assert content_hash.content_sha256(write(edited, tmp_path / "e.nc")) != base
    narrowed = packed_dataset()
    narrowed["counts"] = narrowed["counts"].astype("uint8")  # same numbers, other representation
    assert content_hash.content_sha256(write(narrowed, tmp_path / "n.nc")) != base


def test_verifier_distinguishes_bytes_content_and_mismatch(tmp_path):
    ds = packed_dataset()
    original = write(ds, tmp_path / "original.nc")
    recompressed = write(ds, tmp_path / "recompressed.nc", level=9)
    edited = packed_dataset()
    edited["sst"].values[0, 1, 1] += 1
    other = write(edited, tmp_path / "other.nc")
    file_hash = content_hash.file_sha256(original)
    content = content_hash.content_sha256(original)
    assert verify_data.judge(original, file_hash, content) == "identical bytes"
    assert verify_data.judge(recompressed, file_hash, content) == "same content"
    assert verify_data.judge(other, file_hash, content) == "MISMATCH"
    assert verify_data.judge(recompressed, file_hash, None) == "MISMATCH"
