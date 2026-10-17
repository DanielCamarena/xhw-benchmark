# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Build the compressed NetCDF test datasets and their manifest from the raw inputs.

    python examples/make_test_datasets.py --data-dir data --output-dir test-data

Four datasets for the Peru box (12S-4S x 82W-77W), each a single NetCDF4 file:

  oisst_peru_1991_2025.nc        NOAA OISST v2.1, 0.25 degree      (lossless recompression)
  ostia_peru_1991_2025.nc        Met Office OSTIA, 0.05 degree     (yearly files joined in time)
  era5_peru_daily_1991_2025.nc   ERA5 local-day TX/TN              (lossless recompression)
  glorys12v1_peru_decimated_1993_2023.nc  GLORYS12V1 thetao on the tutorial's retained voxels

Nothing is re-quantised: variables keep the integers, dtype, scale and offset of the
source and only gain zlib level 9 with byte shuffling, which is lossless. Each result
is checked before it is accepted: single-source datasets must have the same
``content_sha256`` as their source; joined datasets are compared array by array with their
parts. ``test_datasets.json`` records the file hash, content hash, size and origin of each.
Use ``verify_data.py --data-dir test-data`` to check a copy later. Inputs that are absent
are skipped, so a subset can be built.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from content_hash import content_sha256, file_sha256

SOURCES = Path(__file__).parent / "sources"
COMPRESSION = {"engine": "h5netcdf", "filter": "zlib", "complevel": 9, "shuffle": True}
RAW = dict(decode_cf=False, mask_and_scale=False, decode_times=False, decode_coords=False)


def open_raw(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="h5netcdf", **RAW)


def write_compressed(ds: xr.Dataset, path: Path) -> None:
    encoding = {
        v: {"zlib": True, "complevel": COMPRESSION["complevel"], "shuffle": True}
        for v in ds.data_vars
        if ds[v].ndim >= 2
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    ds.to_netcdf(tmp, engine="h5netcdf", encoding=encoding)
    tmp.replace(path)


def same_arrays(a: xr.Dataset, b: xr.Dataset) -> bool:
    """Stored values of every variable in `b` equal those in `a` (same dtype and shape)."""
    for name in b.variables:
        x, y = a[name].values, b[name].values
        if (
            x.dtype != y.dtype
            or x.shape != y.shape
            or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        ):
            return False
    return True


def recompress(source: Path, output: Path) -> dict:
    """Rewrite one file with compression and prove that its content hash is unchanged."""
    with open_raw(source) as ds:
        write_compressed(ds.load(), output)
    before, after = content_sha256(source), content_sha256(output)
    if before != after:
        output.unlink()
        raise RuntimeError(f"Content changed while recompressing {source.name}")
    return {
        "operation": "lossless recompression",
        "derived_from": {
            "file": source.name,
            "sha256": file_sha256(source),
            "content_sha256": before,
        },
    }


def join_in_time(parts: list[Path], output: Path, operation: str, select=None) -> dict:
    """Concatenate files along time (after an optional selection) and compare each part."""
    pieces = []
    for part in parts:
        with open_raw(part) as ds:
            pieces.append((select(ds) if select else ds).load())
    units = {piece["time"].attrs.get("units") for piece in pieces}
    if len(units) != 1:
        raise RuntimeError(f"Parts use different time units: {sorted(map(str, units))}")
    joined = xr.concat(
        pieces,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )
    write_compressed(joined, output)
    with open_raw(output) as written:
        written = written.load()
    start = 0
    for piece in pieces:
        stop = start + piece.sizes["time"]
        if not same_arrays(piece, written.isel(time=slice(start, stop))):
            output.unlink()
            raise RuntimeError(f"Joined file differs from its part {start}:{stop}")
        start = stop
    return {
        "operation": operation,
        "derived_from": {
            "files": {
                p.name: {"sha256": file_sha256(p), "content_sha256": content_sha256(p)}
                for p in parts
            }
        },
    }


NAMES = (
    "oisst",
    "oisst_equatorial",
    "era5_daily",
    "era5_hourly_sample",
    "glorys_decimated",
    "ostia",
)


def hourly_sample(source: Path, output: Path, first="2023-03-01", last="2023-03-04") -> dict:
    """Keep four hourly days of the registered 2023 file, with the stored values unchanged."""
    with xr.open_dataset(source, engine="h5netcdf") as decoded:
        dim = "valid_time" if "valid_time" in decoded.dims else "time"
        index = np.flatnonzero(
            (decoded[dim].values >= np.datetime64(first))
            & (decoded[dim].values < np.datetime64(last) + np.timedelta64(1, "D"))
        )
    with open_raw(source) as raw:
        whole = raw[["mx2t", "mn2t", "t2m"]].load()
    part = whole.isel({dim: index})
    write_compressed(part, output)
    with open_raw(output) as written:
        written = written.load()
    if not same_arrays(part, written):
        output.unlink()
        raise RuntimeError("The hourly sample differs from its source")
    return {
        "operation": f"selection of {first}..{last} (hourly), values unchanged",
        "derived_from": {
            "file": f"era5_peru_hourly/{source.name}",
            "sha256": file_sha256(source),
        },
    }


def build(data_dir: Path, output_dir: Path, only=NAMES) -> dict:
    datasets = {}

    def register(key: str, output: Path, info: dict, note: str) -> None:
        with open_raw(output) as ds:
            sizes = {k: int(v) for k, v in ds.sizes.items()}
            variables = [v for v in ds.data_vars]
        datasets[key] = {
            "file": output.name,
            "sha256": file_sha256(output),
            "content_sha256": content_sha256(output),
            "bytes": output.stat().st_size,
            "dims": sizes,
            "variables": variables,
            "note": note,
            **info,
        }
        print(
            f"{key:14s} {output.stat().st_size / 1e6:8.2f} MB  {output.name}",
            flush=True,
        )

    src = data_dir / "oisst_v21_peru_1991_2025.nc"
    if "oisst" in only and src.is_file():
        out = output_dir / "oisst_peru_1991_2025.nc"
        register(
            "oisst",
            out,
            recompress(src, out),
            "NOAA OISST v2.1 AVHRR-only daily SST, degC, 0.25 degree (registered input).",
        )
    src = data_dir / "oisst_v21_equatorial_1991_2025.nc"
    if "oisst_equatorial" in only and src.is_file():
        out = output_dir / "oisst_equatorial_1991_2025.nc"
        register(
            "oisst_equatorial",
            out,
            recompress(src, out),
            "NOAA OISST v2.1 AVHRR-only daily SST, degC, 0.25 degree, equatorial Pacific "
            "(Nino 3.4 box) 5S-5N, 170W-120W.",
        )
    src = data_dir / "era5_peru_daily_1991_2025.nc"
    if "era5_daily" in only and src.is_file():
        out = output_dir / "era5_peru_daily_1991_2025.nc"
        register(
            "era5_daily",
            out,
            recompress(src, out),
            "ERA5 local-day (UTC-05:00) TX/TN from mx2t/mn2t plus a t2m contrast, in K "
            "(registered input).",
        )
    hourly = data_dir / "era5_peru_hourly" / "2023.nc"
    if "era5_hourly_sample" in only and hourly.is_file():
        out = output_dir / "era5_peru_hourly_sample_2023_03.nc"
        register(
            "era5_hourly_sample",
            out,
            hourly_sample(hourly, out),
            "ERA5 hourly mx2t, mn2t and t2m for 2023-03-01..04 (the aggregation demonstration).",
        )
    files = sorted((data_dir / "glorys12v1_peru").glob("thetao_*.nc"))
    if "glorys_decimated" in only and files:
        rec = json.loads((SOURCES / "glorys.json").read_text(encoding="utf-8"))
        levels = rec["decimation"]["levels"]
        files = [data_dir / "glorys12v1_peru" / name for name in rec["files"]]
        out = output_dir / "glorys12v1_peru_decimated_1993_2023.nc"
        info = join_in_time(
            files,
            out,
            "decimation: every 8th latitude and longitude, retained levels, joined in time",
            select=lambda ds: ds[["thetao"]].isel(
                depth=levels,
                latitude=slice(None, None, 8),
                longitude=slice(None, None, 8),
            ),
        )
        info["decimation"] = {"levels": levels, "horizontal_stride": 8}
        register(
            "glorys_decimated",
            out,
            info,
            "GLORYS12V1 thetao (degC) on the voxels used by the GLORYS tutorial.",
        )
    ostia = sorted((data_dir / "ostia_peru").glob("analysed_sst_*.nc"))
    if "ostia" in only and ostia:
        out = output_dir / "ostia_peru_1991_2025.nc"
        info = join_in_time(ostia, out, "yearly files joined in time")
        register(
            "ostia",
            out,
            info,
            "Met Office OSTIA reprocessed analysed_sst, kelvin, 0.05 degree, dataset "
            "METOFFICE-GLO-SST-L4-REP-OBS-SST version 202003.",
        )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("test-data"))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=NAMES,
        default=list(NAMES),
        help="build only these datasets",
    )
    parser.add_argument("--manifest", type=Path, default=SOURCES / "test_datasets.json")
    args = parser.parse_args()
    datasets = build(args.data_dir, args.output_dir, args.only)
    if not datasets:
        raise SystemExit(f"No registered input found in {args.data_dir}")
    if args.manifest.is_file():  # keep entries of datasets that were not rebuilt this time
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))["datasets"]
        datasets = {**previous, **datasets}
    manifest = {
        "format": "xhw-test-datasets-1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compression": COMPRESSION,
        "datasets": datasets,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
