# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional ERA5 acquisition and daily TX/TN construction; default prints requests.

Requests hourly 2 m temperature (t2m) and the maximum/minimum 2 m temperature since the
previous post-processing (mx2t, mn2t) from the Copernicus Climate Data Store (CDS), plus the
land-sea mask, over 12S-4S x 82W-77W. Install ``examples/requirements-download.txt`` in a
separate acquisition environment, keep your CDS key in ``~/.cdsapirc`` and accept the ERA5
single-levels licence once on the CDS web page.

    python examples/download_era5.py --data-dir data                # print the requests
    python examples/download_era5.py --data-dir data --download     # hourly files + lsm
    python examples/download_era5.py --data-dir data --daily        # local-day TX/TN file

Each year is requested as two half-years (the CDS cost limit rejects a full year of three
variables). For local days at UTC-05:00 through the last year, the first hours of the
following UTC year are also requested. Finished pieces are kept, so an interrupted run only
repeats unfinished requests. ``--daily`` aggregates with ``xhw.aggregate_daily``: TX is the
maximum of period-ending mx2t, TN the minimum of mn2t; t2m extrema are kept only as a
contrast. New files are not automatically equivalent to the registered tutorial inputs.
The development repository also compared this route with the ARCO-ERA5 copy and an Earth
Data Hub mirror; those checks are not part of this example.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from content_hash import file_sha256, registered

DATASET = "reanalysis-era5-single-levels"
VARIABLES = {
    "t2m": "2m_temperature",
    "mx2t": "maximum_2m_temperature_since_previous_post_processing",
    "mn2t": "minimum_2m_temperature_since_previous_post_processing",
}
AREA = [-4, -82, -12, -77]  # north, west, south, east
DAY_DEFINITION = "offset:-05:00"
BOUNDARY_HOURS = 7  # UTC hours 00-06 of the following year close the last local day


def requests(first: int, last: int) -> list[tuple[str, dict]]:
    """Named CDS requests: two half-years per year, boundary hours, and the land-sea mask."""
    names = list(VARIABLES.values())
    out = []
    for year in range(first, last + 1):
        for half, months in (("H1", range(1, 7)), ("H2", range(7, 13))):
            out.append(
                (
                    f"{year}_{half}",
                    {
                        "variable": names,
                        "year": [str(year)],
                        "month": [f"{m:02d}" for m in months],
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": [f"{h:02d}:00" for h in range(24)],
                    },
                )
            )
    out.append(
        (
            f"{last + 1}_boundary",
            {
                "variable": names,
                "year": [str(last + 1)],
                "month": ["01"],
                "day": ["01"],
                "time": [f"{h:02d}:00" for h in range(BOUNDARY_HOURS)],
            },
        )
    )
    out.append(
        (
            "lsm",
            {
                "variable": ["land_sea_mask"],
                "year": ["2020"],
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
            },
        )
    )
    return [
        (
            name,
            {
                "product_type": ["reanalysis"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": AREA,
                **body,
            },
        )
        for name, body in out
    ]


def time_name(ds) -> str:
    return "valid_time" if "valid_time" in ds.dims else "time"


def read_cds(raw: Path):
    """Open a downloaded CDS file; a zip of several NetCDF parts is merged."""
    import xarray as xr

    try:
        if zipfile.is_zipfile(raw):
            with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(raw) as archive:
                archive.extractall(tmp)
                parts = []
                for name in sorted(Path(tmp).glob("*.nc")):
                    with xr.open_dataset(name, engine="h5netcdf") as opened:
                        parts.append(opened.load())
            ds = xr.merge(parts, compat="override")
        else:
            with xr.open_dataset(raw, engine="h5netcdf") as opened:
                ds = opened.load()
    finally:
        raw.unlink(missing_ok=True)
    return ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])


def download(jobs: list[tuple[str, dict]], hourly_dir: Path, workers: int) -> None:
    import cdsapi
    import xarray as xr

    pieces = hourly_dir / "pieces"
    pieces.mkdir(parents=True, exist_ok=True)
    todo = [(n, r) for n, r in jobs if not (pieces / f"{n}.nc").exists()]
    print(f"{len(todo)} of {len(jobs)} CDS requests left, {workers} at a time", flush=True)

    def fetch(job):
        name, body = job
        raw = pieces / f"{name}.download"
        cdsapi.Client(quiet=True).retrieve(DATASET, body, str(raw))
        read_cds(raw).to_netcdf(pieces / f"{name}.tmp", engine="h5netcdf")
        (pieces / f"{name}.tmp").replace(pieces / f"{name}.nc")
        print(f"{name}: done", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, todo))
    # One hourly file per UTC year, as the registered manifest lists them.
    years = sorted({n.split("_")[0] for n, _ in jobs if n != "lsm"})
    for year in years:
        target = hourly_dir / f"{year}.nc"
        if target.exists():
            continue
        parts = []
        for path in sorted(pieces.glob(f"{year}_*.nc")):
            with xr.open_dataset(path, engine="h5netcdf") as opened:
                parts.append(opened.load())
        sub = xr.concat(parts, dim=time_name(parts[0])) if len(parts) > 1 else parts[0]
        sub.to_netcdf(target, engine="h5netcdf")
        print(f"{year}: {sub.sizes[time_name(sub)]} hours", flush=True)
    if not (hourly_dir / "lsm.nc").exists():
        with xr.open_dataset(pieces / "lsm.nc", engine="h5netcdf") as opened:
            both = opened.load()
        lsm = both["lsm"].isel({time_name(both): 0}, drop=True)
        lsm.to_netcdf(hourly_dir / "lsm.nc", engine="h5netcdf")


def build_daily(hourly_dir: Path, target: Path, first: int, last: int) -> None:
    """Aggregate hourly files year by year to local-day TX/TN (UTC-05:00)."""
    import xarray as xr

    import xhw

    def hours(path: Path, count: int | None = None):
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            ds = opened.isel({time_name(opened): slice(0, count)}).load()
        return ds.rename({time_name(ds): "time"}) if time_name(ds) != "time" else ds

    pieces = []
    for year in range(first, last + 1):
        # Local day D at -05:00 ends at 05:00 UTC of D+1: append the next file's first hours.
        hourly = xr.concat(
            [hours(hourly_dir / f"{year}.nc"), hours(hourly_dir / f"{year + 1}.nc", 8)],
            dim="time",
        )
        out = [
            xhw.aggregate_daily(
                hourly,
                variable=v,
                statistic=s,
                sampling=m,
                input_clock="UTC",
                day_definition=DAY_DEFINITION,
                name=n,
            )
            for v, s, m, n in (
                ("mx2t", "max", "period_ending", "tx"),
                ("mn2t", "min", "period_ending", "tn"),
                ("t2m", "max", "instantaneous", "tx_t2m"),
                ("t2m", "min", "instantaneous", "tn_t2m"),
            )
        ]
        pieces.append(
            xr.merge(out, compat="override").sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        )
        print(f"{year}: daily aggregation done", flush=True)
    days = xr.concat(pieces, dim="time")
    with xr.open_dataset(hourly_dir / "lsm.nc", engine="h5netcdf") as opened:
        lsm = opened["lsm"].load()
    days["lsm"] = lsm.drop_vars([c for c in lsm.coords if c not in ("latitude", "longitude")])
    days.attrs = {"source": f"Copernicus Climate Data Store {DATASET} (ERA5, Copernicus licence)"}
    days.to_netcdf(target, engine="h5netcdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--years", default="1991-2025")
    parser.add_argument(
        "--workers", type=int, default=4, help="simultaneous CDS requests (default 4)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="request the hourly files")
    group.add_argument("--daily", action="store_true", help="build the local-day TX/TN file")
    args = parser.parse_args()
    try:
        first, last = map(int, args.years.split("-"))
        if not 1940 <= first <= last:
            raise ValueError
    except ValueError:
        parser.error("--years must be a valid inclusive range, for example 1991-2025")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    hourly_dir = args.data_dir / "era5_peru_hourly"
    daily = args.data_dir / f"era5_peru_daily_{first}_{last}.nc"
    jobs = requests(first, last)
    if args.daily:
        if daily.exists():
            raise FileExistsError(f"{daily.name} already exists; use a separate data directory.")
        build_daily(hourly_dir, daily, first, last)
        files = sorted(hourly_dir.glob("*.nc"))
        digest = file_sha256(daily)
        (args.data_dir / "era5_download_manifest.json").write_text(
            json.dumps(
                {
                    "accessed": datetime.now(timezone.utc).isoformat(),
                    "dataset": DATASET,
                    "area_nwse": AREA,
                    "day_definition": DAY_DEFINITION,
                    "hourly_files_sha256": {p.name: file_sha256(p) for p in files},
                    "file": daily.name,
                    "sha256": digest,
                    "matches_registered_input": digest == registered("era5")["sha256"],
                    "review_status": "new retrieval; not the registered reviewed input",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{daily.name} written; see era5_download_manifest.json")
        return
    for name, body in jobs:
        print(name, json.dumps(body))
    if not args.download:
        print(f"Print only: {len(jobs)} requests. Add --download to submit them to the CDS.")
        return
    try:
        import cdsapi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install examples/requirements-download.txt in an acquisition environment; "
            "see the README. It is not an xHW runtime dependency."
        ) from exc
    download(jobs, hourly_dir, args.workers)
    print(
        f"Hourly files in {hourly_dir.name}/ (cdsapi {version('cdsapi')}). "
        "Next: --daily builds the local-day TX/TN file."
    )


if __name__ == "__main__":
    main()
