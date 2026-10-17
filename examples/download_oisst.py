# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional OISST acquisition; default prints the request without networking.

Reads NOAA OISST v2.1 AVHRR-only daily SST from the public Source Cooperative Icechunk
store, whose ``daily`` group references NOAA's original files on anonymous S3. Install
``examples/requirements-download.txt`` in a separate acquisition environment. No
credentials are needed.

Each store chunk holds one global day: about 0.5 MB is transferred per day even for a
small box, so 1991-2025 moves several GB. Months are read with retries and each year is
kept as a resumable partial file. Preliminary days are rejected. The snapshot is pinned
(default: the one of the registered tutorial input), so the request is reproducible while
that snapshot exists. New files are not automatically equivalent to the registered
tutorial input; compare their SHA-256 with examples/sources/oisst.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from content_hash import file_sha256, registered

STORE = "https://data.source.coop/ocean-icechunks/noaa-oisst/oisst.icechunk"
REGISTERED_SNAPSHOT = registered("oisst")["version"].removeprefix("icechunk snapshot ")


#: region -> latitudes, longitudes (0-360), shift applied to the saved longitudes, manifest key
REGIONS = {
    "peru": ([-12.0, -4.0], [278.0, 283.0], -360.0, "oisst"),
    # Nino 3.4 box of the ENSO index: 5S-5N, 170W-120W
    "equatorial": ([-5.0, 5.0], [190.0, 240.0], -360.0, "oisst_equatorial"),
}


def request(first: int, last: int, snapshot: str, region: str = "peru") -> dict:
    latitude, longitude, shift, _ = REGIONS[region]
    return dict(
        store=STORE,
        group="daily",
        snapshot_id=snapshot,
        region=region,
        variables=["sst", "preliminary"],
        latitude=latitude,
        longitude_0_360=longitude,
        longitude_shift_applied=shift,
        time=[f"{first}-01-01", f"{last}-12-31"],
        output_filename=f"oisst_v21_{region}_{first}_{last}.nc",
    )


def open_daily(snapshot: str):
    import icechunk
    import xarray as xr

    repo = icechunk.Repository.open(icechunk.http_storage(STORE))
    auth = icechunk.containers_credentials(
        {p: icechunk.s3_anonymous_credentials() for p in repo.config.virtual_chunk_containers}
    )
    session = repo.reopen(authorize_virtual_chunk_access=auth).readonly_session(
        snapshot_id=snapshot
    )
    return xr.open_zarr(session.store, group="daily", consolidated=False, chunks=None)


def fetch_year(ds, year: int, config: dict, target: Path, attempts: int = 8):
    import icechunk
    import pandas as pd
    import xarray as xr

    lat, lon = slice(*config["latitude"]), slice(*config["longitude_0_360"])
    months = []
    for first in pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS"):
        last = first + pd.offsets.MonthEnd(0) + pd.Timedelta("1D") - pd.Timedelta("1s")
        for attempt in range(1, attempts + 1):
            try:
                sub = ds[config["variables"]].sel(time=slice(first, last), lat=lat, lon=lon)
                months.append(sub.squeeze("zlev", drop=True).load())
                break
            except icechunk.StorageError as err:  # transient S3 connection drops
                if attempt == attempts:
                    raise
                wait = min(15 * attempt, 120)
                print(f"    {first:%Y-%m}: retry {attempt} in {wait}s ({str(err)[:80]})")
                time.sleep(wait)
                ds = open_daily(config["snapshot_id"])
    xr.concat(months, dim="time").to_netcdf(target, engine="h5netcdf")
    return ds


def download(config: dict, data_dir: Path, first: int, last: int) -> Path:
    import xarray as xr
    import zarr

    zarr.config.set({"async.concurrency": 8})  # more simultaneous requests drop connections
    target = data_dir / config["output_filename"]
    partials = data_dir / f"oisst_partial_{config['region']}"
    partials.mkdir(parents=True, exist_ok=True)
    ds = open_daily(config["snapshot_id"])
    parts = []
    for year in range(first, last + 1):
        part = partials / f"{year}.nc"
        parts.append(part)
        if part.exists():
            print(f"{year}: already downloaded")
            continue
        t0 = time.time()
        ds = fetch_year(ds, year, config, part)
        print(f"{year}: {time.time() - t0:.0f} s")
    pieces = []
    for part in parts:
        with xr.open_dataset(part, engine="h5netcdf") as opened:
            pieces.append(opened.load())
    series = xr.concat(pieces, dim="time")
    if int(series["preliminary"].sum()):
        raise SystemExit("Preliminary days found; this request uses final data only.")
    series = series.drop_vars("preliminary")
    series = series.assign_coords(lon=series["lon"] + config["longitude_shift_applied"])
    series["sst"].attrs = {"units": "degC", "long_name": "Daily sea surface temperature"}
    series.attrs = {
        "source": "NOAA OISST v2.1 AVHRR-only daily (Huang et al. 2021)",
        "store": STORE,
        "store_snapshot": config["snapshot_id"],
    }
    series.to_netcdf(target, engine="h5netcdf")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--region", choices=sorted(REGIONS), default="peru")
    parser.add_argument("--years", default="1991-2025")
    parser.add_argument("--snapshot", default=REGISTERED_SNAPSHOT)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        first, last = map(int, args.years.split("-"))
        if not 1982 <= first <= last:
            raise ValueError
    except ValueError:
        parser.error("--years must be a valid inclusive range, for example 1991-2025")
    config = request(first, last, args.snapshot, args.region)
    print(json.dumps(config, indent=2))
    if not args.download:
        print("Print only. Add --download to transfer data (several GB for 1991-2025).")
        return
    try:
        import icechunk  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install examples/requirements-download.txt in an acquisition environment; "
            "see the README. It is not an xHW runtime dependency."
        ) from exc
    target = args.data_dir / config["output_filename"]
    if target.exists():
        raise FileExistsError(f"{target.name} already exists; use a separate data directory.")
    path = download(config, args.data_dir, first, last)
    digest = file_sha256(path)
    try:
        reference = registered(REGIONS[args.region][3])
    except FileNotFoundError:  # a region whose input has not been registered yet
        reference = {"sha256": None}
    (args.data_dir / f"oisst_{args.region}_download_manifest.json").write_text(
        json.dumps(
            {
                "accessed": datetime.now(timezone.utc).isoformat(),
                "client_version": version("icechunk"),
                "file": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "request": config,
                "matches_registered_input": digest == reference["sha256"],
                "review_status": "new retrieval; not the registered reviewed input",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{path.name} sha256={digest}")
    if reference["sha256"] is None:
        print("This region has no registered input yet; register the file to reproduce it.")
    elif digest != reference["sha256"]:
        print(
            "This file differs from the registered tutorial input. Analyse it as a new run, "
            "with its own SourceRef; the registered cases expect the registered checksum."
        )


if __name__ == "__main__":
    main()
