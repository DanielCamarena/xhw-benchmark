# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional ESA SST CCI acquisition from CEDA; default prints the request.

Reads the ESA SST CCI Level-4 climate data record (analysis, version 3.0.1, 0.05 degree,
daily) from the CEDA archive and keeps only a regional box. No account or key is needed.
Each day is a global HDF5 NetCDF file, but only the bytes of the box are fetched with HTTP
Range requests; the script refuses to fall back to downloading whole files.

    python examples/download_sst_cci.py --data-dir data                      # print the request
    python examples/download_sst_cci.py --data-dir data --download           # fetch and combine
    python examples/download_sst_cci.py --start 2020-01-01 --end 2020-01-31 --download   # one month

It takes one request per day, so 1991-2021 is about 11 400 small requests; use
``--workers``. Daily partial files are kept under ``sst_cci_partial/`` and reused, so an
interrupted run resumes; the combined file keeps the stored (packed) values unchanged.
The record (CDR) ends on 2021-12-31; the same archive continues from 2022-01-01 with an
interim extension (ICDR, a different file series) that reached 2024-06-22 when checked on
2026-10-11. The default stops at the CDR end to keep one homogeneous record; a later --end
joins the two series. For 1991-2025 use download_ostia.py. Values are in kelvin. Install
``examples/requirements-download.txt``. Archive:
https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from content_hash import content_sha256, file_sha256

BASE_URL = (
    "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/"
    "CDR_v3/Analysis/L4/v3.0.1/"
)
CATALOGUE_URL = "https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/"
ARCHIVE_REVISION = "3.0.1"
CDR_LAST_DAY = "2021-12-31"
TIMEOUT = 120
RETRIES = 3
BLOCK_SIZE = 131072
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs).get("href") or "")


def day_directory(day) -> str:
    return urljoin(BASE_URL, f"{day:%Y/%m/%d/}")


def discover(day, client) -> str:
    """URL of the single daily file listed in the archive directory of `day`."""
    directory = day_directory(day)
    response = client.get(directory, timeout=TIMEOUT)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    stamp = f"{day:%Y%m%d}"
    found = [
        urljoin(directory, h)
        for h in parser.links
        if urlparse(h).path.endswith(".nc") and stamp in h
    ]
    found = list(dict.fromkeys(found))
    if len(found) != 1:
        raise ValueError(f"Expected one file for {day.date()}, found {len(found)} in {directory}")
    if urlparse(found[0]).netloc != urlparse(directory).netloc:
        raise ValueError("Unexpected host in the archive listing")
    return found[0]


def partial_path(data_dir: Path, day) -> Path:
    return data_dir / "sst_cci_partial" / f"{day:%Y}" / f"sst_cci_{day:%Y%m%d}.nc"


def fetch_day(task: tuple) -> str:
    """Download the box of one day (runs in a worker process); returns the partial file path."""
    import aiohttp
    import fsspec
    import pandas as pd
    import requests
    import xarray as xr
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    day_text, data_dir, variables, box = task
    day = pd.Timestamp(day_text)
    target = partial_path(Path(data_dir), day)
    if target.exists():
        return str(target)
    client = requests.Session()
    client.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        ),
    )
    south, north, west, east = box
    for attempt in range(RETRIES):
        try:
            url = discover(day, client)
            probe = client.get(url, headers={"Range": "bytes=0-7"}, timeout=TIMEOUT, stream=True)
            try:
                if probe.status_code != 206:
                    raise RuntimeError(
                        "Server does not support HTTP Range; refusing a full download"
                    )
                if probe.raw.read(8) != HDF5_SIGNATURE:
                    raise ValueError("This reader needs an HDF5-based NetCDF file")
                etag = probe.headers.get("ETag", "")
            finally:
                probe.close()
            fs = fsspec.filesystem(
                "http",
                client_kwargs={"trust_env": True, "timeout": aiohttp.ClientTimeout(total=TIMEOUT)},
            )
            with fs.open(url, "rb", block_size=BLOCK_SIZE, cache_type="readahead") as stream:
                with xr.open_dataset(
                    stream,
                    engine="h5netcdf",
                    cache=False,
                    decode_cf=False,
                    mask_and_scale=False,
                    decode_times=False,
                ) as source:
                    absent = set(variables) - set(source.data_vars)
                    if absent:
                        raise ValueError(f"Variables missing from the file: {sorted(absent)}")
                    part = (
                        source[list(variables)]
                        .sel(lat=slice(south, north), lon=slice(west, east))
                        .load()
                    )
            part.attrs.update(
                source_url=url,
                source_etag=etag,
                archive_revision=ARCHIVE_REVISION,
                acquisition="regional HTTP Range subset of the daily global file",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            part.to_netcdf(tmp, engine="h5netcdf")
            os.replace(tmp, target)
            return str(target)
        except Exception:
            if attempt + 1 == RETRIES:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def combine(paths: list[str], output: Path) -> None:
    """Concatenate daily partial files in time, keeping stored values, with lossless zlib."""
    import xarray as xr

    parts = []
    for path in paths:
        with xr.open_dataset(
            path, engine="h5netcdf", decode_cf=False, mask_and_scale=False, decode_times=False
        ) as opened:
            parts.append(opened.load())
    series = xr.concat(
        parts,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )
    encoding = {v: {"zlib": True, "complevel": 4, "shuffle": True} for v in series.data_vars}
    series.to_netcdf(output, engine="h5netcdf", encoding=encoding)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", default="1991-01-01")
    parser.add_argument("--end", default=CDR_LAST_DAY)
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        default=(-12.0, -4.0, -82.0, -77.0),
        metavar=("SOUTH", "NORTH", "WEST", "EAST"),
        help="degrees, longitudes in -180..180 (default: the Peru box)",
    )
    parser.add_argument("--variables", nargs="+", default=["analysed_sst"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    import pandas as pd

    try:
        first, last = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if not pd.Timestamp("1981-09-01") <= first <= last <= pd.Timestamp.today().normalize():
            raise ValueError
    except ValueError:
        parser.error("--start/--end must be dates from 1981-09-01 to today, start <= end")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    days = pd.date_range(first, last, freq="D")
    output = args.data_dir / f"sst_cci_v3_{first:%Y%m%d}_{last:%Y%m%d}.nc"
    print(
        json.dumps(
            {
                "archive": BASE_URL,
                "archive_revision": ARCHIVE_REVISION,
                "days": len(days),
                "first_directory": day_directory(days[0]),
                "last_directory": day_directory(days[-1]),
                "box_south_north_west_east": list(args.box),
                "variables": args.variables,
                "partial_files": str(args.data_dir / "sst_cci_partial"),
                "output": str(output),
            },
            indent=2,
        )
    )
    if not args.download:
        print("Print only. Add --download to fetch the box day by day.")
        return
    try:
        import aiohttp  # noqa: F401
        import fsspec  # noqa: F401
        import requests  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install examples/requirements-download.txt (fsspec, aiohttp, requests) in an "
            "acquisition environment; see the README."
        ) from exc
    if output.exists():
        raise FileExistsError(f"{output.name} already exists; use a separate data directory.")
    tasks = [(f"{d:%Y-%m-%d}", str(args.data_dir), args.variables, tuple(args.box)) for d in days]
    # Processes, not threads: h5py holds a global lock while it reads a remote file.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        paths = []
        for i, path in enumerate(pool.map(fetch_day, tasks), 1):
            paths.append(path)
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} days", flush=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    combine(paths, output)
    manifest = {
        "accessed": datetime.now(timezone.utc).isoformat(),
        "archive": BASE_URL,
        "catalogue": CATALOGUE_URL,
        "archive_revision": ARCHIVE_REVISION,
        "box_south_north_west_east": list(args.box),
        "variables": args.variables,
        "days": len(days),
        "file": output.name,
        "sha256": file_sha256(output),
        "content_sha256": content_sha256(output),
        "bytes": output.stat().st_size,
        "review_status": "new retrieval; not a registered reviewed input",
    }
    (args.data_dir / "sst_cci_download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{output.name} written; see sst_cci_download_manifest.json")


if __name__ == "__main__":
    main()
