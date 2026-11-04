# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional ESA SST CCI acquisition from CEDA; default prints the request.

Reads the ESA SST CCI Level-4 climate data record (analysis, version 3.0.1, 0.05 degree,
daily) from the CEDA archive and keeps only a regional box. No account or key is needed.
Each day is a global HDF5 NetCDF file, read with HTTP Range requests (the script refuses
to fall back to downloading whole files); the transfer is a few MB per day, not only the
box (revision 6, E1).

    python examples/download_sst_cci.py --data-dir data                      # print the request
    python examples/download_sst_cci.py --data-dir data --download           # fetch and combine
    python examples/download_sst_cci.py --start 2020-01-01 --end 2020-01-31 --download   # one month

The box is ``--region`` or ``--box S N W E`` with ``--box-policy`` (default ``enclose``:
nodes on or beyond every edge); a box crossing the antimeridian is read in two parts and
joined west to east, keeping the stored longitudes. One request per day, so 1991-2021 is
about 11 300 days; use ``--workers`` (processes). Daily parts live under
``sst_cci_partial/<request hash>/YYYY/`` and a ledger ``sst_cci_ledger_<hash>.json`` records
each day's URL, ETag, Last-Modified, series and SHA-256; a validated part is reused, after a
HEAD request confirms its ETag (``--no-check-etag`` skips it). Only transient errors are
retried; every failed day is reported and nothing is combined until all days validate. The
combined file keeps the stored (packed) values and dtype.

The record (CDR) ends on 2021-12-31; the same archive continues from 2022-01-01 with an
interim extension (ICDR, a different file series). ``--end`` cannot pass the CDR end without
``--include-icdr``, and never the last published day, which is looked up before starting.
For 1991-2025 use download_ostia.py. Values are in kelvin. Install
``examples/requirements-download.txt``. Archive:
https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import _acquire as aq
import numpy as np
from content_hash import file_sha256

BASE_URL = (
    "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/"
    "CDR_v3/Analysis/L4/v3.0.1/"
)
CATALOGUE_URL = "https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/"
SOURCE = "esa-sst-cci-l4"
ARCHIVE_REVISION = "3.0.1"
#: Identifies how a part is read; a new read variant (revision 6, E1, measured in F3) must
#: change it so parts read the old way are not reused as if they were the same.
READER = "fsspec-readahead-131072+h5netcdf"
FIRST_DAY = date(1981, 9, 1)
CDR_LAST_DAY = date(2021, 12, 31)
GRID_STEP = 0.05  # degrees, latitude and longitude (3600 x 7200 cell centres)
TIMEOUT = 120
BLOCK_SIZE = 131072
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
PACKING = ("scale_factor", "add_offset", "_FillValue", "units")
SAVE_EVERY = 25  # ledger saves while recording finished days
#: Wait before retry ``n`` of a transient error (seconds); tests replace it.
WAIT: Callable[[int], float] = aq.backoff


class RangeNotSupported(aq.PermanentError):
    """The server answered a ranged request with the whole file (not 206)."""


class NotHdf5(aq.PermanentError):
    """The daily file does not start with the HDF5 signature."""


class MissingVariables(aq.PermanentError):
    """A requested variable is not in the daily file."""


class ListingError(aq.PermanentError):
    """A day's directory does not list exactly one file of the expected series."""


class NotPublished(aq.PermanentError):
    """The requested period goes beyond the last day published in the archive."""


def _classify_ceda(exc: BaseException) -> str | None:
    # fsspec's HTTP backend turns a 404 into FileNotFoundError (no status attached).
    if isinstance(exc, FileNotFoundError):
        return aq.PERMANENT
    # Revision 6 retries every 5xx; the shared list has only 500, 502, 503 and 504.
    status = aq._http_status(exc)
    if status is not None and 500 <= status <= 599:
        return aq.TRANSIENT
    return None


def classify(exc: BaseException) -> str:
    """Revision 6, section 6: 5xx, 429, cuts and timeouts are transient; the rest permanent.

    A 404 inside the published coverage, a server without Range, a file that is not HDF5,
    missing variables and a listing without exactly one file are permanent.
    """
    return aq.classify_error(exc, extra=(_classify_ceda,))


# --- Archive layout and HTTP access ----------------------------------------------------------


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs).get("href") or "")


def _links(text: str) -> list[str]:
    parser = Links()
    parser.feed(text)
    return parser.links


def day_directory(day: date) -> str:
    return urljoin(BASE_URL, f"{day:%Y/%m/%d/}")


def series_of(url: str) -> str:
    """``CDR`` or ``ICDR``, read from the file name (``...-GLOB_CDR3.0-...``)."""
    found = re.search(r"_(I?CDR)\d", urlparse(url).path.rsplit("/", 1)[-1])
    if found is None:
        raise ListingError(f"cannot tell the series (CDR/ICDR) of {url}")
    return found.group(1)


def discover(day: date, client) -> str:
    """URL of the single daily file listed in the archive directory of ``day``."""
    directory = day_directory(day)
    stamp = f"{day:%Y%m%d}"
    found = [
        urljoin(directory, h)
        for h in _links(client.listing(directory))
        if urlparse(h).path.endswith(".nc") and stamp in h
    ]
    found = list(dict.fromkeys(found))
    if len(found) != 1:
        raise ListingError(f"expected one file for {day}, found {len(found)} in {directory}")
    if urlparse(found[0]).netloc != urlparse(directory).netloc:
        raise ListingError("unexpected host in the archive listing")
    return found[0]


def _subdirectories(directory: str, text: str) -> list[int]:
    """Two-digit child directories (months or days) listed in ``directory``."""
    out = set()
    for href in _links(text):
        url = urljoin(directory, href)
        if url.startswith(directory) and re.fullmatch(r"\d{2}/?", url[len(directory) :]):
            out.add(int(url[len(directory) :].rstrip("/")))
    return sorted(out)


def _not_found(exc: BaseException) -> bool:
    return isinstance(exc, FileNotFoundError) or aq._http_status(exc) in (404, 410)


def last_published_day(client, end: date) -> date:
    """Latest day on or before ``end`` with a directory in the archive (revision 6, R4).

    Lists the year directory, then its months from the latest down, and their days; a year
    without a directory (404) moves to the previous year.
    """
    for year in range(end.year, FIRST_DAY.year - 1, -1):
        year_dir = urljoin(BASE_URL, f"{year}/")
        try:
            months = _subdirectories(year_dir, client.listing(year_dir))
        except Exception as exc:
            if _not_found(exc):
                continue
            raise
        for month in reversed(months):
            if (year, month) > (end.year, end.month):
                continue
            month_dir = urljoin(year_dir, f"{month:02d}/")
            days = [
                date(year, month, d)
                for d in _subdirectories(month_dir, client.listing(month_dir))
                if 1 <= d <= 31
            ]
            days = [d for d in days if d <= end]
            if days:
                return max(days)
    raise NotPublished(f"no day published on or before {end} in {BASE_URL}")


def _headers(headers: Mapping) -> dict:
    return {"etag": headers.get("ETag", ""), "last_modified": headers.get("Last-Modified", "")}


class CedaClient:
    """HTTP access: listings, HEAD and the 8-byte probe with requests; reads with fsspec."""

    def __init__(self, timeout: float = TIMEOUT):
        import requests

        self.timeout = timeout
        self.session = requests.Session()

    def listing(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def head(self, url: str) -> dict:
        response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
        response.raise_for_status()
        return _headers(response.headers)

    def probe(self, url: str) -> tuple[int, bytes, dict]:
        """Status, first 8 bytes and headers of a ``Range: bytes=0-7`` request."""
        response = self.session.get(
            url, headers={"Range": "bytes=0-7"}, timeout=self.timeout, stream=True
        )
        try:
            response.raise_for_status()
            return response.status_code, response.raw.read(8), _headers(response.headers)
        finally:
            response.close()

    def open(self, url: str):
        import aiohttp
        import fsspec

        fs = fsspec.filesystem(
            "http",
            client_kwargs={"trust_env": True, "timeout": aiohttp.ClientTimeout(total=self.timeout)},
        )
        return fs.open(url, "rb", block_size=BLOCK_SIZE, cache_type="readahead")

    def close(self) -> None:
        self.session.close()


# --- One day (runs in a worker process) ------------------------------------------------------


def part_request(box: aq.Box, policy: str, variables: Sequence[str]) -> dict:
    """What a daily part depends on; its hash names the part folder (revision 6, R1)."""
    return {
        "source": SOURCE,
        "archive": BASE_URL,
        "archive_revision": ARCHIVE_REVISION,
        "reader": READER,
        "box": box.as_dict(),
        "box_policy": policy,
        "variables": list(variables),
    }


def part_path(data_dir: Path, request_hash: str, day: date) -> Path:
    return data_dir / "sst_cci_partial" / request_hash / f"{day:%Y}" / f"sst_cci_{day:%Y%m%d}.nc"


def unvalidated_path(target: Path) -> Path:
    return target.with_name(f"{target.stem}.unvalidated.nc")


def subset(source, variables: Sequence[str], box: aq.Box, policy: str):
    """The box of ``source`` (loaded), joined west to east if it crosses the seam."""
    import xarray as xr

    lat_slice = aq.select_indices(
        source["lat"].values, box.south, box.north, policy, step=GRID_STEP
    )
    lon_slices = aq.select_lon_indices(
        source["lon"].values, box.west, box.east, policy, step=GRID_STEP
    )
    pieces = [source[list(variables)].isel(lat=lat_slice, lon=s).load() for s in lon_slices]
    if len(pieces) == 1:
        return pieces[0]
    return xr.concat(
        pieces,
        dim="lon",
        join="exact",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )


def fetch_once(client, task: Mapping) -> dict:
    """Read the box of one day into ``task["raw"]``; raises on any problem."""
    import xarray as xr

    day = date.fromisoformat(task["day"])
    url = discover(day, client)
    series = series_of(url)
    if series != "CDR" and not task["include_icdr"]:
        raise ListingError(f"{day} is in the {series} series; pass --include-icdr to use it")
    status, first_bytes, headers = client.probe(url)
    if status != 206:
        raise RangeNotSupported(
            f"server answered {status} to a Range request; refusing a full download"
        )
    if first_bytes != HDF5_SIGNATURE:
        raise NotHdf5(f"{url} is not an HDF5-based NetCDF file")
    box = aq.Box(*task["box"])
    variables = list(task["variables"])
    with client.open(url) as stream:
        with xr.open_dataset(
            stream,
            engine="h5netcdf",
            cache=False,
            decode_cf=False,
            mask_and_scale=False,
            decode_times=False,
        ) as source:
            absent = sorted(set(variables) - set(source.data_vars))
            if absent:
                raise MissingVariables(f"variables missing from {url}: {absent}")
            part = subset(source, variables, box, task["policy"])
            dtypes = {v: str(source[v].dtype) for v in variables}
    part.attrs.update(
        source_url=url,
        source_etag=headers["etag"],
        source_last_modified=headers["last_modified"],
        series=series,
        archive_revision=ARCHIVE_REVISION,
        acquisition="regional HTTP Range subset of the daily global file",
        request_sha256=task["request_sha256"],
    )
    aq.write_netcdf_atomic(part, task["raw"])
    return {
        "kind": "fetched",
        "source": {"url": url, "series": series, **headers},
        "fetched_utc": aq.utc_now(),
        "expected": {
            "day": task["day"],
            "lat": part["lat"].values,
            "lon": part["lon"].values,
            "dtypes": dtypes,
        },
    }


def _head_or_none(client, url: str) -> dict | None:
    try:
        return client.head(url)
    except Exception as exc:
        if _not_found(exc):
            return None
        raise


def process_day(task: Mapping, client_factory: Callable[[], object] | None = None) -> dict:
    """Fetch one day, or confirm a reused part's ETag first (``task["mode"] == "check"``).

    Transient errors are retried ``task["retries"]`` times. If the ETag changed and the new
    fetch fails, the exception carries ``etag_change`` so the stale part is not kept.
    """
    client = (client_factory or CedaClient)()

    def retried(fn):
        return aq.call_with_retries(
            fn, retries=task["retries"], classify=classify, wait=lambda n: WAIT(n)
        )

    try:
        change = None
        if task["mode"] == "check":
            now = retried(lambda: _head_or_none(client, task["url"]))
            if now is not None and now["etag"] == task["etag"]:
                return {"kind": "reused", "checked_utc": aq.utc_now()}
            change = {
                "utc": aq.utc_now(),
                "old_etag": task["etag"],
                "new_etag": None if now is None else now["etag"],
            }
        try:
            result = retried(lambda: fetch_once(client, task))
        except Exception as exc:
            if change is not None:
                exc.etag_change = change
            raise
        result["etag_change"] = change
        return result
    finally:
        client.close()


def run_days(
    tasks: Mapping[str, Mapping],
    workers: int,
    client_factory: Callable[[], object] | None,
    on_done: Callable[[aq.Outcome], None],
) -> dict[str, aq.Outcome]:
    """Run every day and collect every error (``as_completed``); processes if ``workers > 1``.

    Processes, not threads: h5py holds a global lock while it reads a remote file. With one
    worker the days run in this process.
    """
    outcomes: dict[str, aq.Outcome] = {}

    def done(key, call):
        try:
            outcome = aq.Outcome(key, True, result=call())
        except Exception as exc:
            outcome = aq.Outcome(key, False, error=exc)
        outcomes[key] = outcome
        on_done(outcome)

    if workers == 1:
        for key, task in tasks.items():
            done(key, lambda task=task: process_day(task, client_factory))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_day, task, client_factory): key for key, task in tasks.items()
            }
            for future in as_completed(futures):
                done(futures[future], future.result)
    return {key: outcomes[key] for key in tasks}


# --- Validation and combination (main process) -----------------------------------------------


def _open(path):
    import xarray as xr

    return xr.open_dataset(
        path, engine="h5netcdf", decode_cf=False, mask_and_scale=False, decode_times=False
    )


def decoded_day(time) -> str | None:
    """ISO date of the first value of a CF ``time`` variable (None if it cannot decode)."""
    import xarray as xr

    try:
        value = xr.decode_cf(xr.Dataset(coords={"time": time.variable}))["time"].values[0]
    except (ValueError, TypeError, OverflowError):
        return None
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="D"))
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else None


def validate_part(
    path: Path, expected: Mapping, box: aq.Box, policy: str, variables: Sequence[str]
) -> dict:
    """Check a written part against what was read; raises ``aq.ValidationError``."""
    try:
        with _open(path) as part:
            problems = []
            if set(part.data_vars) != set(variables):
                problems.append(f"variables {sorted(part.data_vars)} != {sorted(variables)}")
            for v in variables:
                if v in part and str(part[v].dtype) != expected["dtypes"][v]:
                    problems.append(f"{v} dtype {part[v].dtype} != stored {expected['dtypes'][v]}")
            if part.sizes.get("time") != 1:
                problems.append(f"time has {part.sizes.get('time')} values, not 1")
            elif (day := decoded_day(part["time"])) != expected["day"]:
                problems.append(f"time decodes to {day}, not the requested {expected['day']}")
            lat, lon = part["lat"].values, part["lon"].values
    except (OSError, ValueError, KeyError) as exc:
        raise aq.ValidationError(f"{path.name} is unreadable: {exc}") from exc
    if not (np.array_equal(lat, expected["lat"]) and np.array_equal(lon, expected["lon"])):
        problems.append("coordinates differ from those read")
    report = aq.coverage_report(box, lat, lon, policy, lat_step=GRID_STEP, lon_step=GRID_STEP)
    problems += report["problems"]
    if problems:
        raise aq.ValidationError(f"{path.name}: " + "; ".join(problems))
    return {"shape": {"lat": int(lat.size), "lon": int(lon.size)}, "coverage_ok": True}


def combine(
    paths: Sequence[Path],
    output: Path,
    box: aq.Box,
    policy: str,
    variables: Sequence[str],
    days: Sequence[date] | None = None,
) -> dict:
    """Join daily parts in time, keeping stored values and dtype; returns the coverage report.

    Every part must have the same coordinates, variables, dtypes and packing attributes as
    the first (revision 6, R2); the join is ``exact``. With ``days``, each part's time must
    decode to its day, which also covers reused parts. Raises ``aq.ValidationError``.
    """
    import xarray as xr

    parts = []
    for path in paths:
        with _open(path) as opened:
            parts.append(opened.load())
    first = parts[0]
    for path, part in zip(paths, parts, strict=True):
        name = Path(path).name
        if set(part.data_vars) != set(variables):
            raise aq.ValidationError(f"{name}: variables {sorted(part.data_vars)}")
        for axis in ("lat", "lon"):
            if not np.array_equal(part[axis].values, first[axis].values):
                raise aq.ValidationError(f"{name}: {axis} differs from {Path(paths[0]).name}")
        for key in ("units", "calendar"):
            if part["time"].attrs.get(key) != first["time"].attrs.get(key):
                raise aq.ValidationError(f"{name}: time {key} differs from {Path(paths[0]).name}")
        for v in variables:
            if part[v].dtype != first[v].dtype:
                raise aq.ValidationError(f"{name}: {v} dtype {part[v].dtype} != {first[v].dtype}")
            for key in PACKING:
                if not _same(part[v].attrs.get(key), first[v].attrs.get(key)):
                    raise aq.ValidationError(f"{name}: {v} attribute {key} differs")
    for path, part, day in zip(paths, parts, days or [], strict=False):
        if (found := decoded_day(part["time"])) != day.isoformat():
            raise aq.ValidationError(f"{Path(path).name}: time decodes to {found}, not {day}")
    times = np.concatenate([p["time"].values for p in parts])
    if np.any(np.diff(times) <= 0):
        raise aq.ValidationError("daily times are not strictly increasing")
    series = xr.concat(
        parts,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="drop_conflicts",
    )
    for v in variables:
        if series[v].dtype != first[v].dtype:
            raise aq.ValidationError(f"{v} changed dtype to {series[v].dtype} when combined")
    try:
        report = aq.check_coverage(
            box,
            series["lat"].values,
            series["lon"].values,
            policy,
            lat_step=GRID_STEP,
            lon_step=GRID_STEP,
        )
    except aq.CoverageError as exc:
        raise aq.ValidationError(str(exc)) from exc
    encoding = {v: {"zlib": True, "complevel": 4, "shuffle": True} for v in series.data_vars}
    aq.write_netcdf_atomic(series, output, encoding=encoding)
    return report


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


# --- Command line ----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    aq.add_period_arguments(parser, "1991-01-01", CDR_LAST_DAY.isoformat())
    aq.add_box_arguments(parser)
    parser.add_argument("--variables", nargs="+", default=["analysed_sst"])
    parser.add_argument(
        "--include-icdr",
        action="store_true",
        help="allow days after the CDR end (2021-12-31), from the interim ICDR series",
    )
    parser.add_argument(
        "--check-etag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HEAD each reused part's file and fetch the day again if its ETag changed "
        "(default on)",
    )
    aq.add_run_arguments(parser)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.box_obj, args.region_name = aq.box_from_args(args)
        last = date.today() if args.include_icdr else CDR_LAST_DAY
        aq.period_from_args(args, FIRST_DAY, last)
    except ValueError as exc:
        hint = "" if args.include_icdr else " (the CDR ends on 2021-12-31; see --include-icdr)"
        parser.error(f"{exc}{hint}")
    if args.workers < 1 or args.retries < 0:
        parser.error("--workers must be >= 1 and --retries >= 0")
    args.variables = list(dict.fromkeys(args.variables))
    return args


def main(argv: Sequence[str] | None = None, client_factory=None) -> int:
    args = parse_args(argv)
    box, policy, variables = args.box_obj, args.box_policy, args.variables
    request = part_request(box, policy, variables)
    request_hash = aq.request_id(request)
    days = [args.start + timedelta(n) for n in range((args.end - args.start).days + 1)]
    data_dir = args.data_dir
    output = data_dir / f"sst_cci_v3_{args.start:%Y%m%d}_{args.end:%Y%m%d}_{request_hash}.nc"
    ledger_path = data_dir / f"sst_cci_ledger_{request_hash}.json"
    print(
        json.dumps(
            {
                "archive": BASE_URL,
                "archive_revision": ARCHIVE_REVISION,
                "series": "CDR+ICDR" if args.include_icdr else "CDR",
                "days": len(days),
                "first_directory": day_directory(days[0]),
                "last_directory": day_directory(days[-1]),
                "region": args.region_name,
                "box": box.as_dict(),
                "box_policy": policy,
                "longitude_parts": len(aq.split_box(box)),
                "variables": variables,
                "request_id": request_hash,
                "partial_files": str(data_dir / "sst_cci_partial" / request_hash),
                "ledger": str(ledger_path),
                "output": str(output),
                "check_etag": args.check_etag,
            },
            indent=2,
        )
    )
    if not args.download:
        print("Print only. Add --download to fetch the box day by day.")
        return 0
    if client_factory is None:
        try:
            import aiohttp  # noqa: F401
            import fsspec  # noqa: F401
            import requests  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install examples/requirements-download.txt (fsspec, aiohttp, requests) in an "
                "acquisition environment; see the README."
            ) from exc
    client = (client_factory or CedaClient)()
    try:
        last = aq.call_with_retries(
            lambda: last_published_day(client, args.end),
            classify=classify,
            wait=lambda n: WAIT(n),
        )
    finally:
        client.close()
    if last < args.end:
        print(
            f"--end {args.end} is not published: the latest published day on or before it is "
            f"{last}. Nothing was fetched.",
            file=sys.stderr,
        )
        return 1

    ledger = aq.Ledger(ledger_path, autosave=False)
    ids = {day: f"{request_hash}_{day:%Y%m%d}" for day in days}
    tasks = {}
    common = {
        "box": [box.south, box.north, box.west, box.east],
        "policy": policy,
        "variables": variables,
        "include_icdr": args.include_icdr,
        "retries": args.retries,
        "request_sha256": aq.request_sha256(request),
    }
    for day, piece in ids.items():
        target = part_path(data_dir, request_hash, day)
        ledger.plan(piece, {**request, "day": day.isoformat()}, target)
        task = {**common, "day": day.isoformat(), "raw": str(unvalidated_path(target))}
        source = ledger.get(piece).get("source") or {}
        if not args.include_icdr and source.get("series", "CDR") != "CDR":
            task["mode"] = "fetch"  # never reuse an ICDR part without --include-icdr
        elif ledger.reusable(piece):
            if not (args.check_etag and source.get("etag")):
                continue  # reused without a check (none asked, or no ETag was recorded)
            task.update(mode="check", url=source["url"], etag=source["etag"])
        else:
            task["mode"] = "fetch"
        tasks[piece] = task
    ledger.save()
    reused = len(days) - len(tasks)
    print(f"{len(days)} days: {reused} reused, {len(tasks)} to fetch or check", flush=True)

    finished = 0

    def record(outcome: aq.Outcome) -> None:
        nonlocal finished
        finished += 1
        record_outcome(ledger, outcome, tasks[outcome.key], box, policy, variables, args)
        if finished % SAVE_EVERY == 0 or finished == len(tasks):
            ledger.save()
        if finished % 100 == 0 or finished == len(tasks):
            print(f"{finished}/{len(tasks)} days", flush=True)

    try:
        outcomes = run_days(tasks, args.workers, client_factory, record)
    finally:
        ledger.save()
    piece_ids = list(ids.values())
    summary = aq.RunSummary.from_ledger(ledger, piece_ids, outcomes)
    summary.report()
    if summary.exit_code:
        print("Not combined: every day must validate first. Run again to resume.")
        return summary.exit_code

    paths = [ledger.resolve(ledger.get(i)["piece"]["path"]) for i in piece_ids]
    inputs = {i: ledger.get(i)["piece"]["sha256"] for i in piece_ids}
    status = ledger.product_status(output.name, output, inputs)
    # A combination is announced before its file is published, so a run interrupted between
    # publishing and recording it can rebuild that file instead of refusing it.
    pending = ledger.data.setdefault("pending_products", {})
    # The announcement authorises the rebuild even if a part changed since (a new ETag).
    if status in ("unrecorded", "changed") and output.name not in pending:
        print(
            f"{output.name} exists and is not the one this ledger wrote ({status}); "
            "use another --data-dir or remove it.",
            file=sys.stderr,
        )
        return 1
    try:
        if status == "current":
            coverage = coverage_of(output, box, policy)
        else:
            pending[output.name] = inputs
            ledger.save()
            coverage = combine(paths, output, box, policy, variables, days)
    except aq.ValidationError as exc:
        print(f"Not combined: {exc}", file=sys.stderr)
        return 1
    ledger.record_product(output.name, output, inputs, request_sha256=aq.request_sha256(request))
    pending.pop(output.name, None)
    ledger.save()

    day_records = []
    for day, piece in ids.items():
        entry = ledger.get(piece)
        day_records.append(
            {
                "day": day.isoformat(),
                **(entry.get("source") or {}),
                "sha256": entry["piece"]["sha256"],
                "fetched_utc": entry.get("downloaded_utc"),
            }
        )
    counts = {}
    for record_ in day_records:
        counts[record_["series"]] = counts.get(record_["series"], 0) + 1
    manifest = aq.build_manifest(
        source=SOURCE,
        request={
            **request,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "include_icdr": args.include_icdr,
            "region": args.region_name,
        },
        files={output.name: output},
        coverage=coverage,
        retrieval_window=ledger.retrieval_window(piece_ids),
        pieces=piece_ids,
        archive=BASE_URL,
        catalogue=CATALOGUE_URL,
        last_published_day_checked=last.isoformat(),
        series_days=counts,
        days=day_records,
        ledger=ledger_path.name,
        download_software=aq.software_versions(("aiohttp", "requests")),
    )
    manifest_file = aq.write_manifest(data_dir, "sst_cci", manifest)
    print(f"{output.name} written; see {manifest_file.name}")
    return 0


def coverage_of(path: Path, box: aq.Box, policy: str) -> dict:
    """Coverage report of an existing combined file (B6)."""
    with _open(path) as ds:
        try:
            return aq.check_coverage(
                box,
                ds["lat"].values,
                ds["lon"].values,
                policy,
                lat_step=GRID_STEP,
                lon_step=GRID_STEP,
            )
        except aq.CoverageError as exc:
            raise aq.ValidationError(str(exc)) from exc


def record_outcome(
    ledger: aq.Ledger,
    outcome: aq.Outcome,
    task: Mapping,
    box: aq.Box,
    policy: str,
    variables: Sequence[str],
    args: argparse.Namespace,
) -> None:
    """Write one finished day into the ledger: reused, validated part, damaged or failed."""
    piece = outcome.key
    entry = ledger.get(piece)
    if not outcome.ok:
        change = getattr(outcome.error, "etag_change", None)
        error = f"{type(outcome.error).__name__}: {outcome.error}"
        if change is not None:
            ledger.update(
                piece,
                "failed",
                error=error,
                etag_changes=[*entry.get("etag_changes", []), change],
            )
        elif task["mode"] == "check":
            # The part is intact; only its check failed. The run still reports the error.
            ledger.update(piece, etag_check_error=error)
        else:
            ledger.update(piece, "failed", attempts=entry["attempts"] + 1, error=error)
        return
    result = outcome.result
    if result["kind"] == "reused":
        ledger.update(piece, etag_checked_utc=result["checked_utc"], etag_check_error=None)
        return
    fields = {"source": result["source"], "attempts": entry["attempts"] + 1}
    if result["etag_change"] is not None:
        fields["etag_changes"] = [*entry.get("etag_changes", []), result["etag_change"]]
    raw = Path(task["raw"])
    ledger.update(
        piece,
        "downloaded",
        raw={"bytes": raw.stat().st_size, "sha256": file_sha256(raw)},
        **fields,
    )
    target = ledger.resolve(entry["target"])
    try:
        checks = validate_part(raw, result["expected"], box, policy, variables)
    except aq.ValidationError as exc:
        aq.discard_damaged(ledger, piece, raw, exc, keep_failed=args.keep_failed)
        ledger.update(piece, "quarantined" if args.keep_failed else "failed")
        return
    os.replace(raw, target)
    ledger.record_piece(piece, target, checks)


if __name__ == "__main__":
    sys.exit(main())
