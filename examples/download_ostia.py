# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional OSTIA acquisition; default prints the plan without networking.

Retrieves the UK Met Office OSTIA daily sea surface temperature analysis (reprocessed,
0.05 degree, dataset version 202003; the series starts on 1981-10-01 and its end moves, so
it is asked from the service, never assumed) for a box, by default the 12S-4S x 82W-77W
box of the OISST tutorial, so that the effect of resolution can be studied. Install
copernicusmarine separately and run ``copernicusmarine login`` once.

    python examples/download_ostia.py --data-dir data                  # print the plan
    python examples/download_ostia.py --data-dir data --dry-run        # ask the service to check
    python examples/download_ostia.py --data-dir data --download       # one NetCDF per year

How it works (revision 5 of the development repository, decisions 1-2):

- **One request** for the whole period (``--block-days N`` splits it into fixed-length
  blocks), split locally into ``analysed_sst_<year>.nc``; a year cut by ``--start`` or
  ``--end`` is ``analysed_sst_<first>_<last>.nc`` (dates as YYYYMMDD).
- **Box.** ``--box-policy enclose`` (default) and ``cover-cells`` ask the client for
  ``coordinates_selection_method="outside"`` (the smallest grid containing the box);
  ``inside`` asks for ``inside``, as the registered files were requested. The received
  grid is checked against the policy and recorded in the manifest. Files go to
  ``<data-dir>/ostia_<region>`` for ``inside`` (the registered layout) and to
  ``<data-dir>/ostia_<region>_<policy>`` otherwise; a free ``--box`` uses a hash as region.
- **Antimeridian.** A box that crosses 180 degrees (``npacific``, or ``--box 0 10 150 -150``)
  is requested as two parts, west..179.975 and -179.975..east, so the client never receives
  a minimum longitude above the maximum (unverified) nor the seam itself. Each part is
  validated alone; the years join them west to east keeping the stored longitudes, and the
  joined grid must contain the whole box with no longitude missing or repeated at the seam.
- **Time.** The client applies one selection method to every axis, so ``outside`` cannot
  also reject a period the product only partly covers. Before downloading, each request is
  sent with ``coordinates_selection_method="strict-inside"`` and ``dry_run=True``: the
  client then raises ``CoordinatesOutOfDatasetBounds`` for a partly covered period (verified
  with copernicusmarine 2.5.0) and nothing is downloaded. The days received are checked
  again after downloading. ``--allow-partial`` accepts the edge years the product covers
  only in part (from before 1981-10-01, or beyond the current end) and saves them as
  ``..._partial.nc``, marked in the manifest; every later run asks for them again, so a
  grown series replaces the partial file by the complete year. It needs one request (no
  ``--block-days``).
- **Resumption** comes from ``ledger.json`` in the output folder, not from file existence
  (``skip_existing`` is never used): an interrupted request is repeated, a year already
  written from the same blocks is kept. Damaged files are retried once (``--keep-failed``
  moves them to quarantine instead). Blocks are deleted once every year is written, unless
  ``--keep-blocks``.
- **Manifest** ``ostia_manifest_<request hash>_<UTC>.json`` with the request, the fixed
  dataset version, the extent returned by the client, days per file, box coverage, hashes
  and software versions, compared with ``sources/ostia.json`` when that record exists.

``analysed_sst`` is in kelvin: xHW does not convert units, so declare it (or subtract
273.15 yourself) before building specs. Product page:
https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import _acquire as aq
import numpy as np
import xarray as xr
from content_hash import SOURCES

DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
DATASET_VERSION = "202003"  # the only version in the catalogue on 2026-11-03 (revision 5, 4.1)
VARIABLE = "analysed_sst"
FIRST_DAY = date(1981, 10, 1)  # start of the reprocessed series
GRID_STEP = 0.05  # cell centres at -89.975 + k * 0.05 and -179.975 + k * 0.05
GRID_OFFSET = 0.025
LON_FIRST, LON_LAST = -179.975, 179.975  # outermost longitude centres (the seam is 180)
LON_FIRST32, LON_LAST32 = (float(np.float32(v)) for v in (LON_FIRST, LON_LAST))
REGIONS = ("peru", "equatorial", "npacific")
SPACE_METHOD = {"enclose": "outside", "cover-cells": "outside", "inside": "inside"}
RAW = dict(decode_cf=False, mask_and_scale=False, decode_times=False, decode_coords=False)
#: copernicusmarine 2.5.0 request errors (revision 5, 4.3): permanent, never retried.
CLIENT_ERRORS = {
    "DatasetVersionNotFound": f"dataset version {DATASET_VERSION} is no longer in the "
    "catalogue; a new version is a new registered entry, not a silent replacement",
    "DatasetNotFound": f"dataset {DATASET_ID} is not in the catalogue",
    "VariableDoesNotExistInTheDataset": f"variable {VARIABLE} is not in the dataset",
    "CoordinatesOutOfDatasetBounds": "the request falls (partly) outside the product coverage",
}


def _names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def client_rule(exc: BaseException) -> str | None:
    """The request errors of revision 5 (4.3) are permanent; others follow ``_acquire``."""
    return aq.PERMANENT if _names(exc) & set(CLIENT_ERRORS) else None


def classify(exc: BaseException) -> str:
    return aq.classify_error(exc, extra=[client_rule])


def hint(exc: BaseException) -> str:
    """Error text, with what a known client error means for this script."""
    known = [CLIENT_ERRORS[n] for n in CLIENT_ERRORS if n in _names(exc)]
    text = f"{type(exc).__name__}: {exc}"
    return f"{text} ({known[0]})" if known else text


# --- Plan ----------------------------------------------------------------------------------


def spatial_request(box: aq.Box, policy: str) -> dict:
    """Box and selection method as the client receives them (longitudes in -180..180)."""
    return dict(
        minimum_latitude=box.south,
        maximum_latitude=box.north,
        minimum_longitude=aq._wrap(box.west, 180),
        maximum_longitude=aq._wrap(box.east, 180, upper=True),
        coordinates_selection_method=SPACE_METHOD[policy],
    )


def lon_parts(box: aq.Box, policy: str) -> list[aq.Box]:
    """The box, or its two parts west to east cut at the outermost centres of the grid.

    Every centre of a part lies in -179.975..179.975, so a seam edge at those centres is
    met by ``inside``, ``outside`` and ``strict-inside`` alike. With ``inside``, a part with
    no centre (a west edge beyond 179.975 or an east edge before -179.975) is left out.
    """
    if not box.crosses_antimeridian:
        return [box]
    west, east = aq.split_box(box)
    # inside keeps nodes <= the edge: ask for the float32 value of the outermost centre
    # (179.97500610...), which contains it whether the service stores float32 or float64.
    first, last = (LON_FIRST32, LON_LAST32) if policy == "inside" else (LON_FIRST, LON_LAST)
    parts = [
        aq.Box(box.south, box.north, min(west.west, last), last),
        aq.Box(box.south, box.north, first, max(east.east, first)),
    ]
    if policy == "inside":
        keep = (west.west <= last, east.east >= first)
        parts = [part for part, ok in zip(parts, keep) if ok]
    return parts


def check_packing(pieces: list[xr.Dataset], what: str) -> None:
    """Raw pieces joined as stored must share dtype and packing, or values would change."""
    keys = ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value", "units")
    packing = {
        tuple(
            str(p[VARIABLE].dtype) if k == "dtype" else repr(p[VARIABLE].attrs.get(k)) for k in keys
        )
        for p in pieces
    }
    if len(packing) != 1:
        raise aq.PermanentError(f"{what} store {VARIABLE} differently: {sorted(packing)}")


def part_check_box(part: aq.Box, policy: str) -> aq.Box:
    """The box a part is checked against: with ``inside`` its seam edge is the seam itself.

    Stored float32 centres (179.975006) lie just beyond the requested edge 179.975, which
    ``inside`` would reject; ``enclose`` and ``cover-cells`` keep the requested edges.
    """
    if policy != "inside":
        return part
    west = -180.0 if part.west <= LON_FIRST else part.west
    east = 180.0 if part.east >= LON_LAST else part.east
    return aq.Box(part.south, part.north, west, east)


def blocks(start: date, end: date, block_days: int | None) -> list[tuple[date, date]]:
    """The whole period, or consecutive blocks of ``block_days`` days from ``start``."""
    if block_days is None:
        return [(start, end)]
    out, first = [], start
    while first <= end:
        last = min(first + timedelta(block_days - 1), end)
        out.append((first, last))
        first = last + timedelta(1)
    return out


def spans(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar years of ``start..end``, the first and last cut by the period."""
    return [
        (max(start, date(y, 1, 1)), min(end, date(y, 12, 31)))
        for y in range(start.year, end.year + 1)
    ]


def file_name(first: date, last: date, partial: bool = False) -> str:
    whole = (first, last) == (date(first.year, 1, 1), date(first.year, 12, 31))
    stem = f"{VARIABLE}_{first.year}" if whole else f"{VARIABLE}_{first:%Y%m%d}_{last:%Y%m%d}"
    return f"{stem}{'_partial' if partial else ''}.nc"


def block_request(space: dict, first: date, last: date) -> dict:
    """Client arguments of one block, without the output location (the request identity)."""
    return dict(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        variables=[VARIABLE],
        **space,
        start_datetime=f"{first.isoformat()}T00:00:00",
        end_datetime=f"{last.isoformat()}T00:00:00",
    )


def folder_name(key: str, policy: str) -> str:
    """``ostia_<region>`` for ``inside`` (the registered layout), else with the policy."""
    return f"ostia_{key}" if policy == "inside" else f"ostia_{key}_{policy}"


def _extent(response) -> list[dict]:
    """``coordinates_extent`` of a client response as plain values (any format)."""
    return [
        {
            key: aq.as_json(getattr(item, key, None))
            for key in ("coordinate_id", "minimum", "maximum", "unit")
        }
        for item in getattr(response, "coordinates_extent", None) or []
    ]


# --- Validation ----------------------------------------------------------------------------


def _times(ds: xr.Dataset) -> np.ndarray:
    if VARIABLE not in ds or ds[VARIABLE].dims != ("time", "latitude", "longitude"):
        raise aq.ValidationError(f"{VARIABLE}(time, latitude, longitude) not in the file")
    times = ds["time"].values
    if times.dtype.kind != "M":
        raise aq.ValidationError("time is not decodable as dates")
    return times


def month_windows(first: date, last: date):
    """``first..last`` cut at calendar months."""
    a = first
    while a <= last:
        following = date(a.year + a.month // 12, a.month % 12 + 1, 1)
        b = min(last, following - timedelta(1))
        yield a, b
        a = b + timedelta(1)


def check_days(times: np.ndarray, first: date, last: date, *, lead: bool, tail: bool) -> dict:
    """Received days of ``first..last``: daily, continuous, complete.

    Days outside ``first..last`` are allowed (an ``outside`` selection may add one at each
    end) and are dropped later. Missing days are allowed only at the start (``lead``) or
    the end (``tail``) of the requested period, which is how the client cuts a period that
    the product covers in part; a gap anywhere else is an error.
    """
    if times.size == 0:
        raise aq.PermanentError(f"no day received for {first}..{last}")
    if np.any(np.diff(times) != np.timedelta64(1, "D")):
        raise aq.PermanentError("time is not daily and continuous (gap, repeat or other step)")
    days = times.astype("datetime64[D]")
    wanted = np.arange(np.datetime64(first, "D"), np.datetime64(last, "D") + np.timedelta64(1, "D"))
    inside = days[(days >= wanted[0]) & (days <= wanted[-1])]
    if inside.size == 0:
        raise aq.PermanentError(f"no day of {first}..{last} received")
    missing_lead = int((inside[0] - wanted[0]).astype(int))
    missing_tail = int((wanted[-1] - inside[-1]).astype(int))
    if (missing_lead and not lead) or (missing_tail and not tail):
        raise aq.PermanentError(
            f"received {inside[0]}..{inside[-1]} of the requested {first}..{last}: the product "
            "covers only part of it (--allow-partial keeps the edge years)"
        )
    return {
        "days": int(inside.size),
        "first_day": str(inside[0]),
        "last_day": str(inside[-1]),
        "missing_days": {"start": missing_lead, "end": missing_tail},
        "extra_days": int(days.size - inside.size),
    }


def check_grid(lat: np.ndarray, lon: np.ndarray, tol: float = 1e-4) -> None:
    """Regular 0.05 degree spacing on the OSTIA cell centres."""
    for name, values in (("latitude", lat), ("longitude", lon)):
        if values.size > 1 and not np.allclose(np.diff(values), GRID_STEP, atol=tol):
            raise aq.ValidationError(f"{name} is not a regular {GRID_STEP} degree grid")
        k = (values - GRID_OFFSET) / GRID_STEP
        if not np.allclose(k, np.round(k), atol=tol / GRID_STEP):
            raise aq.ValidationError(f"{name} values are not OSTIA cell centres")


# --- Client --------------------------------------------------------------------------------


def load_client():
    try:
        import copernicusmarine
    except ImportError as exc:
        raise ImportError(
            "Install copernicusmarine in an acquisition environment; "
            "see the README. It is not an xHW runtime dependency."
        ) from exc
    return copernicusmarine


def preflight(client, request: dict, allow_partial: bool) -> dict:
    """Ask the service with ``strict-inside`` (no data) whether it covers the whole block.

    Returns ``{"covered": True}`` or, for a partly covered block with ``allow_partial``,
    the extent the service would return with the box's own method. Raises
    ``PermanentError`` with that extent otherwise; other client errors propagate.
    """
    # Only time is gated here: the longitudes stay within the outermost centres, which are
    # inside the bounds whether the service stores them as float32 or float64.
    strict = dict(
        request,
        coordinates_selection_method="strict-inside",
        minimum_longitude=max(request["minimum_longitude"], LON_FIRST),
        maximum_longitude=min(request["maximum_longitude"], LON_LAST),
    )
    try:
        client.subset(**strict, dry_run=True, disable_progress_bar=True)
        return {"covered": True}
    except Exception as exc:
        if "CoordinatesOutOfDatasetBounds" not in _names(exc):
            raise
        error = f"{type(exc).__name__}: {exc}"
    try:
        extent = _extent(client.subset(**request, dry_run=True, disable_progress_bar=True))
    except Exception as exc:  # the period is entirely outside the coverage
        raise aq.PermanentError(f"{error}; {hint(exc)}") from exc
    time = next((e for e in extent if e["coordinate_id"] == "time"), {})
    period = f"{request['start_datetime'][:10]}..{request['end_datetime'][:10]}"
    if not allow_partial:
        raise aq.PermanentError(
            f"the product covers only {time.get('minimum')}..{time.get('maximum')} of the "
            f"requested {period}; change --start/--end, or add --allow-partial to keep the "
            "edge years as _partial files"
        )
    return {"covered": False, "strict_inside_error": error, "service_extent": extent}


# --- Command line --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, client=None) -> int:
    """Run the command line; ``client`` replaces ``copernicusmarine`` (tests)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    aq.add_box_arguments(parser, regions=REGIONS)
    aq.add_period_arguments(parser, "1991-01-01", "2025-12-31")
    parser.add_argument("--block-days", type=int, help="fixed-length blocks instead of one request")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="keep edge years the product covers only in part, as ..._partial.nc",
    )
    parser.add_argument(
        "--keep-blocks", action="store_true",
        help="keep the downloaded blocks after splitting (rewritten with --compress like the "
        "years; otherwise they are deleted as received)",
    )  # fmt: skip
    aq.compression_arguments(parser)
    aq.add_run_arguments(parser, workers=1)
    parser.set_defaults(retries=1)  # one global retry (revision 5, E1); network errors unverified
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        box, region = aq.box_from_args(args)
        start, end = aq.period_from_args(args, None if args.allow_partial else FIRST_DAY)
    except ValueError as exc:
        parser.error(str(exc))
    if box.width >= 360:
        parser.error("a global box is not supported; give a box narrower than 360 degrees")
    if args.block_days is not None and args.block_days < 1:
        parser.error("--block-days must be >= 1")
    if args.block_days is not None and args.allow_partial:
        # Blocks beyond the coverage, or cut inside it, cannot be told from damaged ones.
        parser.error("--allow-partial needs one request for the whole period (no --block-days)")
    if args.workers < 1 or args.retries < 0:
        parser.error("--workers must be >= 1 and --retries >= 0")

    key = region or "box_" + aq.request_id(box.as_dict(), length=8)
    target = args.data_dir / folder_name(key, args.box_policy)
    parts = lon_parts(box, args.box_policy)
    requests = {}
    for a, b in blocks(start, end, args.block_days):
        for k, part in enumerate(parts):
            req = block_request(spatial_request(part, args.box_policy), a, b)
            requests[aq.request_id(req, "block")] = (a, b, k, req)
    years = {file_name(a, b): (a, b) for a, b in spans(start, end)}
    run_request = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "variable": VARIABLE,
        "box": box.as_dict(),
        "region": region,
        "box_policy": args.box_policy,
        "coordinates_selection_method": SPACE_METHOD[args.box_policy],
        "longitude_parts": [part.as_dict() for part in parts],
        "period": [start.isoformat(), end.isoformat()],
        "block_days": args.block_days,
        "allow_partial": args.allow_partial,
    }
    print(json.dumps({"output": str(target), **run_request}, indent=2))
    for block_id, (_, _, _, req) in requests.items():
        print(f"{block_id}: {json.dumps(req)}")
    print(f"files: {', '.join(years)}")
    if not (args.download or args.dry_run):
        print("Print only. Add --dry-run to check the requests or --download to retrieve them.")
        return 0
    if args.dry_run:
        client = client or load_client()
        failed = False
        for block_id, (_, _, _, req) in requests.items():
            try:
                check = preflight(client, req, args.allow_partial)
                response = client.subset(**req, dry_run=True, disable_progress_bar=True)
                print(f"{block_id}: ok {json.dumps({**check, 'extent': _extent(response)})}")
            except Exception as exc:
                failed = True
                print(f"{block_id}: {hint(exc)}", file=sys.stderr)
        return 1 if failed else 0
    return Run(args, client, target, box, parts, requests, years, run_request).download()


# --- Download ------------------------------------------------------------------------------


class Run:
    """One ``--download``: blocks through the ledger, then years built from the blocks.

    A block is one period and one longitude part (two parts across the antimeridian).
    """

    def __init__(self, args, client, target, box, parts, requests, years, run_request):
        self.args, self.client, self.target, self.box = args, client, target, box
        self.parts, self.requests, self.years = parts, requests, years
        self.run_request = run_request
        self.compress = aq.compression(args)
        target.mkdir(parents=True, exist_ok=True)
        self.ledger = aq.Ledger(target / "ledger.json")
        for block_id, (a, b, k, req) in requests.items():
            part = f"_part{k + 1}" if len(parts) > 1 else ""
            name = f"{VARIABLE}_block_{a:%Y%m%d}_{b:%Y%m%d}{part}.nc"
            self.ledger.plan(block_id, req, target / "blocks" / name)
        self.first_day = min(a for a, _, _, _ in requests.values())
        self.last_day = max(b for _, b, _, _ in requests.values())
        self.grid_lock = threading.Lock()
        self.extents: dict[str, list] = {}

    # Products (one per year) depend on the SHA-256 of the blocks they were built from and
    # on their encoding.

    def blocks_of(self, name: str) -> list[str]:
        a, b = self.years[name]
        return [i for i, (c, d, _, _) in self.requests.items() if c <= b and a <= d]

    def inputs(self, name: str) -> dict:
        blocks = {
            i: (self.ledger.get(i).get("piece") or {}).get("sha256") for i in self.blocks_of(name)
        }
        return {**blocks, "encoding": aq.encoding_label(self.compress)}

    def path_of(self, name: str) -> Path:
        record = self.ledger.data["products"].get(name)
        return self.ledger.resolve(record["path"]) if record else self.target / name

    def current(self, name: str) -> bool:
        """Written from the current blocks and complete; a partial year is always redone.

        A year that differs only in its encoding is rewritten in place first, so it never
        needs its blocks (deleted after splitting) again.
        """
        path, inputs = self.path_of(name), self.inputs(name)
        status = self.ledger.refresh_product(name, path, inputs, self.compress)
        record = self.ledger.data["products"].get(name) or {}
        return status in aq.UP_TO_DATE and not record.get("checks", {}).get("partial")

    def edges(self, a: date, b: date) -> dict:
        allow = self.args.allow_partial
        return {"lead": allow and a == self.first_day, "tail": allow and b == self.last_day}

    def download(self) -> int:
        needed = sorted({i for n in self.years if not self.current(n) for i in self.blocks_of(n)})
        outcomes: dict[str, aq.Outcome] = {}
        if needed:
            self.client = self.client or load_client()

            def report(outcome: aq.Outcome) -> None:
                if outcome.ok:
                    entry = self.ledger.get(outcome.key)
                    error = f": {entry['error']}" if entry.get("error") else ""
                    print(f"{outcome.key}: {entry['status']}{error}", flush=True)
                else:
                    print(f"{outcome.key}: failed: {hint(outcome.error)}", flush=True)

            tasks = {i: (lambda i=i: self.fetch_block(i)) for i in needed}
            outcomes = aq.run_parallel(tasks, self.args.workers, report)
        summary = aq.RunSummary.from_ledger(self.ledger, list(self.requests), outcomes)
        statuses, errors = dict(summary.statuses), dict(summary.errors)
        if self.args.keep_blocks:  # blocks kept follow --compress; the others are deleted
            for i in self.requests:
                self.ledger.reencode_piece(i, self.compress)
        written = {}
        for name, (a, b) in self.years.items():
            try:
                if not self.current(name):
                    failed = [i for i in self.blocks_of(name) if summary.statuses[i] != "validated"]
                    if failed:
                        raise aq.AcquireError(f"blocks it needs are not validated: {failed}")
                    self.build_year(name, a, b)
                written[name] = self.path_of(name)
                statuses[name] = "validated"
            except Exception as exc:
                statuses[name], errors[name] = "failed", hint(exc)
        if not self.args.keep_blocks and len(written) == len(self.years):
            for i in self.requests:
                piece = self.ledger.get(i).get("piece")
                if piece:
                    self.ledger.resolve(piece["path"]).unlink(missing_ok=True)
        if written:
            print(f"manifest: {self.write_manifest(written)}")
        summary = aq.RunSummary(statuses, errors)
        summary.report()
        return summary.exit_code

    def fetch_block(self, block_id: str) -> dict:
        entry = self.ledger.get(block_id)
        if any((entry.get("checks") or {}).get("missing_days", {}).values()):
            # The series end moves under the same version (revision 5, R2b): ask again.
            self.ledger.update(block_id, "planned")
        elif self.ledger.reusable(block_id):
            return entry
        a, b, k, req = self.requests[block_id]
        check = preflight(self.client, req, self.args.allow_partial)
        final = self.ledger.resolve(self.ledger.get(block_id)["target"])

        def fetch(raw: Path) -> None:
            raw.unlink(missing_ok=True)  # never skip_existing: the ledger decides (R5)
            response = self.client.subset(
                **req,
                output_directory=str(raw.parent),
                output_filename=raw.name,
                disable_progress_bar=True,
            )
            self.extents[block_id] = _extent(response)

        def finish(raw: Path) -> tuple[Path, dict]:
            try:
                with xr.open_dataset(raw, engine="h5netcdf") as ds:
                    times = _times(ds)
                    lat = np.asarray(ds["latitude"].values, dtype=float)
                    lon = np.asarray(ds["longitude"].values, dtype=float)
            except aq.ValidationError:
                raise
            except Exception as exc:  # unreadable: a damaged file, retried once
                raise aq.ValidationError(f"cannot read the received file: {exc}") from exc
            days = check_days(times, a, b, **self.edges(a, b))
            check_grid(lat, lon)
            coverage = aq.check_coverage(  # this part; the joined box is checked per year
                part_check_box(self.parts[k], self.args.box_policy),
                lat,
                lon,
                self.args.box_policy,
                lat_step=GRID_STEP,
                lon_step=GRID_STEP,
            )
            self.check_reference_grid(req, lat, lon)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(raw, final)
            checks = {
                **days,
                "coverage": coverage,
                "client_extent": self.extents.get(block_id),
                "preflight": check,
            }
            return final, checks

        return aq.acquire(
            self.ledger,
            block_id,
            fetch=fetch,
            finish=finish,
            raw=self.target / "blocks" / ".tmp" / f"{block_id}.nc",
            keep_failed=self.args.keep_failed,
            quarantine_dir=self.target / "quarantine",
            retries=self.args.retries,
            classify=classify,
        )

    def check_reference_grid(self, req: dict, lat: np.ndarray, lon: np.ndarray) -> None:
        """Every block of a request must have the grid of its first validated block."""
        space_key = aq.request_sha256({k: v for k, v in req.items() if "datetime" not in k})
        with self.grid_lock:
            grids = self.ledger.data.setdefault("grids", {})
            ref = grids.get(space_key)
            if ref is None:
                grids[space_key] = {"latitude": lat.tolist(), "longitude": lon.tolist()}
                self.ledger.save()
                return
            for name, values in (("latitude", lat), ("longitude", lon)):
                expected = np.asarray(ref[name])
                if expected.shape != values.shape or not np.allclose(
                    expected, values, atol=aq.TOLERANCE
                ):
                    raise aq.ValidationError(f"{name} differs from the first block's grid")

    def joined_coverage(self, lat: np.ndarray, lon: np.ndarray) -> dict:
        """Coverage of the whole box by the joined parts, with a continuous seam.

        Longitudes are unwrapped from the first one, so the west part (up to 179.975) and
        the east part (from -179.975) must follow each other at exactly one grid step.
        """
        unwrapped = lon[0] + (lon - lon[0]) % 360
        if lon.size > 1 and not np.allclose(np.diff(unwrapped), GRID_STEP, atol=1e-4):
            raise aq.ValidationError("joined longitudes are not continuous (seam gap or repeat)")
        return aq.check_coverage(
            self.box, lat, lon, self.args.box_policy, lat_step=GRID_STEP, lon_step=GRID_STEP
        )

    def load_period(self, ids: list[str], a: date, b: date, *, empty: bool = False) -> xr.Dataset:
        """Days ``a..b`` of one block period, its longitude parts joined west to east.

        With ``empty`` no day is read: the result has the coordinates, attributes and
        packing of the join, at no memory cost.
        """
        parts = []
        for i in sorted(ids, key=lambda i: self.requests[i][2]):
            path = self.ledger.resolve(self.ledger.get(i)["piece"]["path"])
            with xr.open_dataset(path, engine="h5netcdf") as decoded:
                days = decoded["time"].values.astype("datetime64[D]")
            keep = np.flatnonzero((days >= np.datetime64(a, "D")) & (days <= np.datetime64(b, "D")))
            with xr.open_dataset(path, engine="h5netcdf", **RAW) as raw:
                parts.append(raw.isel(time=keep[:0] if empty else keep).load())
        for other in parts[1:]:
            same = (
                np.array_equal(parts[0]["time"].values, other["time"].values)
                and parts[0]["time"].attrs.get("units") == other["time"].attrs.get("units")
                and np.array_equal(parts[0]["latitude"].values, other["latitude"].values)
            )
            if not same:
                raise aq.PermanentError("the longitude parts differ in time or latitude")
        check_packing(parts, "longitude parts")
        if len(parts) == 1:
            return parts[0]
        return xr.concat(
            parts,
            dim="longitude",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        )

    @staticmethod
    def join_periods(pieces: list[xr.Dataset]) -> xr.Dataset:
        """The pieces of one year (one per block period) joined in time, values as stored."""
        encodings = {
            (p["time"].attrs.get("units"), p["time"].attrs.get("calendar")) for p in pieces
        }
        if len(encodings) != 1:  # raw times are joined as stored; another epoch would shift them
            raise aq.PermanentError(
                f"blocks encode time differently: {sorted(map(str, encodings))}"
            )
        check_packing(pieces, "blocks")
        if len(pieces) == 1:
            return pieces[0]
        return xr.concat(
            pieces,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        )

    def build_year(self, name: str, a: date, b: date) -> None:
        """Write one year (or part of one) from its blocks, with the stored values unchanged.

        The year is written month by month (``aq.BlockWriter``), so memory holds one month
        of the box. Its attributes, packing and checks come first from a join of the same
        blocks with no day read, so the file is the one a single join of the year gives.
        """
        periods: dict[tuple[date, date], list[str]] = {}
        for i in self.blocks_of(name):
            c, d, _, _ = self.requests[i]
            periods.setdefault((c, d), []).append(i)
        ordered = [ids for _, ids in sorted(periods.items())]
        template = self.join_periods([self.load_period(ids, a, b, empty=True) for ids in ordered])
        coverage = self.joined_coverage(
            np.asarray(template["latitude"].values, dtype=float),
            np.asarray(template["longitude"].values, dtype=float),
        )
        days = np.concatenate([self.period_days(ids, a, b) for ids in ordered])
        checks = check_days(days, a, b, **self.edges(a, b))
        partial = checks["days"] < (b - a).days + 1
        path = self.target / file_name(a, b, partial)
        old = self.ledger.data["products"].get(name)
        with aq.BlockWriter(path, compress=self.compress, raw=True) as writer:
            for c, d in month_windows(a, b):
                pieces = [self.load_period(ids, c, d) for ids in ordered]
                pieces = [p for p in pieces if p.sizes["time"]]
                if not pieces:
                    continue
                month = self.join_periods(pieces)
                month.attrs = dict(template.attrs)
                for k in month.variables:
                    month[k].attrs = dict(template[k].attrs)
                writer.append(month)
            writer.commit()
        if old and self.ledger.resolve(old["path"]).resolve() != path.resolve():
            self.ledger.resolve(old["path"]).unlink(missing_ok=True)
        self.ledger.record_product(
            name, path, self.inputs(name), checks={**checks, "partial": partial}, coverage=coverage
        )

    def period_days(self, ids: list[str], a: date, b: date) -> np.ndarray:
        """Decoded days ``a..b`` of one block period (from its first part)."""
        first = min(ids, key=lambda i: self.requests[i][2])
        path = self.ledger.resolve(self.ledger.get(first)["piece"]["path"])
        with xr.open_dataset(path, engine="h5netcdf") as decoded:
            times = decoded["time"].values
        days = times.astype("datetime64[D]")
        return times[(days >= np.datetime64(a, "D")) & (days <= np.datetime64(b, "D"))]

    def write_manifest(self, written: dict[str, Path]) -> Path:
        ledger, ids = self.ledger, list(self.requests)
        details = {path.name: ledger.data["products"][n]["checks"] for n, path in written.items()}
        block_checks = {i: ledger.get(i).get("checks") or {} for i in ids}
        coverage = ledger.data["products"][next(iter(written))].get("coverage")
        used = {i for name in written for i in self.blocks_of(name)}  # behind the files written
        part_coverage = [
            next(
                (
                    block_checks[i]["coverage"]
                    for i in sorted(used)
                    if self.requests[i][2] == k and block_checks[i].get("coverage")
                ),
                None,
            )
            for k in range(len(self.parts))
        ]
        manifest = aq.build_manifest(
            source="ostia",
            request=self.run_request,
            files={path.name: path for path in written.values()},
            coverage=coverage,
            retrieval_window=ledger.retrieval_window(ids),
            pieces=ids,
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            units="kelvin",
            days=details,
            partial_files=sorted(name for name, d in details.items() if d["partial"]),
            client_extent={i: c.get("client_extent") for i, c in block_checks.items()},
            requests={i: r[3] for i, r in self.requests.items()},
            part_coverage=part_coverage,
        )
        registered = SOURCES / "ostia.json"
        if registered.is_file():  # written by the titular from the downloaded files (decision 3)
            reference = json.loads(registered.read_text(encoding="utf-8")).get("files", {})
            manifest["registered_comparison"] = {
                name: aq.matches_registered(record, reference[name]) if name in reference else None
                for name, record in manifest["files"].items()
            }
        return aq.write_manifest(self.target, "ostia", manifest)


if __name__ == "__main__":
    sys.exit(main())
