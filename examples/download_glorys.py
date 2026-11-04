# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional GLORYS12V1 acquisition (Copernicus Marine); default prints the plan.

Without ``--download`` or ``--dry-run`` nothing touches the network. ``--dry-run`` asks the
service for sizes and the extent it would return; its ``data_transfer_size`` overstates the
real transfer about 12 times (revision 4, E3), so do not plan times or costs with it.
``--download`` fetches one request per calendar year (the block size is measured in F3,
revision 4 E1) into ``<data-dir>/glorys12v1_<region>/``:

- a ledger (``ledger.json``) records every year; rerunning or extending the period reuses
  the validated years whose file still has its recorded SHA-256;
- the client writes into ``.tmp/``; a file gets its final name only after its days, box
  coverage (``--box-policy``) and coordinate vectors are checked, so a cut never leaves a
  truncated file under a final name;
- a year the service covers only in part is rejected, unless ``--allow-partial`` saves it
  as ``thetao_<year>_partial.nc`` (fetched again on the next run);
- each run writes ``glorys_manifest_<request hash>_<UTC>.json`` and lists it in
  ``manifests.json``;
- a box that crosses the antimeridian (``--region npacific``, ``--box 25 55 150 -150``) is
  split in its two parts in -180..180 (revision 0, B5), each part is requested, validated
  and kept in ``parts/`` like a year of its own, and the year's file joins them west to
  east along longitude with the stored longitudes (for example 160..179.917 then
  -180..-130), streamed in time slabs and recorded as a ledger product of its two parts.

The defaults are the registered request: Peru box, 0-500 m, 1993-2023, ``thetao``, dataset
version 202311. ``--box-policy enclose`` (default) sends ``coordinates_selection_method=
"outside"``; ``inside`` is how the registered files were requested. Install
copernicusmarine separately and log in; it is not an xHW dependency. New files are not
automatically equivalent to the registered tutorial inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import _acquire as aq
from content_hash import content_sha256, file_sha256, registered

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DATASET_VERSION = "202311"  # revision 4, decision 1
SERVICE = "arco-time-series"
FIRST_DAY = date(1993, 1, 1)
DEFAULT_START, DEFAULT_END = "1993-01-01", "2023-12-31"
STEP = 1 / 12  # degrees, horizontal grid of GLORYS12V1
LAST_LON = 180 - STEP  # assumed last longitude node (-180 is stored, 180 is not); unverified
LAT_LON_TOL = 1e-6  # degrees (revision 4, section 7)
DEPTH_TOL = 1e-3  # metres
#: Box policy -> ``coordinates_selection_method`` (revision 0, B3). ``cover-cells`` sends a
#: box shrunk by half a step, so the minimal enclosing nodes are the minimal covering cells.
SELECTION = {"enclose": "outside", "cover-cells": "outside", "inside": "inside"}
JOIN_SLAB_BYTES = 256 << 20  # bytes of one variable read at a time while joining parts
_sleep = time.sleep  # tests replace it


class IncompleteCoverageError(aq.PermanentError):
    """The service returned fewer days than requested (the current year, for example)."""


@dataclass(frozen=True)
class Piece:
    """One request: a period of one calendar year (or one part of its box), its identity,
    its file and its grid family (the request without its period)."""

    start: date
    end: date
    id: str
    request: dict
    target: Path
    family: str

    @property
    def partial_target(self) -> Path:
        return partial_path(self.target)


@dataclass(frozen=True)
class Year:
    """The file of one period: one piece, or the parts of a box crossing the antimeridian."""

    start: date
    end: date
    name: str
    target: Path
    pieces: tuple[Piece, ...]

    @property
    def joined(self) -> bool:
        return len(self.pieces) > 1

    @property
    def partial_target(self) -> Path:
        return partial_path(self.target)


def partial_path(target: Path) -> Path:
    return target.with_name(f"{target.stem}_partial{target.suffix}")


def piece_periods(start: date, end: date) -> list[tuple[date, date]]:
    """One period per calendar year of ``start..end``.

    The block size (several years per request, calendar or aligned with the service's
    2081-day chunks) is chosen in F3 (revision 4, E1); a block would then be split into these
    yearly files when it is published.
    """
    return [
        (max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


def file_stem(variables: list[str], start: date, end: date) -> str:
    name = "-".join(variables)
    if (start, end) == (date(start.year, 1, 1), date(start.year, 12, 31)):
        return f"{name}_{start.year}"
    return f"{name}_{start:%Y%m%d}-{end:%Y%m%d}"


def client_box(box: aq.Box, policy: str) -> aq.Box:
    """The box sent to the client: ``box`` itself, or shrunk by half a step for cover-cells."""
    if policy != "cover-cells":
        return box
    half = STEP / 2

    def shrink(low: float, high: float) -> tuple[float, float]:
        if high - low < STEP:
            mid = (low + high) / 2
            return mid, mid
        return low + half, high - half

    south, north = shrink(box.south, box.north)
    width = box.width  # also for a box crossing the antimeridian (west > east)
    if width >= 360:
        return aq.Box(south, north, box.west, box.east)
    if width < STEP:
        west, width = box.west + width / 2, 0.0
    else:
        west, width = box.west + half, width - 2 * half
    # Back to -180..180 with the shrunk width: the box crosses 180 only if it still reaches it
    # (a box at 179.99..-179.8 may stop crossing once shrunk), so split_box sees valid edges.
    west = (west + 180.0) % 360.0 - 180.0
    east = west + width
    return aq.Box(south, north, west, east - 360.0 if east > 180.0 else east)


def request(
    start: date,
    end: date,
    box: aq.Box,
    policy: str,
    *,
    variables: list[str],
    max_depth: float,
    dataset_version: str,
    service: str,
) -> dict:
    """Arguments of ``copernicusmarine.subset`` except the output location (the identity)."""
    return _request(
        start,
        end,
        client_box(box, policy),
        policy,
        variables=variables,
        max_depth=max_depth,
        dataset_version=dataset_version,
        service=service,
    )


def requests(start: date, end: date, box: aq.Box, policy: str, **options) -> list[dict]:
    """One request per part of the box sent: two, west part first, if it crosses 180 deg.

    The parts come from ``split_box`` (west..180 and -180..east), so no request has
    ``minimum_longitude > maximum_longitude`` (whether the client accepts that is unverified).
    The grid is assumed to end at ``LAST_LON`` (179.917): under ``outside`` a box reaching
    beyond it, up to 180, needs the -180 node, which one request cannot wrap to, so it also
    gets an east part of that single node.
    """
    sent = client_box(box, policy)
    parts = aq.split_box(sent)
    if SELECTION[policy] == "outside" and len(parts) == 1 and sent.east > LAST_LON + LAT_LON_TOL:
        parts = [
            aq.Box(sent.south, sent.north, sent.west, 180.0),
            aq.Box(sent.south, sent.north, -180.0, -180.0),
        ]
    return [_request(start, end, part, policy, **options) for part in parts]


def _request(
    start: date,
    end: date,
    sent: aq.Box,
    policy: str,
    *,
    variables: list[str],
    max_depth: float,
    dataset_version: str,
    service: str,
) -> dict:
    if sent.crosses_antimeridian:
        raise ValueError("split the box with aq.split_box before requesting it")
    return {
        "dataset_id": DATASET_ID,
        "dataset_version": dataset_version,
        "variables": list(variables),
        "minimum_latitude": sent.south,
        "maximum_latitude": sent.north,
        "minimum_longitude": sent.west,
        "maximum_longitude": sent.east,
        "minimum_depth": 0.0,
        "maximum_depth": max_depth,
        "start_datetime": f"{start.isoformat()}T00:00:00",
        "end_datetime": f"{end.isoformat()}T00:00:00",
        "coordinates_selection_method": SELECTION[policy],
        "service": service,
    }


# --- Validation -------------------------------------------------------------------------


def response_summary(response) -> dict:
    """Status, sizes, service and ``coordinates_extent`` of a ``subset`` response."""
    out = {
        key: aq.as_json(getattr(response, key, None))
        for key in ("status", "message", "file_size", "data_transfer_size", "service")
    }
    extent = {}
    for item in getattr(response, "coordinates_extent", None) or []:
        name = getattr(item, "coordinate_id", None)
        if name:
            extent[name] = {
                "minimum": aq.as_json(getattr(item, "minimum", None)),
                "maximum": aq.as_json(getattr(item, "maximum", None)),
            }
    out["coordinates_extent"] = extent
    return out


def _extent_days(info: dict) -> tuple[date, date] | None:
    """First and last day of the time extent of a response, or None if not readable."""
    import pandas as pd

    time_extent = info.get("coordinates_extent", {}).get("time")
    if not time_extent:
        return None
    try:
        first, last = (pd.Timestamp(time_extent[k]) for k in ("minimum", "maximum"))
    except (TypeError, ValueError):
        return None
    return first.date(), last.date()


class GridReference:
    """Expected latitude, longitude and depth vectors of each request family.

    The first valid piece of a family (the request without its period) sets them; every
    later piece must match value by value (``grid_reference.json`` next to the ledger).
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def check(self, family: str, grid: dict, piece_id: str) -> str:
        """``set`` or ``matched``; raises ``ValidationError`` on a different grid."""
        import numpy as np

        with self._lock:
            data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            reference = data.get(family)
            if reference is None:
                data[family] = {**grid, "from_piece": piece_id, "set_utc": aq.utc_now()}
                aq.write_json_atomic(self.path, data)
                return "set"
        for axis, tol in (
            ("latitude", LAT_LON_TOL),
            ("longitude", LAT_LON_TOL),
            ("depth", DEPTH_TOL),
        ):
            got, want = np.asarray(grid[axis], float), np.asarray(reference[axis], float)
            if got.shape != want.shape or not np.allclose(got, want, rtol=0, atol=tol):
                raise aq.ValidationError(
                    f"{axis}: {got.size} values {got[:1]}..{got[-1:]} differ from the reference "
                    f"of {reference['from_piece']} ({want.size} values {want[:1]}..{want[-1:]})"
                )
        return "matched"


def validate(
    raw: Path,
    piece: Piece,
    *,
    box: aq.Box,
    policy: str,
    variables: list[str],
    info: dict,
    grids: GridReference,
    allow_partial: bool,
    check_box: bool = True,
) -> tuple[bool, dict]:
    """Check one received file; returns ``(partial, checks)``.

    Raises ``ValidationError`` for an unreadable file or one that does not match its request
    (variables without the time/depth/latitude/longitude dimensions, no depth levels, a time
    extent that contradicts the ``subset`` response, a grid unlike the reference),
    ``IncompleteCoverageError`` for fewer days than requested without ``allow_partial`` and
    ``CoverageError`` if the received nodes fail the box policy. A part of a box crossing the
    antimeridian passes ``check_box=False``: its seam edge (180 deg) need not be enclosed, and
    the whole box is checked on the joined longitudes instead (``join_parts``).
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    try:
        with xr.open_dataset(raw) as ds:
            missing = [v for v in variables if v not in ds.data_vars]
            sizes = {v: dict(ds[v].sizes) for v in variables if v in ds.data_vars}
            times = pd.DatetimeIndex(ds["time"].values)
            lat = np.asarray(ds["latitude"].values, dtype=float)
            lon = np.asarray(ds["longitude"].values, dtype=float)
            depth = np.asarray(ds["depth"].values, dtype=float)
    except Exception as exc:
        raise aq.ValidationError(f"{raw.name} cannot be read: {exc}") from exc
    if missing:
        raise aq.ValidationError(f"variables missing from the file: {', '.join(missing)}")
    if times.size == 0:
        raise aq.ValidationError("the file has no time steps")
    if depth.size == 0 or not np.all(np.isfinite(depth)) or np.any(depth < 0):
        raise aq.ValidationError(f"depth must be finite levels >= 0, got {depth.tolist()}")
    if np.any(np.diff(depth) <= 0):
        raise aq.ValidationError("depth levels are not strictly increasing")
    expected_sizes = {
        "time": times.size,
        "depth": depth.size,
        "latitude": lat.size,
        "longitude": lon.size,
    }
    for name, dims in sizes.items():
        if dims != expected_sizes:
            raise aq.ValidationError(f"{name} has dimensions {dims}, expected {expected_sizes}")
    days = times.floor("D")
    if np.any(np.diff(days.values) != np.timedelta64(1, "D")):
        raise aq.ValidationError("time is not a continuous daily series")
    first, last = days[0].date(), days[-1].date()
    if first < piece.start or last > piece.end:
        raise aq.ValidationError(f"days {first}..{last} fall outside {piece.start}..{piece.end}")
    response_days = _extent_days(info)
    if response_days is not None and response_days != (first, last):
        raise aq.ValidationError(
            f"the subset response announces {response_days[0]}..{response_days[1]} but the "
            f"file holds {first}..{last}"
        )
    expected = (piece.end - piece.start).days + 1
    partial = (first, last) != (piece.start, piece.end)
    if partial and not allow_partial:
        raise IncompleteCoverageError(
            f"the service returned {first}..{last} ({times.size} days) of "
            f"{piece.start}..{piece.end} ({expected} days); --allow-partial saves it as "
            f"{piece.partial_target.name}"
        )
    coverage = (
        aq.check_coverage(box, lat, lon, policy, lat_step=STEP, lon_step=STEP)
        if check_box
        else None
    )
    grid = {"latitude": lat.tolist(), "longitude": lon.tolist(), "depth": depth.tolist()}
    reference = grids.check(piece.family, grid, piece.id)
    return partial, {
        "days": int(times.size),
        "expected_days": expected,
        "first_day": first.isoformat(),
        "last_day": last.isoformat(),
        "partial": partial,
        "coverage": coverage,
        "grid": {
            "shape": [int(lat.size), int(lon.size), int(depth.size)],
            "depth_m": [float(depth[0]), float(depth[-1])],
            "reference": reference,
        },
        "response": info,
        "response_days_agree": None if response_days is None else True,
    }


# --- Join of the two parts of a box crossing the antimeridian ---------------------------

DECODING_ATTRS = ("scale_factor", "add_offset", "_FillValue", "missing_value", "units")


def _same_attr(a, b) -> bool:
    """Equal attribute values; two NaN fill values are equal."""
    import numpy as np

    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:  # strings and None
        return bool(np.array_equal(a, b))


def _seam(box: aq.Box, lons: list) -> tuple[list[slice], int]:
    """Slices of each part's longitudes that join west to east without a gap or a repeat.

    Longitudes are unwrapped around the centre of the box, so the stored values of both
    parts (160..179.917 and -180..-130) are compared directly. A node repeated at the seam
    (180 in the west part and -180 in the east part) is kept once; any other overlap, or a
    gap wider than 1.5 grid steps, is a ``ValidationError``.
    """
    import numpy as np

    centre = box.west + box.width / 2
    slices, dropped, previous = [], 0, None
    for i, lon in enumerate(lons):
        u = centre + ((np.asarray(lon, dtype=float) - centre + 180.0) % 360.0 - 180.0)
        if u.size == 0 or np.any(np.diff(u) <= 0):
            raise aq.ValidationError(f"longitudes of part {i} are empty or not increasing")
        start = 0
        if previous is not None:
            start = int(np.searchsorted(u, previous + LAT_LON_TOL, "right"))
            if start > 1 or (start == 1 and abs(u[0] - previous) > LAT_LON_TOL):
                raise aq.ValidationError(f"part {i} overlaps the previous part at the seam")
            if start >= u.size:
                raise aq.ValidationError(f"part {i} adds no longitude")
            gap = u[start] - previous
            if gap > 1.5 * STEP:
                raise aq.ValidationError(f"gap of {gap:.4f} deg at the seam before part {i}")
            dropped += start
        slices.append(slice(start, u.size))
        previous = u[-1]
    return slices, dropped


def join_parts(
    parts: list[Path], out: Path, *, box: aq.Box, policy: str, variables: list[str],
    compress: int | None = aq.DEFAULT_COMPRESSION,
) -> dict:  # fmt: skip
    """Join the parts along longitude into ``out`` (atomically) and return its checks.

    Values are copied as stored (packed integers stay packed, no CF decoding), one slab of
    days at a time (``JOIN_SLAB_BYTES`` per variable and part), so a large box never has to
    fit in memory; the variables are written with zlib ``compress`` and byte shuffle (or
    uncompressed), whatever the parts' compression. The parts must
    share days, depths, latitudes, dimension order, dtypes and the decoding attributes; the
    joined longitudes must pass ``check_coverage`` for the whole box under ``policy``.
    """
    import h5netcdf
    import numpy as np
    import xarray as xr

    opened = [xr.open_dataset(p, decode_cf=False) for p in parts]
    try:
        first = opened[0]
        for i, other in enumerate(opened[1:], start=1):
            for name, tol in (("time", 0.0), ("depth", DEPTH_TOL), ("latitude", LAT_LON_TOL)):
                a, b = first[name].values, other[name].values
                if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=tol):
                    raise aq.ValidationError(f"part {i} has other {name} values than part 0")
            if first["time"].attrs.get("units") != other["time"].attrs.get("units"):
                raise aq.ValidationError(f"part {i} has other time units than part 0")
            for v in variables:
                a, b = first[v], other[v]
                same = a.dims == b.dims and a.dtype == b.dtype
                if not same or any(
                    not _same_attr(a.attrs.get(k), b.attrs.get(k)) for k in DECODING_ATTRS
                ):
                    raise aq.ValidationError(f"{v}: part {i} is stored differently from part 0")
        keep, dropped = _seam(box, [ds["longitude"].values for ds in opened])
        lon = np.concatenate([ds["longitude"].values[k] for ds, k in zip(opened, keep)])
        lat = first["latitude"].values
        coverage = aq.check_coverage(box, lat, lon, policy, lat_step=STEP, lon_step=STEP)
        sizes = {
            "time": first.sizes["time"],
            "depth": first.sizes["depth"],
            "latitude": lat.size,
            "longitude": lon.size,
        }
        with aq.atomic_target(out) as tmp, h5netcdf.File(tmp, "w") as f:
            f.dimensions = sizes
            f.attrs.update(first.attrs)

            zlib = {"compression": "gzip", "compression_opts": compress, "shuffle": True}
            layout = {v: zlib for v in variables} if compress else {}

            def create(name, var, data=None):
                attrs = dict(var.attrs)
                fill = attrs.pop("_FillValue", None)
                created = f.create_variable(
                    name, var.dims, var.dtype, data=data, fillvalue=fill, **layout.get(name, {})
                )
                created.attrs.update(attrs)
                return created

            for name in ("time", "depth", "latitude"):
                create(name, first[name], first[name].values)
            create("longitude", first["longitude"], lon.astype(first["longitude"].dtype))
            for v in variables:
                var = first[v]
                target = create(v, var)
                axis_t, axis_x = var.dims.index("time"), var.dims.index("longitude")
                # Bytes of one day of the widest part: no part's slab exceeds the budget
                # (except that one day is the minimum slab).
                other_axes = int(np.prod(var.shape)) // (var.shape[axis_t] * var.shape[axis_x])
                widest = max(k.stop - k.start for k in keep)
                day_bytes = var.dtype.itemsize * other_axes * widest
                days = max(1, JOIN_SLAB_BYTES // max(day_bytes, 1))
                for t0 in range(0, sizes["time"], days):
                    t = slice(t0, min(t0 + days, sizes["time"]))
                    x0 = 0
                    for ds, k in zip(opened, keep):
                        block = ds[v].isel(time=t, longitude=k).values
                        index = [slice(None)] * var.ndim
                        index[axis_t] = t
                        index[axis_x] = slice(x0, x0 + block.shape[axis_x])
                        target[tuple(index)] = block
                        x0 += block.shape[axis_x]
    finally:
        for ds in opened:
            ds.close()
    return {
        "coverage": coverage,
        "longitude": {
            "parts": [
                [float(ds["longitude"].values[k][0]), float(ds["longitude"].values[k][-1])]
                for ds, k in zip(opened, keep)
            ],
            "seam_repeats_dropped": dropped,
            "n": int(lon.size),
        },
        "shape": [sizes["time"], sizes["depth"], sizes["latitude"], sizes["longitude"]],
    }


# --- Plan and run -----------------------------------------------------------------------


@dataclass
class Plan:
    key: str
    region: str | None
    box: aq.Box
    policy: str
    folder: Path
    run_request: dict
    years: list[Year]

    @property
    def pieces(self) -> list[Piece]:
        return [piece for year in self.years for piece in year.pieces]


def make_plan(args: argparse.Namespace, box: aq.Box, region: str | None) -> Plan:
    if region is not None:
        key = "glorys" if region == "peru" else f"glorys_{region}"
        folder = args.data_dir / f"glorys12v1_{region}"
    else:
        key = "glorys_box"
        folder = args.data_dir / f"glorys12v1_box_{aq.request_id(box.as_dict(), length=8)}"
    options = dict(
        variables=args.variables,
        max_depth=args.max_depth,
        dataset_version=args.dataset_version,
        service=args.service,
    )

    def family(req: dict) -> dict:
        return {k: v for k, v in req.items() if k not in ("start_datetime", "end_datetime")}

    years = []
    for start, end in piece_periods(args.start, args.end):
        reqs = requests(start, end, box, args.box_policy, **options)
        stem = file_stem(args.variables, start, end)
        target = folder / f"{stem}.nc"
        if len(reqs) == 1:
            targets = [target]
        else:
            targets = [folder / "parts" / f"{stem}_{side}.nc" for side in ("west", "east")]
        pieces = tuple(
            Piece(
                start,
                end,
                aq.request_id(req, prefix=part.stem),
                req,
                part,
                aq.request_id(family(req), prefix="grid"),
            )
            for req, part in zip(reqs, targets)
        )
        years.append(Year(start, end, stem, target, pieces))
    families = [family(p.request) for p in years[0].pieces]
    if len(families) == 1:
        run_request = dict(families[0])
    else:
        run_request = {k: v for k, v in families[0].items() if "longitude" not in k}
        run_request["longitude_parts"] = [
            [f["minimum_longitude"], f["maximum_longitude"]] for f in families
        ]
    run_request.update(
        start=args.start.isoformat(),
        end=args.end.isoformat(),
        requested_box=box.as_dict(),
        box_policy=args.box_policy,
        region=region,
    )
    return Plan(key, region, box, args.box_policy, folder, run_request, years)


def load_client():
    try:
        import copernicusmarine
    except ImportError as exc:
        raise SystemExit(
            "Install copernicusmarine in an acquisition environment and log in; see the "
            "README. It is not an xHW runtime dependency."
        ) from exc
    return copernicusmarine


def client_version(client) -> str | None:
    try:
        return version("copernicusmarine")
    except PackageNotFoundError:
        return getattr(client, "__version__", None)


def foreign_file(ledger: aq.Ledger, owner: str, path: Path) -> str | None:
    """Why an existing ``path`` must not be replaced by ``owner`` (a piece id or the name of
    a joined product), or None."""
    if not path.exists():
        return None
    owned = [(i, (e.get("piece") or {}).get("path")) for i, e in ledger.pieces.items()]
    owned += [(name, p.get("path")) for name, p in ledger.data["products"].items()]
    for other, stored in owned:
        if stored and ledger.resolve(stored).resolve() == path.resolve():
            if other == owner:
                return None
            return (
                f"{path.name} belongs to another request ({other}); "
                "use another --data-dir for this request"
            )
    return (
        f"{path.name} exists but this folder's ledger did not produce it; move it or "
        "use another --data-dir"
    )


def recover_download(ledger: aq.Ledger, piece: Piece, raw: Path) -> bool:
    """Put back in ``raw`` a download that an interrupted run left unrecorded.

    The ledger says ``downloaded`` with the raw file's SHA-256; the file is either still in
    ``.tmp/`` (cut before validation) or already under its final name (cut between the
    publication and the ledger record). Only a file with exactly that SHA-256 is taken back,
    and it is validated again like a new download.
    """
    entry = ledger.get(piece.id) or {}
    digest = (entry.get("raw") or {}).get("sha256")
    if entry.get("status") != "downloaded" or not digest:
        return False
    for path in (raw, piece.target, piece.partial_target):
        if path.exists() and file_sha256(path) == digest:
            if path != raw:
                raw.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, raw)
            return True
    return False


def run_piece(
    client, ledger: aq.Ledger, plan: Plan, piece: Piece, args, grids, check_box: bool = True
) -> dict:
    entry = ledger.plan(piece.id, piece.request, piece.target)
    if entry["status"] == "validated" and (entry.get("checks") or {}).get("partial"):
        reverted = [*entry.get("reverted", []), {"utc": aq.utc_now(), "reason": "partial year"}]
        ledger.update(piece.id, "planned", reverted=reverted)
    raw_path = plan.folder / ".tmp" / f"{piece.id}.nc"
    state = {"recovered": recover_download(ledger, piece, raw_path), "info": {}}
    for path in (piece.target, piece.partial_target):
        problem = foreign_file(ledger, piece.id, path)
        if problem:
            return ledger.update(piece.id, "failed", error=problem)

    def fetch(raw: Path) -> None:
        if state["recovered"]:  # validate the recovered download instead of fetching
            state.update(recovered=False, info={"recovered": "interrupted run"})
            return
        raw.unlink(missing_ok=True)
        response = client.subset(
            **piece.request,
            output_directory=str(raw.parent),
            output_filename=raw.name,
            dry_run=False,
        )
        state["info"] = response_summary(response)

    def finish(raw: Path) -> tuple[Path, dict]:
        info = state["info"]
        partial, checks = validate(
            raw,
            piece,
            box=plan.box,
            policy=plan.policy,
            variables=args.variables,
            info=info,
            grids=grids,
            allow_partial=args.allow_partial,
            check_box=check_box,
        )
        final = piece.partial_target if partial else piece.target
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(raw, final)
        if not partial:
            piece.partial_target.unlink(missing_ok=True)  # ours: foreign_file checked it
        return final, checks

    return aq.acquire(
        ledger,
        piece.id,
        fetch=fetch,
        finish=finish,
        raw=raw_path,
        keep_failed=args.keep_failed,
        quarantine_dir=plan.folder / "quarantine",
        retries=args.retries,
        sleep=lambda seconds: _sleep(seconds),
    )


def part_inputs(
    ledger: aq.Ledger, year: Year, compress: int | None = aq.DEFAULT_COMPRESSION
) -> dict[str, str] | None:
    """Identity of the join of ``year``: SHA-256 of each validated part and the encoding,
    or None if a part is not validated."""
    entries = [ledger.get(p.id) or {} for p in year.pieces]
    if any(e.get("status") != "validated" for e in entries):
        return None
    inputs = {p.id: e["piece"]["sha256"] for p, e in zip(year.pieces, entries)}
    return {**inputs, "encoding": aq.encoding_label(compress)}


def join_year(
    ledger: aq.Ledger, plan: Plan, year: Year, variables: list[str],
    compress: int | None = aq.DEFAULT_COMPRESSION,
) -> dict:  # fmt: skip
    """Join the validated parts of ``year`` (or reuse the current join, rewritten in place
    if only its encoding differs); raises on failure."""
    inputs = part_inputs(ledger, year, compress)
    if inputs is None:
        states = ", ".join(f"{p.target.name} {ledger.statuses([p.id])[p.id]}" for p in year.pieces)
        raise aq.AcquireError(f"{year.name}: cannot join, parts not validated ({states})")
    checks = [ledger.get(p.id)["checks"] for p in year.pieces]
    spans = {(c["first_day"], c["last_day"]) for c in checks}
    if len(spans) != 1:
        raise aq.ValidationError(f"{year.name}: the parts cover different days {sorted(spans)}")
    partial = checks[0]["partial"]
    final = year.partial_target if partial else year.target
    if ledger.refresh_product(year.name, final, inputs, compress) in aq.UP_TO_DATE:
        return ledger.data["products"][year.name]
    pending = JoinIntents(plan.folder / "joins_pending.json")
    ours = pending.get(year.name)  # a join published by an interrupted run, never recorded
    for path in (year.target, year.partial_target):
        if ours is not None and Path(ours).resolve() == path.resolve():
            continue
        problem = foreign_file(ledger, year.name, path)
        if problem:
            raise aq.PermanentError(problem)
    paths = [ledger.resolve(ledger.get(p.id)["piece"]["path"]) for p in year.pieces]
    pending.set(year.name, final)  # before publishing: a cut before record_product is ours
    joined = join_parts(
        paths, final, box=plan.box, policy=plan.policy, variables=variables, compress=compress
    )
    if not partial:
        year.partial_target.unlink(missing_ok=True)  # ours: foreign_file checked it
    days = {k: checks[0][k] for k in ("days", "expected_days", "first_day", "last_day")}
    record = ledger.record_product(
        year.name,
        final,
        inputs,
        bytes=final.stat().st_size,
        content_sha256=content_sha256(final),
        checks={
            **joined,
            **days,
            "partial": partial,
            "parts": {p.target.name: c for p, c in zip(year.pieces, checks)},
        },
    )
    pending.clear(year.name)
    return record


class JoinIntents:
    """Joined files about to be published (``joins_pending.json``), until they are recorded.

    A run cut between the publication of a join and ``Ledger.record_product`` leaves a file
    that no ledger record owns; its intent says this folder produced it, so the next run may
    rebuild it instead of refusing it as a file of unknown origin.
    """

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get(self, name: str) -> str | None:
        with self._lock:
            return self._read().get(name)

    def set(self, name: str, path: Path) -> None:
        with self._lock:
            aq.write_json_atomic(self.path, {**self._read(), name: str(path.resolve())})

    def clear(self, name: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(name, None) is not None:
                aq.write_json_atomic(self.path, data)


class RunManifest:
    """One manifest per run, written when the run starts and updated after every piece.

    File records come from the ledger (hashed when each piece was validated), so an update
    does not hash the files again.
    """

    def __init__(
        self, plan: Plan, ledger: aq.Ledger, version_of_client: str | None,
        compress: int | None = aq.DEFAULT_COMPRESSION,
    ):  # fmt: skip
        self.plan = plan
        self.ledger = ledger
        self.compress = compress
        self.client_version = version_of_client
        self.path: Path | None = None
        self.joined: set[str] = set()  # joins checked current (or built) in this run
        self._lock = threading.Lock()

    def build(self) -> dict:
        """Validated files of the run: single pieces, and joins whose inputs are the current
        validated parts (checked by their recorded SHA-256, without reading the file)."""
        plan, ledger = self.plan, self.ledger
        ids = [p.id for p in plan.pieces]
        done = [ledger.get(i) for i in ids if (ledger.get(i) or {}).get("status") == "validated"]
        files, checks, extents = {}, {}, {}
        for year in plan.years:
            if year.joined:
                product = ledger.data["products"].get(year.name)
                inputs = part_inputs(ledger, year, self.compress)
                if year.name not in self.joined or product is None or product["inputs"] != inputs:
                    continue
                name = Path(product["path"]).name
                files[name] = {k: product[k] for k in ("bytes", "sha256", "content_sha256")}
                checks[name] = product["checks"]
                extents[name] = {
                    part: c["response"].get("coordinates_extent")
                    for part, c in product["checks"]["parts"].items()
                }
                continue
            entry = ledger.get(year.pieces[0].id) or {}
            if entry.get("status") != "validated":
                continue
            record = dict(entry["piece"])
            name = Path(record.pop("path")).name
            files[name] = record
            checks[name] = entry["checks"]
            extents[name] = entry["checks"]["response"].get("coordinates_extent")
        manifest = aq.build_manifest(
            source=plan.key,
            request=plan.run_request,
            files={},
            coverage=checks[next(iter(checks))]["coverage"] if checks else None,
            retrieval_window=ledger.retrieval_window([e["id"] for e in done]),
            pieces=ids,
            dataset_id=DATASET_ID,
            dataset_version=plan.run_request["dataset_version"],
            service=plan.run_request["service"],
            requests=[
                {k: v for k, v in p.request.items() if "datetime" not in k}
                for p in plan.years[0].pieces
            ],
            client_version=self.client_version,
            statuses=ledger.statuses(ids),
            partial_files=sorted(name for name, c in checks.items() if c.get("partial")),
            coordinates_extent=extents,
            file_checks=checks,
            parts={
                year.target.name: [p.target.name for p in year.pieces]
                for year in plan.years
                if year.joined
            }
            or None,
        )
        manifest["files"] = files
        if plan.region == "peru":
            reference = registered("glorys")["files"]
            manifest["registered_comparison"] = {
                name: aq.matches_registered(record, reference[name])
                for name, record in files.items()
                if name in reference
            }
        return manifest

    def update(self, _outcome=None) -> Path:
        with self._lock:
            manifest = self.build()
            if self.path is None:
                self.path = aq.write_manifest(self.plan.folder, self.plan.key, manifest)
            else:
                aq.write_json_atomic(self.path, manifest)
            return self.path


def dry_run(client, plan: Plan) -> int:
    failures = 0
    for piece in plan.pieces:
        try:
            response = client.subset(
                **piece.request,
                output_directory=str(plan.folder),
                output_filename=piece.target.name,
                dry_run=True,
            )
        except Exception as exc:
            failures += 1
            print(f"{piece.target.name}: {type(exc).__name__}: {exc}")
            continue
        info = response_summary(response)
        print(
            f"{piece.target.name}: file {info['file_size']} MB, transfer estimate "
            f"{info['data_transfer_size']} MB (overstated ~12x), service {info['service']}"
        )
        days = _extent_days(info)
        if days is not None and days != (piece.start, piece.end):
            print(f"  partial: the service would return {days[0]}..{days[1]}")
        print(f"  extent: {json.dumps(info['coordinates_extent'])}")
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, aq.Box, str | None]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    aq.add_box_arguments(parser)
    aq.add_period_arguments(parser, DEFAULT_START, DEFAULT_END)
    parser.set_defaults(start=None, end=None)
    parser.add_argument("--years", help="A-B or A (inclusive); replaces --start/--end")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--variables", nargs="+", default=["thetao"])
    parser.add_argument("--max-depth", type=float, default=500.0, help="metres (default 500)")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--service", default=SERVICE)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="save a year the service covers only in part as <name>_partial.nc",
    )
    aq.add_run_arguments(parser, workers=1)
    aq.compression_arguments(parser)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.years is not None:
        if args.start is not None or args.end is not None:
            parser.error("give --years or --start/--end, not both")
        try:
            first, _, last = args.years.partition("-")
            args.start = date(int(first), 1, 1)
            args.end = date(int(last or first), 12, 31)
        except ValueError:
            parser.error("--years must be A-B or A, for example 1993-2023")
    args.start = args.start or date.fromisoformat(DEFAULT_START)
    args.end = args.end or date.fromisoformat(DEFAULT_END)
    try:
        aq.period_from_args(args, first=FIRST_DAY)
        box, region = aq.box_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    if args.workers < 1 or args.retries < 0:
        parser.error("--workers must be >= 1 and --retries >= 0")
    args.variables = list(dict.fromkeys(args.variables))
    return args, box, region


def main(argv: list[str] | None = None) -> int:
    args, box, region = parse_args(argv)
    plan = make_plan(args, box, region)
    if not (args.download or args.dry_run):
        print(f"# {len(plan.pieces)} requests into {plan.folder} (nothing downloaded)")
        for year in plan.years:
            for piece in year.pieces:
                joins = {"joins_into": year.target.name} if year.joined else {}
                line = {"file": piece.target.name, **joins, "id": piece.id, **piece.request}
                print(json.dumps(line))
        return 0
    client = load_client()
    if args.dry_run:
        return dry_run(client, plan)
    plan.folder.mkdir(parents=True, exist_ok=True)
    ledger = aq.Ledger(plan.folder / "ledger.json")
    grids = GridReference(plan.folder / "grid_reference.json")
    compress = aq.compression(args)
    manifest = RunManifest(plan, ledger, client_version(client), compress)

    def task(year: Year):
        try:
            for piece in year.pieces:  # every part is tried, even if another one failed
                run_piece(client, ledger, plan, piece, args, grids, check_box=not year.joined)
                # received uncompressed: kept (and reused) with the encoding asked
                ledger.reencode_piece(piece.id, compress)
            if year.joined:
                product = join_year(ledger, plan, year, args.variables, compress)
                manifest.joined.add(year.name)
                return product
            return ledger.get(year.pieces[0].id)
        finally:
            manifest.update()  # in the worker, before it takes the next year

    def key(year: Year) -> str:
        return f"{year.target.name} (joined)" if year.joined else year.pieces[0].id

    tasks = {key(year): (lambda year=year: task(year)) for year in plan.years}
    manifest.update()
    outcomes = None
    try:
        outcomes = aq.run_parallel(tasks, args.workers)
    finally:
        print(f"manifest: {manifest.update()}")
        summary = aq.RunSummary.from_ledger(
            ledger, [p.id for p in plan.pieces], outcomes and {k: outcomes[k] for k in outcomes}
        )
        for year in plan.years:
            if year.joined:
                outcome = (outcomes or {}).get(key(year))
                if outcome is not None and outcome.ok:
                    summary.statuses[key(year)] = "validated"
                else:
                    summary.statuses[key(year)] = "failed"
                    summary.errors[key(year)] = (
                        f"{type(outcome.error).__name__}: {outcome.error}"
                        if outcome is not None
                        else "run interrupted before the join"
                    )
        summary.report()
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
