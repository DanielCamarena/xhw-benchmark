# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional OISST acquisition; default prints the plan without networking.

Reads NOAA OISST v2.1 AVHRR-only daily SST from the public Source Cooperative Icechunk
store, whose ``daily`` group references NOAA's original files on anonymous S3. Install
``examples/requirements-download.txt`` in a separate acquisition environment. No
credentials are needed.

Each store chunk holds one global day of about 0.67 MB (0.64-0.70 MB measured), and it is
transferred whole even for a small box: 1991-2025 moves about 8.5 GB and 1982-2025 about
10.8 GB. Several boxes (``--region peru equatorial``) are read in one pass, so the transfer
does not grow with the number of boxes.

Cycle (revision 3 of the development repository):

1. Plan one piece per year and box, identified by snapshot, box, box policy and dates. The
   snapshot is pinned (default: the one of the registered inputs).
2. Before reading ``sst``, check the snapshot's ``time`` and ``preliminary``: a year the
   snapshot does not cover, or one with preliminary days, stops the run before any transfer.
3. Read each month once for all boxes; a month with preliminary days is never saved.
   Errors are retried by icechunk ``kind`` (``_acquire.classify_error``), and the store is
   reopened inside the retry loop.
4. Validate every piece (all days, continuous, box coverage), write it atomically and record
   it in ``oisst_ledger.json``. A piece is reused only if the ledger holds the same request
   and its SHA-256 still matches.
5. Join the pieces of each box (one snapshot only), keep NOAA's attributes and add
   ``source``, ``store``, ``store_snapshot`` and ``history``, and write a manifest with the
   schema of ``examples/sources/oisst*.json`` plus ``content_sha256`` and versions. A product
   written without its manifest gets it on the next run if its SHA-256 matches the ledger.
   ``--no-join`` stops after step 4: the yearly pieces stay under ``oisst_partial/`` and no
   product or manifest is written. A later run without it joins them without reading again.

``--box-policy inside`` with the registered snapshot, region and period reproduces the
selection of the registered inputs. New files are not automatically equivalent to them:
``matches_registered`` in the manifest compares ``content_sha256``.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from _acquire import (
    DEFAULT_COMPRESSION,
    POLICIES,
    REGIONS,
    AcquireError,
    BlockWriter,
    Box,
    CoverageError,
    Ledger,
    RunSummary,
    ValidationError,
    backoff,
    build_manifest,
    call_with_retries,
    check_coverage,
    classify_error,
    compression,
    compression_arguments,
    discard_damaged,
    encoding_label,
    matches_registered,
    request_id,
    select_indices,
    select_lon_indices,
    utc_now,
    with_layout,
    write_manifest,
    write_netcdf_atomic,
)
from content_hash import registered

STORE = "https://data.source.coop/ocean-icechunks/noaa-oisst/oisst.icechunk"
GROUP = "daily"
REGISTERED_SNAPSHOT = registered("oisst")["version"].removeprefix("icechunk snapshot ")
SOURCE = "NOAA OISST v2.1 AVHRR-only daily (Huang et al. 2021)"
FIRST_DAY = date(1981, 9, 1)
MB_PER_DAY = 0.669  # mean size of one global sst chunk (revision 3, 10.3)
GRID_STEP = 0.25
#: The store's grid: cell centres, float32 (used to preview the selection offline).
OISST_LAT = np.arange(-89.875, 90, GRID_STEP, dtype=np.float32)
OISST_LON = np.arange(0.125, 360, GRID_STEP, dtype=np.float32)
#: Region -> registered manifest in examples/sources.
REGISTERED_KEYS = {"peru": "oisst", "equatorial": "oisst_equatorial", "npacific": "oisst_npacific"}
LEDGER = "oisst_ledger.json"
PARTIALS = "oisst_partial"


@dataclass(frozen=True)
class Target:
    """A box to cut, its label in file names and its registered manifest (if any)."""

    label: str
    box: Box
    key: str | None = None


@dataclass(frozen=True)
class Piece:
    """One year (or part of it) of one box."""

    target: Target
    start: date
    end: date
    id: str
    request: dict = field(compare=False)
    path: Path = field(compare=False)


@dataclass(frozen=True)
class Selection:
    """Positional selection of a box on the store's grid (longitudes west to east)."""

    lat: slice
    lon: tuple[slice, ...]

    def lon_index(self) -> np.ndarray:
        return np.concatenate([np.arange(s.start, s.stop) for s in self.lon])


@dataclass
class RunResult:
    exit_code: int
    summary: RunSummary | None
    products: dict[str, dict]
    problems: list[str]


# --- Plan -------------------------------------------------------------------------------


def make_targets(regions: Sequence[str] = (), boxes: Sequence[Sequence[float]] = ()) -> list:
    """Named regions and free boxes (``S N W E``), without repeats; ``peru`` if none."""
    targets: dict[str, Target] = {}
    for name in regions or ([] if boxes else ["peru"]):
        targets[name] = Target(name, REGIONS[name], REGISTERED_KEYS.get(name))
    for values in boxes:
        box = Box(*values)
        label = f"box_{request_id(box.as_dict(), length=8)}"
        targets[label] = Target(label, box)
    return list(targets.values())


def piece_request(target: Target, start: date, end: date, snapshot: str, policy: str) -> dict:
    return {
        "source": "oisst",
        "store": STORE,
        "group": GROUP,
        "snapshot": snapshot,
        "variable": "sst",
        "box": target.box.as_dict(),
        "box_policy": policy,
        "time": [start.isoformat(), end.isoformat()],
    }


def plan_pieces(
    targets: Sequence[Target], start: date, end: date, snapshot: str, policy: str, data_dir: Path
) -> list[Piece]:
    """One piece per calendar year (clipped to start..end) and target, years in order."""
    pieces = []
    for year in range(start.year, end.year + 1):
        first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        for target in targets:
            request = piece_request(target, first, last, snapshot, policy)
            piece_id = request_id(request, prefix=f"oisst_{target.label}_{year}")
            path = data_dir / PARTIALS / snapshot / target.label / f"{piece_id}.nc"
            pieces.append(Piece(target, first, last, piece_id, request, path))
    return pieces


def period_name(start: date, end: date) -> str:
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31):
        return f"{start.year}_{end.year}"
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


def product_name(target: Target, start: date, end: date, policy: str) -> str:
    """``oisst_v21_<label>[_<policy>]_<period>.nc``; ``inside`` keeps the registered name."""
    suffix = "" if policy == "inside" else f"_{policy.replace('-', '')}"
    return f"oisst_v21_{target.label}{suffix}_{period_name(start, end)}.nc"


def grid_step(coords) -> float:
    a = np.asarray(coords, dtype=float)
    return float(np.median(np.abs(np.diff(a)))) if a.size > 1 else GRID_STEP


def select_box(box: Box, lat, lon, policy: str) -> Selection:
    """Index ranges of ``box`` on the grid under ``policy`` (``CoverageError`` if impossible)."""
    lat_slice = select_indices(lat, box.south, box.north, policy, step=grid_step(lat))
    lon_slices = select_lon_indices(lon, box.west, box.east, policy, step=grid_step(lon))
    return Selection(lat_slice, tuple(lon_slices))


def output_longitudes(lon) -> tuple[np.ndarray, str]:
    """Shift per longitude and the convention of the saved file.

    -180..180 when the selected longitudes increase in it (the registered Peru and Nino 3.4
    inputs: shift -360), else 0..360 (a box crossing the antimeridian, as the registered
    North Pacific input: shift 0), else unwrapped eastward from the first longitude.
    """
    a = np.asarray(lon, dtype=float)
    for name, shifted in (
        ("-180..180", np.where(a >= 180, a - 360, a)),
        ("0..360", np.where(a < 0, a + 360, a)),
    ):
        if a.size < 2 or np.all(np.diff(shifted) > 0):
            return shifted - a, name
    return a[0] + np.mod(a - a[0], 360) - a, "unwrapped"


def describe_plan(
    targets: Sequence[Target], start: date, end: date, snapshot: str, policy: str
) -> dict:
    """What a download would do, computed offline on the store's grid."""
    days = (end - start).days + 1
    out = {
        "store": STORE,
        "group": GROUP,
        "snapshot": snapshot,
        "variable": "sst",
        "time": [start.isoformat(), end.isoformat()],
        "box_policy": policy,
        "pieces": len(targets) * (end.year - start.year + 1),
        "transfer_estimate_gb": round(days * MB_PER_DAY / 1000, 2),
        "transfer_note": "each day is one global chunk, read once for all boxes",
        "boxes": [],
    }
    for target in targets:
        entry = {"label": target.label, "registered": target.key, "box": target.box.as_dict()}
        try:
            sel = select_box(target.box, OISST_LAT, OISST_LON, policy)
        except (CoverageError, ValueError) as exc:
            entry["error"] = str(exc)
        else:
            lat, lon = OISST_LAT[sel.lat], OISST_LON[sel.lon_index()]
            shift, convention = output_longitudes(lon)
            saved = lon + shift
            entry["selection"] = {
                "lat": [float(lat[0]), float(lat[-1]), int(lat.size)],
                "lon": [float(saved[0]), float(saved[-1]), int(lon.size)],
                "lon_convention": convention,
            }
        entry["product"] = product_name(target, start, end, policy)
        out["boxes"].append(entry)
    return out


# --- Store and errors -------------------------------------------------------------------


def open_daily(snapshot: str):
    import icechunk

    repo = icechunk.Repository.open(icechunk.http_storage(STORE))
    auth = icechunk.containers_credentials(
        {p: icechunk.s3_anonymous_credentials() for p in repo.config.virtual_chunk_containers}
    )
    session = repo.reopen(authorize_virtual_chunk_access=auth).readonly_session(
        snapshot_id=snapshot
    )
    return xr.open_zarr(session.store, group=GROUP, consolidated=False, chunks=None)


class Store:
    """The pinned snapshot, opened on first use and again after a failed read."""

    def __init__(self, snapshot: str, opener: Callable[[str], xr.Dataset] | None = None):
        self.snapshot = snapshot
        self.opener = opener or open_daily
        self._ds = None

    @property
    def ds(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = self.opener(self.snapshot)
        return self._ds

    def reset(self) -> None:
        self._ds = None


def _wrapped_kind(exc: BaseException) -> str | None:
    """Classify an error raised ``from`` an icechunk error by that error's ``kind``."""
    if getattr(exc, "kind", None) is not None:
        return None
    cause, seen = exc.__cause__, set()
    while cause is not None and id(cause) not in seen:
        if getattr(cause, "kind", None) is not None:
            return classify_error(cause)
        seen.add(id(cause))
        cause = cause.__cause__
    return None


def classify(exc: BaseException) -> str:
    """``_acquire.classify_error``, also looking through wrapped icechunk errors."""
    return classify_error(exc, extra=[_wrapped_kind])


@dataclass
class Retry:
    """Retry policy: transient kinds up to ``retries`` times, ``virtual-chunk-fetch`` once."""

    retries: int = 3
    wait: Callable[[int], float] = backoff
    sleep: Callable[[float], None] = time.sleep


def with_retries(store: Store, fn: Callable[[xr.Dataset], object], retry: Retry, what: str):
    """``fn(store.ds)`` with retries; the store is reopened inside the loop (R7)."""

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        print(f"    {what}: retry {attempt} in {delay:.0f} s ({type(exc).__name__}: {exc})"[:160])
        store.reset()

    return call_with_retries(
        lambda: fn(store.ds),
        retries=retry.retries,
        classify=classify,
        wait=retry.wait,
        sleep=retry.sleep,
        on_retry=on_retry,
    )


def _days(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["time"].values).astype("datetime64[D]")


def check_snapshot(ds: xr.Dataset, pieces: Sequence[Piece], snapshot: str) -> dict[str, str]:
    """Problems per piece id found in the snapshot's ``time`` and ``preliminary``.

    Reads only ``preliminary`` (a few chunks); ``time`` is already in memory.
    """
    days = _days(ds)
    preliminary = np.asarray(ds["preliminary"].values) != 0
    problems: dict[str, str] = {}
    if days.size > 1 and np.any(np.diff(days) <= np.timedelta64(0, "D")):
        return {p.id: f"snapshot {snapshot}: time is not strictly increasing" for p in pieces}
    for piece in pieces:
        first, last = np.datetime64(piece.start, "D"), np.datetime64(piece.end, "D")
        expected = np.arange(first, last + np.timedelta64(1, "D"))
        missing = expected[~np.isin(expected, days)]
        inside = (days >= first) & (days <= last)
        months = sorted({str(d)[:7] for d in days[inside & preliminary]})
        if missing.size:
            problems[piece.id] = (
                f"snapshot {snapshot} lacks {missing.size} of {expected.size} days of "
                f"{piece.start}..{piece.end} (first {missing[0]}, last {missing[-1]})"
            )
        elif months:
            problems[piece.id] = (
                f"snapshot {snapshot} has preliminary days in {', '.join(months)}; this "
                "request uses final data only (use a later snapshot)"
            )
    return problems


# --- Read and validate ------------------------------------------------------------------


def _as_key(index: np.ndarray):
    """A slice for contiguous indices (what zarr reads best), else the index array."""
    index = np.asarray(index)
    if index.size and np.all(np.diff(index) == 1):
        return slice(int(index[0]), int(index[-1]) + 1)
    return index


def read_strip(ds: xr.Dataset, time_index, lat_index, lon_index) -> xr.Dataset:
    """Load ``sst`` and ``preliminary`` for some days over one latitude-longitude strip."""
    sub = ds[["sst", "preliminary"]].isel(time=time_index, lat=lat_index, lon=lon_index)
    if "zlev" in sub.dims:
        sub = sub.squeeze("zlev", drop=True)
    return sub.load()


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    out, day = [], start
    while day <= end:
        last = min(end, date(day.year, day.month, calendar.monthrange(day.year, day.month)[1]))
        out.append((day, last))
        day = last + timedelta(1)
    return out


def fetch_year(
    store: Store,
    pieces: Sequence[Piece],
    selections: dict[str, Selection],
    retry: Retry,
) -> dict[str, xr.Dataset]:
    """Read the pieces of one period (one year), month by month, once for all boxes.

    Each month is one read of the strip that holds every box; a month with preliminary
    days raises ``ValidationError`` before anything is kept (R3).
    """
    start, end = pieces[0].start, pieces[0].end
    sels = [selections[p.target.label] for p in pieces]
    lat_union = slice(min(s.lat.start for s in sels), max(s.lat.stop for s in sels))
    lon_union = np.unique(np.concatenate([s.lon_index() for s in sels]))
    lon_key = _as_key(lon_union)
    months: dict[str, list[xr.Dataset]] = {p.id: [] for p in pieces}
    for first, last in month_ranges(start, end):
        lo, hi = np.datetime64(first, "D"), np.datetime64(last, "D")

        def read(ds, lo=lo, hi=hi):
            days = _days(ds)
            index = np.flatnonzero((days >= lo) & (days <= hi))
            return read_strip(ds, _as_key(index), lat_union, lon_key)

        strip = with_retries(store, read, retry, f"{first:%Y-%m}")
        n = int(np.count_nonzero(np.asarray(strip["preliminary"].values)))
        if n:
            raise ValidationError(
                f"{first:%Y-%m}: {n} preliminary days in snapshot {store.snapshot}; this "
                f"request uses final data only, so nothing of {start.year} was saved"
            )
        strip = strip.drop_vars("preliminary")
        for piece, sel in zip(pieces, sels):
            lat_pos = slice(sel.lat.start - lat_union.start, sel.lat.stop - lat_union.start)
            lon_pos = _as_key(np.searchsorted(lon_union, sel.lon_index()))
            months[piece.id].append(strip.isel(lat=lat_pos, lon=lon_pos))
    return {key: xr.concat(parts, dim="time") for key, parts in months.items()}


def validate_piece(data: xr.Dataset, piece: Piece, policy: str) -> dict:
    """Checks of one piece: every day of its period once and in order, box coverage (R2)."""
    days = _days(data)
    expected = np.arange(
        np.datetime64(piece.start, "D"), np.datetime64(piece.end, "D") + np.timedelta64(1, "D")
    )
    if days.size != expected.size or np.any(days != expected):
        missing = np.setdiff1d(expected, days)
        extra = np.setdiff1d(days, expected)
        raise ValidationError(
            f"{piece.id}: {days.size} days instead of {expected.size} "
            f"(missing {missing.size}, unexpected {extra.size}, or out of order)"
        )
    if "preliminary" in data and int(np.count_nonzero(data["preliminary"].values)):
        raise ValidationError(f"{piece.id}: preliminary days")
    lat, lon = data["lat"].values, data["lon"].values
    try:
        coverage = check_coverage(
            piece.target.box, lat, lon, policy, lat_step=grid_step(lat), lon_step=grid_step(lon)
        )
    except CoverageError as exc:
        raise ValidationError(f"{piece.id}: {exc}") from exc
    return {
        "days": int(days.size),
        "first": str(days[0]),
        "last": str(days[-1]),
        "preliminary_days": 0,
        "sizes": {k: int(v) for k, v in data.sizes.items()},
        "snapshot": piece.request["snapshot"],
        "coverage": coverage,
    }


def _verify_written(path: Path, data: xr.Dataset) -> None:
    """Reread a written piece whole; an unreadable or different file is a local fault.

    Every variable and coordinate must decode to the values that were written (missing
    values in the same places), so a file that opens but holds other values is rejected.
    """
    with xr.open_dataset(path, engine="h5netcdf") as check:
        if dict(check.sizes) != dict(data.sizes):
            raise ValidationError(f"{path.name}: sizes {dict(check.sizes)} != {dict(data.sizes)}")
        for name, variable in data.variables.items():
            if name not in check.variables:
                raise ValidationError(f"{path.name}: {name} is missing")
            written, expected = np.asarray(check[name].values), np.asarray(variable.values)
            nan = expected.dtype.kind in "fcmM" and written.dtype.kind in "fcmM"
            if written.shape != expected.shape or not np.array_equal(
                written, expected, equal_nan=nan
            ):
                raise ValidationError(f"{path.name}: {name} differs from the data read")


def write_piece(data: xr.Dataset, piece: Piece, compress: int | None = DEFAULT_COMPRESSION) -> None:
    data = data.copy()
    data.attrs = {**data.attrs, "store": STORE, "store_snapshot": piece.request["snapshot"]}
    write_netcdf_atomic(with_layout(data, compress), piece.path)
    _verify_written(piece.path, data)


# --- Join and manifest ------------------------------------------------------------------


def _shift_lon(ds: xr.Dataset, shift: np.ndarray) -> xr.Dataset:
    if np.all(shift == shift[0]):
        return ds.assign_coords(lon=ds["lon"] + float(shift[0]))
    values = (ds["lon"].values + shift).astype(ds["lon"].dtype)
    return ds.assign_coords(lon=ds["lon"].copy(data=values))


def write_joined(
    ledger: Ledger,
    pieces: Sequence[Piece],
    snapshot: str,
    policy: str,
    retrieved: str,
    path: Path,
    compress: int | None = DEFAULT_COMPRESSION,
) -> dict:
    """Join the pieces of one box (one snapshot only) into ``path``, shift longitudes and
    add attributes.

    The pieces are written one at a time (``BlockWriter``), so memory holds one piece (one
    year, decoded as float64: North Pacific ~100 MB), not the whole series. The file has the
    ``content_sha256`` of one ``to_netcdf`` of the concatenated pieces; its attributes are
    those of the first piece, as that concatenation kept them. ``compress`` is zlib with byte
    shuffle (or None).
    """
    box = pieces[0].target.box
    first_day = last_day = None

    with BlockWriter(path, compress=compress) as writer:
        for piece in pieces:
            piece_path = ledger.resolve(ledger.get(piece.id)["piece"]["path"])
            with xr.open_dataset(piece_path, engine="h5netcdf") as opened:
                if opened.attrs.get("store_snapshot") != snapshot:
                    raise ValidationError(
                        f"{piece_path.name} comes from snapshot "
                        f"{opened.attrs.get('store_snapshot')}, not {snapshot}"
                    )
                part = opened.load()
            days = _days(part)
            steps = np.diff(days if last_day is None else np.concatenate([[last_day], days]))
            if days.size and np.any(steps != np.timedelta64(1, "D")):
                raise ValidationError("joined pieces are not one continuous daily series")
            if days.size:
                first_day = days[0] if first_day is None else first_day
                last_day = days[-1]
            if piece is pieces[0]:  # the other pieces share its grid (BlockWriter checks it)
                shift, convention = output_longitudes(part["lon"].values)
                lon_shift = float(shift[0]) if np.all(shift == shift[0]) else None
                lon_info = {"lon_convention": convention, "lon_shift_applied": lon_shift}
                noaa = dict(part.attrs)
            part = _shift_lon(part, shift)
            if piece is pieces[0]:
                lat, lon = part["lat"].values, part["lon"].values
                check_coverage(
                    box, lat, lon, policy, lat_step=grid_step(lat), lon_step=grid_step(lon)
                )
            # content_sha256 hashes ``units``: keep the spelling of the registered inputs.
            part["sst"].attrs = {**part["sst"].attrs, "units": "degC"}
            writer.append(part)
        original_source = noaa.get("source")
        entry = (
            f"{retrieved} download_oisst.py: sst of {STORE}#{GROUP} snapshot {snapshot}, "
            f"box S={box.south} N={box.north} W={box.west} E={box.east} ({policy}), "
            f"{first_day}..{last_day}, longitudes {convention}"
        )
        history = f"{noaa['history']}\n{entry}" if noaa.get("history") else entry
        source = (
            SOURCE
            if not original_source or original_source == SOURCE
            else f"{SOURCE}; input sources: {original_source}"
        )
        writer.update_attrs(
            {"source": source, "store": STORE, "store_snapshot": snapshot, "history": history}
        )
        writer.commit()
    return lon_info


def _fingerprint() -> dict | None:
    try:
        from provenance import code_fingerprint
    except ImportError:  # provenance.py needs xHW installed
        return None
    return code_fingerprint()


def product_manifest(
    ledger: Ledger,
    pieces: Sequence[Piece],
    path: Path,
    snapshot: str,
    policy: str,
    start: date,
    end: date,
) -> dict:
    """Manifest with the registered schema (``sources/oisst*.json``) plus the common fields."""
    target = pieces[0].target
    record = ledger.data["products"][path.name]
    with xr.open_dataset(path, engine="h5netcdf") as product:
        lat, lon = product["lat"].values, product["lon"].values
        sizes = {k: int(v) for k, v in product.sizes.items()}
    coverage = check_coverage(
        target.box, lat, lon, policy, lat_step=grid_step(lat), lon_step=grid_step(lon)
    )
    request = {
        **piece_request(target, start, end, snapshot, policy),
        "region": target.label,
        "output_filename": path.name,
    }
    west = target.box.west % 360
    ids = [p.id for p in pieces]
    manifest = build_manifest(
        source="oisst",
        request=request,
        files={path.name: path},
        coverage=coverage,
        retrieval_window=ledger.retrieval_window(ids),
        pieces=ids,
        dataset_id=f"{STORE}#{GROUP}",
        product="NOAA OISST v2.1 AVHRR-only daily, final (doi:10.1175/JCLI-D-20-0166.1); "
        "originals s3://noaa-cdr-sea-surface-temp-optimum-interpolation-pds",
        version=f"icechunk snapshot {snapshot}",
        variable="sst",
        query={
            "lat": [target.box.south, target.box.north],
            "lon_0_360": [west, west + target.box.width],
            "lon_shift_applied": record.get("lon_shift_applied"),
            "lon_convention": record.get("lon_convention"),
            "time": [start.isoformat(), end.isoformat()],
        },
        accessed=utc_now(),
        sizes=sizes,
        virtual_refs_checksummed=None,  # not measured: needs reading the store's manifests
        code_fingerprint=_fingerprint(),
    )
    file = manifest["files"][path.name]
    manifest.update(
        file=path.name,
        sha256=file["sha256"],
        content_sha256=file.get("content_sha256"),
        bytes=file["bytes"],
    )
    if target.key is not None:
        reference = registered(target.key)
        manifest["matches_registered"] = {
            **matches_registered(file, reference),
            "registered": f"examples/sources/{target.key}.json",
            "same_request": snapshot == reference["version"].removeprefix("icechunk snapshot ")
            and policy == "inside"
            and [start.isoformat(), end.isoformat()] == reference["query"]["time"],
        }
    else:
        manifest["matches_registered"] = None
    return manifest


def finish_product(
    ledger: Ledger,
    pieces: Sequence[Piece],
    data_dir: Path,
    snapshot: str,
    policy: str,
    start: date,
    end: date,
    compress: int | None = DEFAULT_COMPRESSION,
) -> dict:
    """Build the product of one box if needed and make sure it has a manifest (R6).

    The encoding (``compress``) is part of the product's identity: a product that differs
    only in it is rewritten in place (``Ledger.refresh_product``), with a new manifest.
    """
    target = pieces[0].target
    path = data_dir / product_name(target, start, end, policy)
    name = path.name
    inputs = {p.id: ledger.get(p.id)["piece"]["sha256"] for p in pieces}
    inputs["encoding"] = encoding_label(compress)
    status = ledger.refresh_product(name, path, inputs, compress)
    if status in ("changed", "stale", "unrecorded"):
        why = {
            "changed": "its SHA-256 differs from the one recorded in the ledger",
            "stale": "it was built from other pieces (another snapshot, box or period)",
            "unrecorded": "the ledger has no record of it",
        }[status]
        raise AcquireError(f"{name} exists and {why}; delete it or use another --data-dir")
    if status == "missing":
        for piece in pieces:
            if not ledger.reusable(piece.id):
                raise AcquireError(f"{piece.id}: its file is missing or changed; run again")
        lon_info = write_joined(ledger, pieces, snapshot, policy, utc_now(), path, compress)
        ledger.record_product(name, path, inputs, snapshot=snapshot, **lon_info)
    record = ledger.data["products"][name]
    manifest = record.get("manifest")
    if manifest is None or not ledger.resolve(manifest).exists():
        built = product_manifest(ledger, pieces, path, snapshot, policy, start, end)
        manifest_path = write_manifest(data_dir, f"oisst_{target.label}", built)
        record["manifest"] = ledger._rel(manifest_path)
        record["matches_registered"] = built["matches_registered"]
        ledger.save()
        print(f"{name}: written, manifest {manifest_path.name}")
    else:
        print(f"{name}: up to date (manifest {Path(manifest).name})")
    done = {"missing": "built", "reencoded": "reencoded"}.get(status, "current")
    return {**record, "status": done, "path": path}


# --- Run --------------------------------------------------------------------------------


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _set_concurrency(n: int) -> None:
    import zarr

    zarr.config.set({"async.concurrency": n})  # more simultaneous requests drop connections


def download(
    targets: Sequence[Target],
    start: date,
    end: date,
    *,
    data_dir: Path,
    snapshot: str = REGISTERED_SNAPSHOT,
    policy: str = "enclose",
    keep_failed: bool = False,
    concurrency: int | None = 8,
    retry: Retry | None = None,
    opener: Callable[[str], xr.Dataset] | None = None,
    compress: int | None = DEFAULT_COMPRESSION,
    join: bool = True,
) -> RunResult:
    """Download, validate and join every piece; never raises for a failed piece or product.

    ``join=False`` keeps the validated yearly pieces and writes no product or manifest.

    ``concurrency`` sets zarr's ``async.concurrency`` (None leaves it). ``opener`` replaces
    ``open_daily`` (tests). ``compress`` (zlib level with byte shuffle, or None) applies to
    pieces and products; validated pieces kept with another encoding are rewritten in place.
    """
    retry = retry or Retry()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(data_dir / LEDGER)
    pieces = plan_pieces(targets, start, end, snapshot, policy, data_dir)
    for piece in pieces:
        ledger.plan(piece.id, piece.request, piece.path)
    pending = [p for p in pieces if not ledger.reusable(p.id)]
    rewritten = [p.id for p in pieces if p not in pending and ledger.reencode_piece(p.id, compress)]
    if rewritten:
        print(f"{len(rewritten)} pieces rewritten with {encoding_label(compress)}")
    problems: list[str] = []
    if pending:
        problems = _read_pending(
            ledger, pending, snapshot, policy, keep_failed, concurrency, retry, opener, compress
        )
    ids = [p.id for p in pieces]
    summary = RunSummary.from_ledger(ledger, ids)
    summary.report()
    products: dict[str, dict] = {}
    for target in targets:
        own = [p for p in pieces if p.target == target]
        if any(ledger.get(p.id)["status"] != "validated" for p in own):
            continue
        if not join:
            print(f"{target.label}: {len(own)} yearly pieces in {own[0].path.parent}; not joined")
            continue
        try:
            products[target.label] = finish_product(
                ledger, own, data_dir, snapshot, policy, start, end, compress
            )
        except Exception as exc:
            problems.append(f"{target.label}: {_error(exc)}")
    for problem in problems:
        print(f"  {problem}")
    code = 1 if problems or summary.exit_code else 0
    return RunResult(code, summary, products, problems)


def _read_pending(
    ledger: Ledger,
    pending: Sequence[Piece],
    snapshot: str,
    policy: str,
    keep_failed: bool,
    concurrency: int | None,
    retry: Retry,
    opener,
    compress: int | None = DEFAULT_COMPRESSION,
) -> list[str]:
    """Plan checks, then read year by year; stops at the first failed year."""
    if concurrency is not None:
        _set_concurrency(concurrency)
    store = Store(snapshot, opener)

    def fail(group: Sequence[Piece], message: str) -> list[str]:
        for piece in group:
            ledger.update(piece.id, "failed", error=message)
        return [message]

    try:
        lat, lon = with_retries(
            store, lambda ds: (ds["lat"].values, ds["lon"].values), retry, "open"
        )
        selections = {}
        for piece in pending:
            label = piece.target.label
            if label not in selections:
                selections[label] = select_box(piece.target.box, lat, lon, policy)
        found = with_retries(store, lambda ds: check_snapshot(ds, pending, snapshot), retry, "plan")
    except Exception as exc:
        return fail(pending, f"before reading sst: {_error(exc)}")
    if found:
        for piece in pending:
            if piece.id in found:
                ledger.update(piece.id, "failed", error=found[piece.id])
        return [f"nothing transferred: {msg}" for msg in sorted(set(found.values()))]

    years: dict[date, list[Piece]] = {}
    for piece in pending:
        years.setdefault(piece.start, []).append(piece)
    for group in years.values():
        t0 = time.time()
        for piece in group:
            ledger.update(piece.id, attempts=ledger.get(piece.id)["attempts"] + 1, error=None)
        try:
            data = fetch_year(store, group, selections, retry)
        except Exception as exc:
            return fail(group, f"{group[0].start.year}: {_error(exc)}")
        stop = []
        for piece in group:
            ledger.update(piece.id, "downloaded")
            stop += _publish(
                ledger, piece, data[piece.id], policy, keep_failed, store, retry, selections,
                compress,
            )  # fmt: skip
        print(f"{group[0].start.year}: {time.time() - t0:.0f} s")
        if stop:
            return stop
    return []


def _publish(
    ledger: Ledger,
    piece: Piece,
    data: xr.Dataset,
    policy: str,
    keep_failed: bool,
    store: Store,
    retry: Retry,
    selections: dict[str, Selection],
    compress: int | None = DEFAULT_COMPRESSION,
) -> list[str]:
    """Validate and write one piece; a local write fault re-reads the piece once."""
    for attempt in (1, 2):
        try:
            checks = validate_piece(data, piece, policy)
        except ValidationError as exc:
            if keep_failed:
                quarantine = piece.path.parent / "quarantine" / piece.path.name
                write_netcdf_atomic(data, quarantine)
                ledger.update(
                    piece.id, "quarantined", error=_error(exc), quarantine=ledger._rel(quarantine)
                )
            else:
                ledger.update(piece.id, "failed", error=_error(exc))
            return [_error(exc)]
        try:
            write_piece(data, piece, compress)
        except Exception as exc:
            discard_damaged(ledger, piece.id, piece.path, exc)
            if attempt == 2:
                ledger.update(piece.id, "failed")
                return [f"{piece.id}: written file unreadable twice: {_error(exc)}"]
            try:
                data = fetch_year(store, [piece], selections, retry)[piece.id]
            except Exception as again:
                return [_fail_piece(ledger, piece, again)]
            continue
        ledger.record_piece(piece.id, piece.path, checks)
        return []
    return []  # unreachable


def _fail_piece(ledger: Ledger, piece: Piece, exc: BaseException) -> str:
    message = f"{piece.id}: {_error(exc)}"
    ledger.update(piece.id, "failed", error=message)
    return message


# --- Command line -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--region",
        nargs="+",
        choices=list(REGIONS),
        default=[],
        help="one or more named boxes, read in one pass (default peru if no --box)",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        action="append",
        default=[],
        metavar=("S", "N", "W", "E"),
        help="south north west east in degrees; repeat for several boxes",
    )
    parser.add_argument(
        "--box-policy",
        choices=POLICIES,
        default="enclose",
        help="enclose: nodes on or beyond every edge (default); cover-cells: cells reach "
        "every edge; inside: only nodes inside (the registered inputs)",
    )
    parser.add_argument("--start", type=date.fromisoformat, help="first day (default 1991-01-01)")
    parser.add_argument("--end", type=date.fromisoformat, help="last day (default 2025-12-31)")
    parser.add_argument("--years", help="inclusive years, for example 1991-2025 (or --start/--end)")
    parser.add_argument(
        "--snapshot",
        default=REGISTERED_SNAPSHOT,
        help=f"store snapshot (default {REGISTERED_SNAPSHOT}, the registered one)",
    )
    parser.add_argument("--download", action="store_true", help="transfer data (several GB)")
    parser.add_argument(
        "--no-join",
        action="store_true",
        help="with --download: keep the yearly pieces and write no joined product or manifest "
        "(run again without it to join them; nothing is transferred twice)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="zarr async.concurrency (default 8)"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="retries of transient errors (default 3)"
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="save a piece that fails validation to quarantine instead of discarding it",
    )
    compression_arguments(parser)
    return parser


def period_from_args(args: argparse.Namespace) -> tuple[date, date]:
    if args.years is not None:
        if args.start is not None or args.end is not None:
            raise ValueError("give --years or --start/--end, not both")
        try:
            first, last = map(int, args.years.split("-"))
        except ValueError:
            raise ValueError("--years must be an inclusive range such as 1991-2025") from None
        start, end = date(first, 1, 1), date(last, 12, 31)
    else:
        start = args.start or date(1991, 1, 1)
        end = args.end or date(2025, 12, 31)
    if start > end:
        raise ValueError(f"the period starts ({start}) after it ends ({end})")
    if start < FIRST_DAY:
        raise ValueError(f"OISST v2.1 starts on {FIRST_DAY}")
    return start, end


def main(argv: Sequence[str] | None = None, opener=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        start, end = period_from_args(args)
        targets = make_targets(args.region, args.box)
    except ValueError as exc:
        parser.error(str(exc))
    if args.concurrency < 1 or args.retries < 0:
        parser.error("--concurrency must be >= 1 and --retries >= 0")
    described = describe_plan(targets, start, end, args.snapshot, args.box_policy)
    print(json.dumps(described, indent=2))
    if not args.download:
        print(
            "Print only. Add --download to transfer data (about "
            f"{described['transfer_estimate_gb']} GB for {start}..{end})."
        )
        return 0
    if opener is None:
        try:
            import icechunk  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install examples/requirements-download.txt in an acquisition environment; "
                "see the README. It is not an xHW runtime dependency."
            ) from exc
    result = download(
        targets,
        start,
        end,
        data_dir=args.data_dir,
        snapshot=args.snapshot,
        policy=args.box_policy,
        keep_failed=args.keep_failed,
        concurrency=args.concurrency if opener is None else None,
        retry=Retry(retries=args.retries),
        opener=opener,
        compress=compression(args),
        join=not args.no_join,
    )
    for label, product in result.products.items():
        verdict = product.get("matches_registered")
        if verdict is None:
            print(f"{label}: no registered input to compare with.")
        elif verdict["content_sha256"]:
            print(f"{label}: same values as {verdict['registered']} (content_sha256).")
        else:
            print(
                f"{label}: differs from {verdict['registered']}; analyse it as a new run with "
                "its own SourceRef (the registered cases expect the registered checksum)."
            )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
