# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Shared acquisition machinery for the ``download_*.py`` scripts (no network, no clients).

Seven pieces, each usable on its own:

1. **Box coverage.** The downloaded data must contain the requested box. ``enclose`` (the
   default) asks for nodes on or beyond every edge; ``cover-cells`` for cells (node +/- half
   a step) reaching every edge; ``inside`` keeps only the nodes inside the box, which is the
   historical behaviour used to regenerate the registered inputs. ``cds_area`` rounds a CDS
   ``area`` outward to the grid, ``select_indices``/``select_lon_indices`` pick the smallest
   index range of a local coordinate vector, ``split_box`` cuts a box crossing the
   antimeridian, and ``check_coverage`` verifies the extent actually received.
2. **Ledger.** One JSON entry per piece, identified by a hash of the normalised request, with
   a status (planned, downloaded, validated, failed, quarantined). A piece is reused only if
   it is validated and its file still has the recorded SHA-256.
3. **Atomic writes** (``.tmp`` then ``os.replace``) and provenance sidecars.
4. **Damaged files.** Size, SHA-256 and error are recorded before a damaged raw file is
   deleted and retried once; with ``keep_failed`` it is moved to quarantine instead.
5. **Manifest** with request, hashes, software versions, retrieval window and box coverage,
   named after the request and written atomically, plus an index of all manifests.
6. **Parallel runs** that collect every error (``as_completed``), and an error classifier
   (transient, retry once, permanent) that each source can extend.
7. **CDS cost planner:** ``cost = 6 x variables x levels x hours x real days``, limits
   121 000 (single levels) and 60 000 (pressure levels), with a configurable margin.

Download clients (cdsapi, copernicusmarine, icechunk, fsspec) are never imported here: errors
are classified by class name and attributes. Design follows ``docs/revisiones`` 0-6 of the
development repository; the ledger, sidecar and bounded pool also follow bonino2024xt
(rewritten, not copied).
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from content_hash import content_sha256, file_sha256

POLICIES = ("enclose", "cover-cells", "inside")
TOLERANCE = 1e-6  # degrees; coordinates are often stored as float32


class AcquireError(Exception):
    """Base class of the errors raised by this module."""


class CoverageError(AcquireError):
    """The data do not, or cannot, contain the requested box under the chosen policy."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


class ValidationError(AcquireError):
    """A received file is unreadable or does not match its request (a damaged file)."""


class PermanentError(AcquireError):
    """An error that retrying cannot fix; sources raise it to skip the retries."""


class CostLimitError(PermanentError):
    """A CDS request would exceed the cost limit (the CDS answers 403 and never queues it)."""


class LedgerError(AcquireError):
    """The ledger is inconsistent with the plan (for example, two requests share an id)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- 1. Box coverage -------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Latitudes south..north and longitudes west..east in degrees, in any convention.

    A box whose west edge lies east of its east edge once both are put in -180..180 (for
    example ``west=150, east=-150``) crosses the antimeridian.
    """

    south: float
    north: float
    west: float
    east: float

    def __post_init__(self):
        if not -90 <= self.south <= self.north <= 90:
            raise ValueError(f"need -90 <= south <= north <= 90, got {self.south}, {self.north}")
        if not all(math.isfinite(v) for v in (self.west, self.east)):
            raise ValueError("longitudes must be finite")

    @property
    def width(self) -> float:
        """Longitudinal width in degrees, 0 < width <= 360 (0 only if west == east)."""
        if self.west == self.east:
            return 0.0
        width = (self.east - self.west) % 360
        return 360.0 if width < TOLERANCE else width

    @property
    def crosses_antimeridian(self) -> bool:
        return 0 < self.width < 360 and _wrap(self.west, 180) > _wrap(self.east, 180, upper=True)

    def as_dict(self) -> dict:
        return {"south": self.south, "north": self.north, "west": self.west, "east": self.east}


def _wrap(lon: float, convention: int, upper: bool = False) -> float:
    """``lon`` in [-180, 180) or [0, 360); with ``upper`` in (-180, 180] or (0, 360]."""
    base = -180.0 if convention == 180 else 0.0
    if upper:
        return base + 360.0 - ((base + 360.0 - lon) % 360.0)
    return base + ((lon - base) % 360.0)


def lon_convention(lons) -> int:
    """180 for longitudes in -180..180, 360 for 0..360 (decided by the largest value)."""
    return 360 if float(np.nanmax(np.asarray(lons, dtype=float))) > 180 + TOLERANCE else 180


def split_box(box: Box) -> list[Box]:
    """The box itself, or its two parts in -180..180 if it crosses the antimeridian."""
    if not box.crosses_antimeridian:
        return [box]
    west, east = _wrap(box.west, 180), _wrap(box.east, 180, upper=True)
    return [Box(box.south, box.north, west, 180.0), Box(box.south, box.north, -180.0, east)]


def _snap(value: float, step: float, origin: float, rounding) -> float:
    """``rounding`` of ``value`` to the grid ``origin + k * step``, ignoring float noise."""
    q = (value - origin) / step
    if abs(q - round(q)) < 1e-6:
        q = round(q)
    return round(origin + rounding(q) * step, 10)


def cds_area(box: Box, policy: str = "enclose", step: float = 0.25, origin: float = 0.0) -> list:
    """CDS ``area`` ``[north, west, south, east]`` for ``box`` under ``policy``.

    The CDS returns only the grid nodes inside ``area``, so ``enclose`` rounds every edge
    outward to the grid (``S = floor(S / step) * step``, ``N = ceil(N / step) * step``...) and
    ``cover-cells`` to the nearest node whose half-step cell still reaches the edge.
    ``inside`` returns the box unchanged, as the registered requests were sent. A box that
    crosses the antimeridian must be split first (``split_box``).
    """
    _check_policy(policy)
    if box.crosses_antimeridian:
        raise ValueError("the box crosses the antimeridian; request each part of split_box()")
    s, n, w, e = box.south, box.north, box.west, box.east
    if policy == "enclose":
        s, w = (_snap(v, step, origin, math.floor) for v in (s, w))
        n, e = (_snap(v, step, origin, math.ceil) for v in (n, e))
    elif policy == "cover-cells":
        half = step / 2
        s, w = (_snap(v + half, step, origin, math.floor) for v in (s, w))
        n, e = (_snap(v - half, step, origin, math.ceil) for v in (n, e))
    return [min(n, 90.0), w, max(s, -90.0), e]


def _check_policy(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"box policy must be one of {', '.join(POLICIES)}; got {policy!r}")


def cell_edges(coords, step: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper cell edges of ascending ``coords``: midpoints, half a step at the ends.

    A single node needs ``step``; with ``step`` every cell is node +/- step / 2.
    """
    a = np.asarray(coords, dtype=float)
    if step is not None:
        return a - step / 2, a + step / 2
    if a.size < 2:
        raise CoverageError("one coordinate value: pass the grid step to know its cell")
    mids = (a[1:] + a[:-1]) / 2
    lower = np.concatenate([[a[0] - (a[1] - a[0]) / 2], mids])
    upper = np.concatenate([mids, [a[-1] + (a[-1] - a[-2]) / 2]])
    return lower, upper


def select_indices(
    coords,
    low: float | None,
    high: float | None,
    policy: str = "enclose",
    *,
    step: float | None = None,
    tol: float = TOLERANCE,
) -> slice:
    """Smallest positional slice of a monotonic 1-D ``coords`` that meets ``policy``.

    ``low``/``high`` are the box edges in the units and convention of ``coords``; ``None``
    takes everything to that end of the vector. Works for ascending and descending vectors
    (the slice is positional, so it keeps the stored order). Raises ``CoverageError`` if the
    vector cannot satisfy the policy, instead of silently returning less.
    """
    _check_policy(policy)
    c = np.asarray(coords, dtype=float)
    if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
        raise CoverageError("coordinates must be a non-empty, finite 1-D vector")
    if low is not None and high is not None and low > high:
        raise ValueError(f"low edge {low} is above high edge {high}")
    descending = c.size > 1 and c[0] > c[-1]
    a = c[::-1] if descending else c
    if np.any(np.diff(a) <= 0):
        raise CoverageError("coordinates are not strictly monotonic")
    last = a.size - 1
    if policy == "inside":
        first_i = 0 if low is None else int(np.searchsorted(a, low - tol, "left"))
        last_i = last if high is None else int(np.searchsorted(a, high + tol, "right")) - 1
    else:
        lower, upper = (a, a) if policy == "enclose" else cell_edges(a, step)
        first_i = 0 if low is None else int(np.searchsorted(lower, low + tol, "right")) - 1
        last_i = last if high is None else int(np.searchsorted(upper, high - tol, "left"))
        what = "node" if policy == "enclose" else "cell"
        if first_i < 0:
            raise CoverageError(f"no {what} reaches {low} (data start at {a[0]})")
        if last_i > last:
            raise CoverageError(f"no {what} reaches {high} (data end at {a[-1]})")
    if first_i > last_i:
        raise CoverageError(f"no coordinate inside {low}..{high}")
    if descending:
        return slice(last - last_i, last - first_i + 1)
    return slice(first_i, last_i + 1)


def select_lon_indices(
    lons,
    west: float,
    east: float,
    policy: str = "enclose",
    *,
    step: float | None = None,
    tol: float = TOLERANCE,
) -> list[slice]:
    """Slices of ``lons`` (-180..180 or 0..360) covering west..east, in west-to-east order.

    The box edges may be in either convention. When the box crosses the seam of the data's
    convention (the antimeridian for -180..180, Greenwich for 0..360) the result has two
    slices, the western part first; concatenating them gives a continuous west-to-east
    range. On a global ascending grid the vector is treated as periodic, so the policy can
    take nodes from the other side of the seam (a box starting at 0.01 deg on a 0.125..359.875
    grid gets 359.875 too). On a regional grid only the outer edges follow the policy.
    """
    box = Box(-90, 90, west, east)
    a = np.asarray(lons, dtype=float)
    if box.width >= 360 - tol:
        return [slice(0, a.size)]
    convention = lon_convention(a)
    w = _wrap(west, convention)
    e = _wrap(east, convention, upper=True)
    if a.size > 1 and a[0] < a[-1] and a[-1] - a[0] + np.median(np.diff(a)) >= 360 - tol:
        # A grid that repeats the seam node (0 and 360) is periodic without its last value.
        n = a.size - 1 if a[-1] - a[0] >= 360 - tol else a.size
        extended = np.concatenate([a[:n] - 360, a[:n], a[:n] + 360])
        picked = select_indices(extended, w, w + box.width, policy, step=step, tol=tol)
        index = np.arange(picked.start, picked.stop) % n
        if index.size >= n:
            return [slice(0, n)]
        breaks = np.flatnonzero(np.diff(index) != 1) + 1
        return [slice(int(run[0]), int(run[-1]) + 1) for run in np.split(index, breaks)]
    if w <= e:
        return [select_indices(lons, w, e, policy, step=step, tol=tol)]
    return [
        select_indices(lons, w, None, policy, step=step, tol=tol),
        select_indices(lons, None, e, policy, step=step, tol=tol),
    ]


def _axis_report(
    values, low: float, high: float, policy: str, step: float | None, tol: float
) -> tuple[dict, list[str]]:
    a = np.sort(np.asarray(values, dtype=float).ravel())
    if a.size == 0:
        return {"nodes": None, "cells": None, "n": 0}, ["no coordinate values"]
    lower, upper = cell_edges(a, step) if (step is not None or a.size > 1) else (a, a)
    extent = {
        "nodes": [float(a[0]), float(a[-1])],
        "cells": [float(lower[0]), float(upper[-1])],
        "n": int(a.size),
    }
    problems = []
    if policy == "enclose" and not (a[0] <= low + tol and a[-1] >= high - tol):
        problems.append(f"nodes {a[0]}..{a[-1]} do not enclose {low}..{high}")
    if policy == "cover-cells" and not (lower[0] <= low + tol and upper[-1] >= high - tol):
        problems.append(f"cells {lower[0]}..{upper[-1]} do not cover {low}..{high}")
    if policy == "inside" and not (a[0] >= low - tol and a[-1] <= high + tol):
        problems.append(f"nodes {a[0]}..{a[-1]} fall outside {low}..{high}")
    if a.size > 1:
        spacing = np.diff(a)
        expected = step if step is not None else np.median(spacing)
        if (step is not None or a.size > 2) and np.any(spacing > 1.5 * expected):
            problems.append("coordinate gap inside the received extent")
    return extent, problems


def coverage_report(
    box: Box,
    lat,
    lon,
    policy: str = "enclose",
    *,
    lat_step: float | None = None,
    lon_step: float | None = None,
    tol: float = TOLERANCE,
) -> dict:
    """Requested box, received node and cell extents, policy and problems (``ok`` if none).

    Longitudes are unwrapped around the centre of the box, so data in either convention,
    and the two joined parts of a box that crosses the antimeridian, are compared directly.
    Reported longitudes are in that unwrapped frame.
    """
    _check_policy(policy)
    lat_extent, problems = _axis_report(lat, box.south, box.north, policy, lat_step, tol)
    width = box.width
    centre = box.west + width / 2
    lon_u = centre + ((np.asarray(lon, dtype=float) - centre + 180.0) % 360.0 - 180.0)
    if width >= 360 - tol:
        lon_extent, lon_problems = _axis_report(
            lon_u, centre - 180, centre + 180, "inside", lon_step, tol
        )
        if lon_extent["cells"] and lon_extent["cells"][1] - lon_extent["cells"][0] < 360 - tol:
            lon_problems.append("a global box needs cells all around the globe")
    else:
        lon_extent, lon_problems = _axis_report(
            lon_u, centre - width / 2, centre + width / 2, policy, lon_step, tol
        )
    problems += [f"longitude: {p}" for p in lon_problems]
    return {
        "policy": policy,
        "requested_box": box.as_dict(),
        "latitude": lat_extent,
        "longitude": lon_extent,
        "ok": not problems,
        "problems": problems,
    }


def check_coverage(box: Box, lat, lon, policy: str = "enclose", **kwargs) -> dict:
    """``coverage_report`` that raises ``CoverageError`` (with the report) on any problem."""
    report = coverage_report(box, lat, lon, policy, **kwargs)
    if not report["ok"]:
        raise CoverageError(
            "received extent fails the box policy: " + "; ".join(report["problems"]), report
        )
    return report


# --- 2. Request identity and ledger -----------------------------------------------------

UNORDERED_KEYS = frozenset({"variable", "variables"})


def normalize_request(request: Any, unordered: frozenset = UNORDERED_KEYS) -> Any:
    """Canonical form of a request: numbers as floats, variable lists sorted, keys sorted."""
    if isinstance(request, Mapping):
        return {
            str(k): sorted(normalize_request(x, unordered) for x in v)
            if k in unordered and isinstance(v, (list, tuple))
            else normalize_request(v, unordered)
            for k, v in sorted(request.items())
        }
    if isinstance(request, (list, tuple)):
        return [normalize_request(x, unordered) for x in request]
    if isinstance(request, bool) or request is None or isinstance(request, str):
        return request
    if isinstance(request, (int, float, np.integer, np.floating)):
        return float(request)
    if isinstance(request, Path):
        return request.as_posix()
    return str(request)


def as_json(value: Any) -> Any:
    """A JSON-safe copy of ``value`` as it would be sent (paths as POSIX strings)."""

    def default(obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return str(obj)

    return json.loads(json.dumps(value, default=default))


def request_sha256(request: Mapping, unordered: frozenset = UNORDERED_KEYS) -> str:
    text = json.dumps(normalize_request(request, unordered), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def request_id(request: Mapping, prefix: str = "", length: int = 10) -> str:
    """Short identity of a request: ``<prefix>_<hash>``; any change of the request changes it."""
    digest = request_sha256(request)[:length]
    return f"{prefix}_{digest}" if prefix else digest


STATES = ("planned", "downloaded", "validated", "failed", "quarantined")
LEDGER_SCHEMA = "xhw-benchmark-ledger-v1"


class Ledger:
    """Persistent state of every piece of a plan, saved atomically after each change.

    Paths are stored relative to the ledger's folder. Methods are thread-safe. Saving after
    every change keeps a crash from losing a validated piece; for plans of many thousand
    pieces pass ``autosave=False`` and call ``save()`` periodically.
    """

    def __init__(self, path: str | Path, autosave: bool = True):
        self.path = Path(path)
        self.autosave = autosave
        self._lock = threading.RLock()
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema") != LEDGER_SCHEMA:
                raise LedgerError(f"{self.path.name}: unknown ledger schema {data.get('schema')!r}")
        else:
            data = {"schema": LEDGER_SCHEMA, "pieces": {}, "products": {}}
        self.data = data

    @property
    def pieces(self) -> dict:
        return self.data["pieces"]

    def _rel(self, path: str | Path) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(self.path.parent.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def resolve(self, stored: str) -> Path:
        """Absolute path of a path stored in the ledger."""
        path = Path(stored)
        return path if path.is_absolute() else self.path.parent / path

    def save(self) -> None:
        with self._lock:
            write_json_atomic(self.path, self.data)

    def _changed(self) -> None:
        if self.autosave:
            self.save()

    def get(self, piece_id: str) -> dict | None:
        return self.pieces.get(piece_id)

    def plan(self, piece_id: str, request: Mapping, target: str | Path) -> dict:
        """Register a piece as planned, or return its entry if the same request is known."""
        digest = request_sha256(request)
        with self._lock:
            entry = self.pieces.get(piece_id)
            if entry is not None:
                if entry["request_sha256"] != digest:
                    raise LedgerError(f"{piece_id}: the ledger holds another request with this id")
                return entry
            entry = {
                "id": piece_id,
                "request": as_json(request),
                "request_sha256": digest,
                "target": self._rel(target),
                "status": "planned",
                "attempts": 0,
                "planned_utc": utc_now(),
                "error": None,
                "damage": [],
            }
            self.pieces[piece_id] = entry
            self._changed()
            return entry

    def update(self, piece_id: str, status: str | None = None, **fields) -> dict:
        """Set ``status`` (stamping ``<status>_utc``) and any other fields of a piece."""
        if status is not None and status not in STATES:
            raise ValueError(f"unknown status {status!r}; one of {', '.join(STATES)}")
        with self._lock:
            entry = self.pieces[piece_id]
            entry.update(fields)
            if status is not None:
                entry["status"] = status
                entry[f"{status}_utc"] = utc_now()
            self._changed()
            return entry

    def record_piece(
        self,
        piece_id: str,
        path: str | Path | Sequence[str | Path],
        checks: Mapping | None = None,
    ) -> dict:
        """Mark a piece validated with the hashes of its final file, or files.

        A request may produce several files (one per variable and month): all are recorded
        under ``files`` and all must be intact for the piece to be reused.
        """
        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
        if not paths:
            raise ValueError(f"{piece_id}: no final file to record")
        records = {}
        for one in paths:
            record = {"bytes": one.stat().st_size, "sha256": file_sha256(one)}
            if one.suffix == ".nc":
                record["content_sha256"] = content_sha256(one)
            records[self._rel(one)] = record
        first = self._rel(paths[0])
        piece = {"path": first, **records[first]}
        if len(paths) > 1:
            piece["files"] = records
        return self.update(
            piece_id, "validated", piece=piece, checks=dict(checks or {}), error=None
        )

    def reusable(self, piece_id: str) -> bool:
        """True only if the piece is validated and its file still has the recorded SHA-256.

        A validated piece whose file is missing or changed goes back to ``planned``, with the
        reason kept in ``reverted``.
        """
        with self._lock:
            entry = self.pieces.get(piece_id)
            if entry is None or entry["status"] != "validated":
                return False
            piece = entry["piece"]
            files = piece.get("files") or {piece["path"]: piece}
            reason = None
            for stored, record in files.items():
                path = self.resolve(stored)
                if not path.exists():
                    reason = f"file missing: {stored}"
                elif file_sha256(path) != record["sha256"]:
                    reason = f"sha256 changed: {stored}"
                if reason:
                    break
            if reason is None:
                return True
            entry.setdefault("reverted", []).append({"utc": utc_now(), "reason": reason})
            self.update(piece_id, "planned")
            return False

    def record_product(
        self, name: str, path: str | Path, inputs: Mapping[str, str], **extra
    ) -> dict:
        """Record a product built from pieces, with the SHA-256 of each input as used.

        Written before the product's manifest, so an interrupted run can tell a finished
        product from a stale or partial one (``product_status``).
        """
        path = Path(path)
        record = {
            "path": self._rel(path),
            "sha256": file_sha256(path),
            "inputs": dict(sorted(inputs.items())),
            "written_utc": utc_now(),
            **extra,
        }
        with self._lock:
            self.data["products"][name] = record
            self._changed()
        return record

    def product_status(self, name: str, path: str | Path, inputs: Mapping[str, str]) -> str:
        """``current``, ``missing``, ``unrecorded``, ``changed`` or ``stale``.

        ``stale`` means the product was built from another list of pieces, or from a piece
        whose SHA-256 has changed since: it must be rebuilt.
        """
        path = Path(path)
        record = self.data["products"].get(name)
        if not path.exists():
            return "missing"
        if record is None or self.resolve(record["path"]).resolve() != path.resolve():
            return "unrecorded"
        if file_sha256(path) != record["sha256"]:
            return "changed"
        if record["inputs"] != dict(sorted(inputs.items())):
            return "stale"
        return "current"

    def statuses(self, ids: Sequence[str] | None = None) -> dict[str, str]:
        ids = list(self.pieces) if ids is None else ids
        return {i: self.pieces[i]["status"] if i in self.pieces else "unplanned" for i in ids}

    def retrieval_window(self, ids: Sequence[str] | None = None) -> list[str] | None:
        """First and last download time (UTC) of the given pieces, or None."""
        ids = list(self.pieces) if ids is None else ids
        stamps = sorted(
            self.pieces[i]["downloaded_utc"]
            for i in ids
            if i in self.pieces and self.pieces[i].get("downloaded_utc")
        )
        return [stamps[0], stamps[-1]] if stamps else None


# --- 3. Atomic writes and sidecars ------------------------------------------------------


@contextmanager
def atomic_target(target: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to ``target``; ``os.replace`` it there only on success."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, data: Any) -> Path:
    with atomic_target(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return Path(path)


def write_netcdf_atomic(ds, path: str | Path, **kwargs) -> Path:
    """``ds.to_netcdf`` through a temporary file (h5netcdf engine unless given)."""
    kwargs.setdefault("engine", "h5netcdf")
    with atomic_target(path) as tmp:
        ds.to_netcdf(tmp, **kwargs)
    return Path(path)


def sidecar_path(path: str | Path) -> Path:
    """``t2m_1991-01.nc`` -> ``t2m_1991-01.provenance.json``."""
    return Path(path).with_suffix(".provenance.json")


def write_sidecar(path: str | Path, request: Mapping, **extra) -> Path:
    """Provenance next to a final file: its request, size and SHA-256."""
    path = Path(path)
    return write_json_atomic(
        sidecar_path(path),
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "request": as_json(request),
            "request_sha256": request_sha256(request),
            "written_utc": utc_now(),
            **extra,
        },
    )


def sidecar_matches(path: str | Path, request: Mapping) -> bool:
    """True if the file exists, has a sidecar for the same request and the same SHA-256."""
    path, side = Path(path), sidecar_path(path)
    if not (path.exists() and side.exists()):
        return False
    try:
        record = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("request_sha256") == request_sha256(request) and record.get(
        "sha256"
    ) == file_sha256(path)


# --- 6. Error classification and retries ------------------------------------------------

TRANSIENT, ONCE, PERMANENT = "transient", "once", "permanent"
TRANSIENT_HTTP = frozenset({408, 429, 500, 502, 503, 504})
#: Exception class names (any class in the MRO) and their class. Verified classes: icechunk
#: in revision 3 (10.1), copernicusmarine 2.5.0 in revision 5 (4.3); network names follow
#: requests, urllib3 and aiohttp.
ERROR_NAMES = {
    "SnapshotNotFoundError": PERMANENT,
    "RepositoryNotFoundError": PERMANENT,
    "InvalidInputError": PERMANENT,
    "DatasetVersionNotFound": PERMANENT,
    "DatasetNotFound": PERMANENT,
    "VariableDoesNotExistInTheDataset": PERMANENT,
    "CoordinatesOutOfDatasetBounds": PERMANENT,
    "TimeoutError": TRANSIENT,
    "ConnectionError": TRANSIENT,
    "Timeout": TRANSIENT,
    "ReadTimeout": TRANSIENT,
    "ConnectTimeout": TRANSIENT,
    "ChunkedEncodingError": TRANSIENT,
    "ProtocolError": TRANSIENT,
    "IncompleteRead": TRANSIENT,
    "ClientConnectionError": TRANSIENT,
    "ServerDisconnectedError": TRANSIENT,
    "ClientPayloadError": TRANSIENT,
}
#: icechunk ``ErrorKind`` values, normalised to kebab case. ``object-store`` also covers a
#: wrong host or URL, so its retries are bounded; ``virtual-chunk-fetch`` may hide a
#: transient 5xx (unverified) and is retried once.
ERROR_KINDS = {
    "object-store": TRANSIENT,
    "io": TRANSIENT,
    "concurrency": TRANSIENT,
    "virtual-chunk-fetch": ONCE,
    "virtual-chunk-modified": PERMANENT,
    "snapshot-not-found": PERMANENT,
    "repository-not-found": PERMANENT,
    "unauthorized-virtual-chunk-container": PERMANENT,
}


def _kebab(value: Any) -> str:
    text = str(getattr(value, "name", value)).rsplit(".", 1)[-1]
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", text)
    return text.replace("_", "-").lower()


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    for value in (
        getattr(response, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "code", None),
    ):
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


def classify_error(
    exc: BaseException, extra: Sequence[Callable[[BaseException], str | None]] = ()
) -> str:
    """``transient``, ``once`` (retry at most once) or ``permanent``.

    ``extra`` rules run first; the first one that returns a class wins, so a source can
    add or override cases. Unknown errors are permanent: retrying blindly repeats a
    permanent error in every piece.
    """
    for rule in extra:
        verdict = rule(exc)
        if verdict is not None:
            return verdict
    if isinstance(exc, AcquireError):
        return PERMANENT
    text = str(exc).lower()
    if "cost limits exceeded" in text or "request is too large" in text:
        return PERMANENT
    kind = getattr(exc, "kind", None)
    if kind is not None and _kebab(kind) in ERROR_KINDS:
        return ERROR_KINDS[_kebab(kind)]
    for cls in type(exc).__mro__:
        if cls.__name__ in ERROR_NAMES:
            return ERROR_NAMES[cls.__name__]
    status = _http_status(exc)
    if status is not None:
        return TRANSIENT if status in TRANSIENT_HTTP else PERMANENT
    return PERMANENT


def backoff(attempt: int) -> float:
    """Wait before retry ``attempt`` (1, 2...): ``min(30 * 2**(attempt - 1), 600)`` seconds."""
    return float(min(30 * 2 ** (attempt - 1), 600))


def call_with_retries(
    fn: Callable[[], Any],
    *,
    retries: int = 3,
    classify: Callable[[BaseException], str] = classify_error,
    wait: Callable[[int], float] = backoff,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Any:
    """Call ``fn``, retrying transient errors up to ``retries`` times (``once`` errors once)."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            verdict = classify(exc)
            limit = {TRANSIENT: retries, ONCE: min(retries, 1)}.get(verdict, 0)
            if attempt >= limit:
                raise
            attempt += 1
            delay = wait(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)


# --- 4. Damaged files -------------------------------------------------------------------


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def discard_damaged(
    ledger: Ledger,
    piece_id: str,
    raw: str | Path,
    error: BaseException,
    *,
    keep_failed: bool = False,
    quarantine_dir: str | Path | None = None,
) -> dict:
    """Record a damaged raw file's size, SHA-256 and error, then delete or quarantine it.

    The file is lost, not the evidence. Returns the evidence recorded in the ledger.
    """
    raw = Path(raw)
    evidence = {"utc": utc_now(), "error": _error_text(error), "bytes": None, "sha256": None}
    dest = None
    if raw.exists():
        evidence["bytes"] = raw.stat().st_size
        evidence["sha256"] = file_sha256(raw)
        if keep_failed:
            folder = Path(quarantine_dir) if quarantine_dir else raw.parent / "quarantine"
            dest = folder / f"{piece_id}_{evidence['sha256'][:8]}{raw.suffix}"
            evidence["quarantine"] = ledger._rel(dest)
    # The evidence reaches the disk before the file is touched; if saving fails, the file stays.
    entry = ledger.get(piece_id)
    ledger.update(piece_id, damage=[*entry.get("damage", []), evidence], error=evidence["error"])
    if not ledger.autosave:
        ledger.save()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(raw, dest)
    else:
        raw.unlink(missing_ok=True)
    return evidence


def acquire(
    ledger: Ledger,
    piece_id: str,
    *,
    fetch: Callable[[Path], Any],
    finish: Callable[[Path], tuple[Path, Mapping]],
    raw: str | Path,
    keep_failed: bool = False,
    quarantine_dir: str | Path | None = None,
    retry_damaged: bool = True,
    retries: int = 3,
    classify: Callable[[BaseException], str] = classify_error,
    wait: Callable[[int], float] = backoff,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Download, validate and publish one planned piece, recording everything in the ledger.

    ``fetch(raw)`` writes the server's answer to ``raw`` (transient errors are retried).
    ``finish(raw)`` opens and validates it, writes the final file (or files) atomically and
    returns ``(final_path_or_paths, checks)``; it raises ``ValidationError`` for an
    unreadable file or one that does not match its request. A damaged raw file is recorded
    and deleted, and the piece is fetched once more (``retry_damaged``); with
    ``keep_failed`` it goes to quarantine instead. A validated piece whose file is intact is
    not fetched again. Never raises for a failed piece: the returned entry says what
    happened.
    """
    if ledger.reusable(piece_id):
        return ledger.get(piece_id)
    raw = Path(raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    damaged = 0
    while True:
        entry = ledger.get(piece_id)
        ledger.update(piece_id, attempts=entry["attempts"] + 1, error=None)
        try:
            call_with_retries(
                lambda: fetch(raw), retries=retries, classify=classify, wait=wait, sleep=sleep
            )
        except Exception as exc:
            raw.unlink(missing_ok=True)
            return ledger.update(piece_id, "failed", error=_error_text(exc))
        ledger.update(
            piece_id, "downloaded", raw={"bytes": raw.stat().st_size, "sha256": file_sha256(raw)}
        )
        try:
            path, checks = finish(raw)
        except ValidationError as exc:
            damaged += 1
            discard_damaged(
                ledger, piece_id, raw, exc, keep_failed=keep_failed, quarantine_dir=quarantine_dir
            )
            if keep_failed:
                return ledger.update(piece_id, "quarantined")
            if retry_damaged and damaged == 1:
                continue
            return ledger.update(piece_id, "failed")
        except Exception as exc:
            raw.unlink(missing_ok=True)
            return ledger.update(piece_id, "failed", error=_error_text(exc))
        raw.unlink(missing_ok=True)
        return ledger.record_piece(piece_id, path, checks)


# --- 6. Parallel runs and summary -------------------------------------------------------


@dataclass
class Outcome:
    key: str
    ok: bool
    result: Any = None
    error: BaseException | None = None


def run_parallel(
    tasks: Mapping[str, Callable[[], Any]],
    workers: int,
    on_done: Callable[[Outcome], None] | None = None,
) -> dict[str, Outcome]:
    """Run every task in a bounded thread pool; no failure cancels the others.

    Results are collected with ``as_completed`` and returned in the order of ``tasks``,
    with the exception of each failed task.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    outcomes: dict[str, Outcome] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                outcome = Outcome(key, True, result=future.result())
            except Exception as exc:
                outcome = Outcome(key, False, error=exc)
            outcomes[key] = outcome
            if on_done is not None:
                on_done(outcome)
    return {key: outcomes[key] for key in tasks}


@dataclass
class RunSummary:
    """Status of every piece of a run and the errors of those not validated."""

    statuses: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_ledger(
        cls, ledger: Ledger, ids: Sequence[str], outcomes: Mapping[str, Outcome] | None = None
    ) -> RunSummary:
        statuses = ledger.statuses(ids)
        errors = {}
        for i in ids:
            outcome = (outcomes or {}).get(i)
            if outcome is not None and not outcome.ok:
                errors[i] = _error_text(outcome.error)
                if statuses[i] == "validated":  # failed after its piece was validated
                    statuses[i] = "failed"
            elif statuses[i] != "validated":
                errors[i] = (ledger.get(i) or {}).get("error") or "not validated"
        return cls(statuses, errors)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for status in self.statuses.values():
            out[status] = out.get(status, 0) + 1
        return out

    @property
    def exit_code(self) -> int:
        return 0 if all(s == "validated" for s in self.statuses.values()) else 1

    def report(self, stream=None) -> None:
        stream = stream or sys.stdout
        counts = ", ".join(f"{n} {s}" for s, n in sorted(self.counts().items()))
        print(f"{len(self.statuses)} pieces: {counts}", file=stream)
        for key, error in self.errors.items():
            print(f"  {key} [{self.statuses[key]}]: {error}", file=stream)


# --- 5. Manifest ------------------------------------------------------------------------

SOFTWARE = (
    "numpy", "xarray", "h5netcdf", "h5py", "netCDF4", "pandas", "zarr", "icechunk", "fsspec",
    "cdsapi", "ecmwf-datastores-client", "copernicusmarine", "xhw",
)  # fmt: skip


def software_versions(packages: Sequence[str] = SOFTWARE) -> dict:
    """Python and installed package versions (None for packages not installed)."""
    out = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def file_record(path: str | Path) -> dict:
    """Size and SHA-256 of a file, plus ``content_sha256`` for NetCDF files."""
    path = Path(path)
    record = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if path.suffix == ".nc":
        record["content_sha256"] = content_sha256(path)
    return record


def build_manifest(
    *,
    source: str,
    request: Mapping,
    files: Mapping[str, str | Path],
    coverage: Mapping | None = None,
    retrieval_window: Sequence[str] | None = None,
    pieces: Sequence[str] | None = None,
    fingerprint: bool = False,
    **extra,
) -> dict:
    """Common manifest: request, file hashes, software, retrieval window and box coverage.

    ``coverage`` is a ``coverage_report`` (requested box, received extents, policy).
    ``fingerprint`` adds ``provenance.code_fingerprint()``, which needs xHW installed.
    """
    manifest = {
        "source": source,
        "request": as_json(request),
        "request_sha256": request_sha256(request),
        "written_utc": utc_now(),
        "retrieval_window_utc": list(retrieval_window) if retrieval_window else None,
        "box": dict(coverage) if coverage is not None else None,
        "files": {name: file_record(path) for name, path in files.items()},
        "pieces": list(pieces) if pieces is not None else None,
        "software": software_versions(),
        "review_status": "new retrieval; not the registered reviewed input",
        **extra,
    }
    if fingerprint:
        from provenance import code_fingerprint

        manifest["code_fingerprint"] = code_fingerprint()
    return manifest


def manifest_path(
    directory: str | Path, prefix: str, request: Mapping, when: datetime | None = None
) -> Path:
    """``<prefix>_manifest_<request hash>_<UTC stamp>.json``: repeated runs never overwrite."""
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(directory) / f"{prefix}_manifest_{request_id(request)}_{stamp}.json"


def matches_registered(record: Mapping, reference: Mapping) -> dict:
    """Compare a file record with a registered one (``examples/sources/*.json``).

    ``content_sha256`` decides (values and decoding, not container bytes); ``sha256`` is
    reported too. Each is None when the registered record lacks it.
    """

    def same(key):
        if not reference.get(key):
            return None
        return record.get(key) == reference[key]

    return {"content_sha256": same("content_sha256"), "sha256": same("sha256")}


def write_manifest(directory: str | Path, prefix: str, manifest: Mapping) -> Path:
    """Write the manifest atomically and add it to the folder's ``manifests.json`` index."""
    directory = Path(directory)
    path = manifest_path(directory, prefix, manifest["request"])
    write_json_atomic(path, manifest)
    index_path = directory / "manifests.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    index.append(
        {
            "file": path.name,
            "source": manifest.get("source"),
            "request_sha256": manifest.get("request_sha256"),
            "written_utc": manifest.get("written_utc"),
        }
    )
    write_json_atomic(index_path, index)
    return path


# --- 7. CDS cost planner ----------------------------------------------------------------

CDS_COST_PER_FIELD = 6
CDS_COST_LIMITS = {
    "reanalysis-era5-single-levels": 121_000,
    "reanalysis-era5-pressure-levels": 60_000,
}  # measured with estimate_costs on 2026-11-03; the cost does not depend on the area
PERIODS = {"month": 1, "half-year": 6, "year": 12}


def cds_cost(n_variables: int, n_hours: int, n_days: int, n_levels: int = 1) -> int:
    """CDS cost of a request: 6 x variables x levels x hours x real (existing) days."""
    return CDS_COST_PER_FIELD * n_variables * max(n_levels, 1) * n_hours * n_days


def cds_date_fields(start: date, end: date) -> dict:
    """``year``/``month``/``day`` lists whose real dates are exactly ``start..end``.

    Valid for one partial month, or whole months of one year (days 1-31; the CDS ignores
    dates that do not exist, and so does its cost).
    """
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    years = sorted({d.year for d in days})
    months = sorted({d.month for d in days})
    day_numbers = sorted({d.day for d in days})
    if len(years) == 1 and len(months) > 1:
        day_numbers = list(range(1, 32))
    product = {
        date(y, m, d)
        for y in years
        for m in months
        for d in day_numbers
        if d <= calendar.monthrange(y, m)[1]
    }
    if product != set(days):
        raise ValueError(f"{start}..{end} is not one partial month or whole months of one year")
    return {
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in day_numbers],
    }


@dataclass(frozen=True)
class CdsChunk:
    start: date
    end: date
    levels: tuple[str, ...]
    cost: int

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def date_fields(self) -> dict:
        return cds_date_fields(self.start, self.end)


def _blocks(start: date, end: date, months_per_block: int) -> list[tuple[date, date]]:
    """Calendar blocks of whole months (aligned to the year); partial months stand alone."""
    blocks: list[tuple[date, date]] = []
    current = None
    day = start
    while day <= end:
        month_end = date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])
        piece_end = min(month_end, end)
        whole = day.day == 1 and piece_end == month_end
        aligned = (day.month - 1) % months_per_block == 0
        if whole and current is not None and not aligned:
            current = (current[0], piece_end)
            blocks[-1] = current
        else:
            current = (day, piece_end) if whole else None
            blocks.append((day, piece_end))
        day = piece_end + timedelta(1)
    return blocks


def _groups(levels: Sequence[str], size: int) -> list[tuple[str, ...]]:
    """Contiguous groups of at most ``size`` levels, as balanced as possible."""
    n = math.ceil(len(levels) / size)
    q, r = divmod(len(levels), n)
    out, i = [], 0
    for k in range(n):
        step = q + (1 if k < r else 0)
        out.append(tuple(levels[i : i + step]))
        i += step
    return out


def plan_cds_chunks(
    start: date,
    end: date,
    *,
    dataset: str,
    n_variables: int,
    n_hours: int = 24,
    levels: Sequence[str] = (),
    period: str = "month",
    margin: float = 0.9,
    level_group: int | None = None,
    limit: int | None = None,
) -> list[CdsChunk]:
    """Split ``start..end`` (and the levels) so every request stays under ``margin x limit``.

    Blocks follow ``period`` (calendar months, half-years or years). A block that is too
    large even for one level is split into months, then into blocks of days; otherwise its
    levels are split into the fewest balanced groups that fit (or groups of ``level_group``,
    which must fit). Raises ``CostLimitError`` if one day of one level already exceeds it.
    """
    if period not in PERIODS:
        raise ValueError(f"period must be one of {', '.join(PERIODS)}")
    if not 0 < margin <= 1:
        raise ValueError("margin must be in (0, 1]")
    if start > end:
        raise ValueError("start is after end")
    if n_variables < 1 or n_hours < 1 or (level_group is not None and level_group < 1):
        raise ValueError("n_variables, n_hours and level_group must be positive")
    limit = CDS_COST_LIMITS[dataset] if limit is None else limit
    budget = int(limit * margin)
    levels = tuple(str(level) for level in levels)
    if cds_cost(n_variables, n_hours, 1) > budget:
        raise CostLimitError(
            f"one day of one level costs more than {budget}; request fewer variables"
        )
    chunks: list[CdsChunk] = []

    def add(first: date, last: date) -> None:
        days = (last - first).days + 1
        one = cds_cost(n_variables, n_hours, days)
        if one > budget:
            if (first.year, first.month) != (last.year, last.month):
                for a, b in _blocks(first, last, 1):
                    add(a, b)
                return
            span = budget // cds_cost(n_variables, n_hours, 1)
            day = first
            while day <= last:
                add(day, min(day + timedelta(span - 1), last))
                day += timedelta(span)
            return
        size = level_group if level_group is not None else budget // one
        if level_group is not None and one * level_group > budget:
            raise CostLimitError(
                f"{level_group} levels of {first}..{last} cost {one * level_group} > {budget}"
            )
        for group in _groups(levels, size) if levels else [()]:
            chunks.append(
                CdsChunk(first, last, group, cds_cost(n_variables, n_hours, days, len(group)))
            )

    for first, last in _blocks(start, end, PERIODS[period]):
        add(first, last)
    return chunks


# --- Command line -----------------------------------------------------------------------


#: Named boxes, shortcuts of ``--box`` (revision 0, decision 5). Each source maps a region
#: to its own manifest key (for OISST: oisst, oisst_equatorial, oisst_npacific).
REGIONS = {
    "peru": Box(-12.0, -4.0, -82.0, -77.0),
    "equatorial": Box(-5.0, 5.0, -170.0, -120.0),  # Nino 3.4
    "npacific": Box(25.0, 55.0, 160.0, -130.0),  # 160E-130W, crosses the antimeridian
}


def add_box_arguments(
    parser: argparse.ArgumentParser,
    default_region: str | None = "peru",
    regions: Sequence[str] = tuple(REGIONS),
) -> None:
    """``--region`` or ``--box S N W E``, and ``--box-policy`` (default ``enclose``)."""
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--region",
        choices=list(regions),
        default=default_region,
        help=f"named box (default {default_region})",
    )
    where.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("S", "N", "W", "E"),
        help="south north west east, in degrees; replaces --region",
    )
    parser.add_argument(
        "--box-policy",
        choices=POLICIES,
        default="enclose",
        help="enclose: nodes on or beyond every edge (default); cover-cells: cells reach "
        "every edge; inside: only nodes inside (the registered inputs)",
    )


def box_from_args(args: argparse.Namespace) -> tuple[Box, str | None]:
    """The requested box and its region name (None for a free ``--box``)."""
    if args.box is not None:
        return Box(*args.box), None
    if args.region is None:
        raise ValueError("give --region or --box")
    return REGIONS[args.region], args.region


def add_period_arguments(parser: argparse.ArgumentParser, start: str, end: str) -> None:
    """``--start`` and ``--end`` (inclusive ISO dates)."""
    parser.add_argument("--start", type=date.fromisoformat, default=date.fromisoformat(start))
    parser.add_argument("--end", type=date.fromisoformat, default=date.fromisoformat(end))


def period_from_args(
    args: argparse.Namespace, first: date | None = None, last: date | None = None
) -> tuple[date, date]:
    """``(start, end)``, checked to be ordered and, if given, within ``first..last``."""
    if args.start > args.end:
        raise ValueError(f"--start {args.start} is after --end {args.end}")
    if first is not None and args.start < first:
        raise ValueError(f"--start {args.start} is before the first available day {first}")
    if last is not None and args.end > last:
        raise ValueError(f"--end {args.end} is after the last available day {last}")
    return args.start, args.end


def add_run_arguments(parser: argparse.ArgumentParser, workers: int = 4) -> None:
    """``--workers``, ``--retries`` and ``--keep-failed``."""
    parser.add_argument(
        "--workers", type=int, default=workers, help=f"simultaneous requests (default {workers})"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="retries of transient errors (default 3)"
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="move damaged files to quarantine instead of deleting them and retrying once",
    )
