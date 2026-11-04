# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional ERA5 single-levels acquisition and daily products; default prints the plan.

Requests hourly surface variables from the Copernicus Climate Data Store (CDS) dataset
``reanalysis-era5-single-levels``, plus the land-sea mask. Install
``examples/requirements-download.txt`` in a separate acquisition environment, keep your CDS
key in ``~/.cdsapirc`` and accept the ERA5 single-levels licence once on the CDS web page.

    python examples/download_era5.py --data-dir data                 # print the plan
    python examples/download_era5.py --data-dir data --download      # hourly files + lsm
    python examples/download_era5.py --data-dir data --daily         # local-day products
    python examples/download_era5.py --data-dir data --consolidate-years   # yearly files

Defaults reproduce the registered tutorial input: 1991-2025 over the Peru box, t2m, mx2t and
mn2t, local days at UTC-05:00. ``--daily`` writes the registered TX/TN file (TX = max of
period-ending mx2t, TN = min of mn2t, t2m extrema as a contrast) and one daily file per
other variable. Accumulated variables become mean fluxes (W m-2), or a daily total for
``tp``; units are converted here because xHW does not convert them. ``--derived`` adds the
products of ``examples/_derived.py``; ``--consolidate-years`` writes one hourly file per UTC
year with the variables together, the layout of the registered input.

What the script guarantees (revisions 0 and 2 of the development repository):

- the received data contain the requested box: ``--box-policy enclose`` (default) rounds
  the CDS ``area`` outward to the 0.25 deg grid, and every answer is checked;
- requests stay under the CDS cost limit (121 000 = 6 x variables x hours x days), monthly
  by default and grouped by GRIB step type (``--request-grouping``);
- every request has an identity (hash of its body) and a state in ``ledger.json``; a file
  is reused only if its SHA-256 is the recorded one, and daily products are built only
  from the files of the current plan, never from whatever is in the folder;
- a damaged answer is recorded (size, SHA-256, error), deleted and requested once more,
  or kept in quarantine with ``--keep-failed``; every failure is reported at the end;
- preliminary ERA5T data (``expver`` 0005) are rejected unless ``--allow-era5t``; data
  without ``expver`` are rejected unless ``--accept-unknown-expver-before`` applies.

The plan of the registered input (72 requests: half-years of the three variables together,
7 boundary hours and the mask) is reproduced with
``--chunk-period half-year --request-grouping all --boundary-hours 7``. New files are not
automatically equivalent to the registered tutorial input; the manifest compares their
``content_sha256`` with examples/sources/era5.json. The development repository also
compared this route with the ARCO-ERA5 copy and an Earth Data Hub mirror; those checks
are not part of this example.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import _acquire as aq
import _derived
from content_hash import registered

DATASET = "reanalysis-era5-single-levels"
GRID_STEP = 0.25
DAY_DEFINITION = "offset:-05:00"
INVARIANT_DATE = date(2020, 1, 1)  # the land-sea mask of the registered input
DEFAULT_VARIABLES = ("t2m", "mx2t", "mn2t")
CATALOGUE_VERSION = "era5-single-1"
SOURCE = f"Copernicus Climate Data Store {DATASET} (ERA5, Copernicus licence)"


@dataclass(frozen=True)
class Variable:
    """How one ERA5 variable is requested, received, validated and aggregated."""

    cds: str
    kind: str  # instantaneous, accumulated, period_extreme, invariant
    units: str  # native units, normalised (see _units); empty: not checked
    step_type: str  # GRIB stepType: instant, accum, max, min
    daily: tuple[str, ...] = ()  # statistics of the per-variable daily file
    verified: bool = True  # name and units seen in a CDS answer on 2026-11-03

    @property
    def sampling(self) -> str:
        return "instantaneous" if self.kind == "instantaneous" else "period_ending"


#: Revision 2, section 7.1. ``mn2t`` (registered product), ``v10`` and ``strd`` (bonino2024xt)
#: were not requested in the 2026-11-03 checks; ``lsm`` units were not checked either.
CATALOGUE = {
    "t2m": Variable("2m_temperature", "instantaneous", "K", "instant", ("mean", "max", "min")),
    "mx2t": Variable(
        "maximum_2m_temperature_since_previous_post_processing", "period_extreme", "K", "max",
        ("max",),
    ),
    "mn2t": Variable(
        "minimum_2m_temperature_since_previous_post_processing", "period_extreme", "K", "min",
        ("min",), verified=False,
    ),
    "lsm": Variable("land_sea_mask", "invariant", "", "instant", verified=False),
    "z": Variable("geopotential", "invariant", "m2 s-2", "instant"),
    "sp": Variable("surface_pressure", "instantaneous", "Pa", "instant", ("mean",)),
    "msl": Variable("mean_sea_level_pressure", "instantaneous", "Pa", "instant", ("mean",)),
    "u10": Variable("10m_u_component_of_wind", "instantaneous", "m s-1", "instant", ("mean",)),
    "v10": Variable(
        "10m_v_component_of_wind", "instantaneous", "m s-1", "instant", ("mean",), verified=False
    ),
    "d2m": Variable("2m_dewpoint_temperature", "instantaneous", "K", "instant", ("mean",)),
    "sshf": Variable("surface_sensible_heat_flux", "accumulated", "J m-2", "accum", ("mean",)),
    "slhf": Variable("surface_latent_heat_flux", "accumulated", "J m-2", "accum", ("mean",)),
    "ssrd": Variable(
        "surface_solar_radiation_downwards", "accumulated", "J m-2", "accum", ("mean",)
    ),
    "strd": Variable(
        "surface_thermal_radiation_downwards", "accumulated", "J m-2", "accum", ("mean",),
        verified=False,
    ),
    "ssr": Variable("surface_net_solar_radiation", "accumulated", "J m-2", "accum", ("mean",)),
    "str": Variable("surface_net_thermal_radiation", "accumulated", "J m-2", "accum", ("mean",)),
    "tp": Variable("total_precipitation", "accumulated", "m", "accum", ("total",)),
    "fg10": Variable(
        "10m_wind_gust_since_previous_post_processing", "period_extreme", "m s-1", "max", ("max",)
    ),
    "fdir": Variable(
        "total_sky_direct_solar_radiation_at_surface", "accumulated", "J m-2", "accum",
        ("mean",), verified=False,
    ),
}  # fmt: skip
INVARIANTS = ("lsm", "z")
STEP_ORDER = ("instant", "accum", "max", "min")


class Era5tError(aq.PermanentError):
    """Preliminary (ERA5T) or unknown-status data refused by the ``expver`` policy."""


def _units(text) -> str:
    """``m s**-1`` -> ``m s-1``: the spelling of the CDS NetCDF and of the catalogue agree."""
    return re.sub(r"\s+", " ", str(text).replace("**", "").replace("^", "")).strip()


# --- Plan --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """One CDS request: variables of one group over whole days ``start..end`` at ``hours``."""

    id: str
    body: dict
    variables: tuple[str, ...]
    start: date
    end: date
    hours: tuple[int, ...]
    part: str = ""  # "w" or "e" for the two parts of a box crossing the antimeridian
    box: aq.Box | None = None  # the box of this part (None: the whole requested box)

    @property
    def invariant(self) -> bool:
        return all(CATALOGUE[v].kind == "invariant" for v in self.variables)

    @property
    def cost(self) -> int:
        return aq.cds_cost(len(self.variables), len(self.hours), (self.end - self.start).days + 1)

    def stamps(self) -> list[datetime]:
        """Every hour the answer must contain, in order."""
        days = (self.end - self.start).days + 1
        midnight = datetime.combine(self.start, datetime.min.time())
        return [midnight + timedelta(days=d, hours=h) for d in range(days) for h in self.hours]

    def outputs(self, root: Path) -> dict[tuple[str, date, date], Path]:
        """Final files: one per variable and calendar month (or part of one)."""
        out = {}
        full = self.hours == tuple(range(24))
        for var in self.variables:
            suffix = f"_{self.part}" if self.part else ""
            if CATALOGUE[var].kind == "invariant":
                out[(var, self.start, self.end)] = root / var / f"{var}{suffix}.nc"
                continue
            day = self.start
            while day <= self.end:
                month_end = (day.replace(day=28) + timedelta(4)).replace(day=1) - timedelta(1)
                last = min(month_end, self.end)
                if full and day.day == 1 and last == month_end:
                    name = f"{var}_{day:%Y-%m}.nc"
                else:
                    name = f"{var}_{day:%Y-%m-%d}_{last:%Y-%m-%d}"
                    if not full:
                        name += f"_h{self.hours[0]:02d}-{self.hours[-1]:02d}"
                    name += ".nc"
                name = name[:-3] + suffix + ".nc"
                out[(var, day, last)] = root / var / f"{day:%Y}" / name
                day = last + timedelta(1)
        return out


def _offset(day_definition: str) -> timedelta:
    if day_definition == "UTC":
        return timedelta(0)
    match = re.fullmatch(r"offset:([+-])(\d{2}):(\d{2})", day_definition)
    if not match:
        raise ValueError(f"day definition must be UTC or offset:+HH:MM, got {day_definition!r}")
    sign = 1 if match.group(1) == "+" else -1
    return sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))


def utc_window(variables, start: date, end: date, day_definition: str) -> tuple[datetime, datetime]:
    """First and last UTC hour that the local days ``start..end`` need (revision 2, V4).

    Local day D covers UTC ``[D 00:00 - offset, D+1 00:00 - offset)``. Instantaneous stamps
    ``t`` are taken on ``[a, b)``; a period-ending stamp belongs to the day where its hour
    starts, so ``t - 1 h`` is on ``[a, b)``, as ``xhw.aggregate_daily`` counts them. With an
    offset in minutes (``+05:30``) the bounds fall between hourly stamps.
    """
    kinds = {CATALOGUE[v].sampling for v in variables if CATALOGUE[v].kind != "invariant"}
    windows = [hour_window(k, start, end, day_definition) for k in kinds or {"instantaneous"}]
    return min(w[0] for w in windows), max(w[1] for w in windows)


def _floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def hour_window(sampling: str, start: date, end: date, day_definition: str):
    """First and last hourly stamp of local days ``start..end`` for one sampling."""
    offset = _offset(day_definition)
    hour = timedelta(hours=1)
    a = datetime.combine(start, datetime.min.time()) - offset
    b = datetime.combine(end + timedelta(1), datetime.min.time()) - offset
    first = a if a == _floor_hour(a) else _floor_hour(a) + hour  # stamps in [a, b)
    last = (b if b == _floor_hour(b) else _floor_hour(b) + hour) - hour
    if sampling == "period_ending":  # xhw.aggregate_daily: the hour ending at t starts at t - 1 h
        return first + hour, last + hour
    return first, last


def _groups(variables, grouping: str) -> list[tuple[str, ...]]:
    timed = [v for v in variables if CATALOGUE[v].kind != "invariant"]
    if grouping == "variable":
        return [(v,) for v in timed]
    if grouping == "all":
        return [tuple(timed)] if timed else []
    by_step = {s: tuple(v for v in timed if CATALOGUE[v].step_type == s) for s in STEP_ORDER}
    return [group for group in by_step.values() if group]


@dataclass(frozen=True)
class Part:
    """One CDS area of the requested box: the whole box, or one side of the antimeridian."""

    label: str
    area: list
    box: aq.Box | None = None


def box_parts(box: aq.Box, policy: str) -> list[Part]:
    """The CDS areas of ``box``: one, or two (``w`` and ``e``) if it crosses the antimeridian.

    Whether the CDS accepts ``west > east`` in ``area`` was not verified, so a crossing box
    is requested in two parts (revision 0, B5) and joined west to east when read.
    """
    if not box.crosses_antimeridian:
        return [Part("", aq.cds_area(box, policy, GRID_STEP))]
    west, east = aq.split_box(box)
    return [
        Part("w", aq.cds_area(west, policy, GRID_STEP), west),
        Part("e", aq.cds_area(east, policy, GRID_STEP), east),
    ]


def _parts(area) -> list[Part]:
    """``area`` as a list of parts: a plain ``[N, W, S, E]`` is one part."""
    if area and isinstance(area[0], Part):
        return list(area)
    return [Part("", list(area))]


def _request(area, variables, start: date, end: date, hours, part: Part | None = None) -> Request:
    body = {
        "product_type": ["reanalysis"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
        "variable": [CATALOGUE[v].cds for v in variables],
        **aq.cds_date_fields(start, end),
        "time": [f"{h:02d}:00" for h in hours],
    }
    kind = "inv" if all(CATALOGUE[v].kind == "invariant" for v in variables) else "sl"
    label = variables[0] + (f"+{len(variables) - 1}" if len(variables) > 1 else "")
    if part is not None and part.label:
        label += f"_{part.label}"
    rid = aq.request_id(body, f"{kind}_{start:%Y%m%d}_{label}")
    if part is None:
        return Request(rid, body, tuple(variables), start, end, tuple(hours))
    return Request(rid, body, tuple(variables), start, end, tuple(hours), part.label, part.box)


def _edge_hours(first: datetime, last: datetime, day_start: datetime, day_end: datetime):
    """``[(UTC day, hours)]`` of ``first..last`` outside the whole days ``day_start..day_end``.

    Usually the day before or after; a 09:00-09:00 product at a large offset may need two.
    """
    out = []
    t = _floor_hour(first)
    while t <= last:
        if t < day_start or t > day_end:
            if out and out[-1][0] == t.date():
                out[-1][1].append(t.hour)
            else:
                out.append((t.date(), [t.hour]))
        t += timedelta(hours=1)
    return out


def plan(
    area: list,
    variables,
    start: date,
    end: date,
    *,
    day_definition: str = DAY_DEFINITION,
    chunk_period: str = "month",
    grouping: str = "steptype",
    boundary_hours: int | None = None,
    invariants=("lsm",),
    margin: float = 0.9,
    extra_windows=None,
) -> list[Request]:
    """CDS requests for the hourly data that the local days ``start..end`` need.

    ``area`` is a CDS ``[N, W, S, E]`` or the parts of ``box_parts`` (a box crossing the
    antimeridian): every request is then made once per part.

    Whole UTC days ``start..end`` are split by ``chunk_period`` and the cost limit; the
    hours of the neighbouring UTC days that the day definition needs are requested apart
    (``boundary_hours`` forces hours 00..N-1 of the following day instead); invariants
    are requested once, on 2020-01-01 00 UTC. ``extra_windows`` maps a variable to more UTC
    hours it needs (``(first, last)``), as the 09:00-09:00 days of a derived product.
    """
    unknown = [v for v in [*variables, *invariants] if v not in CATALOGUE]
    if unknown:
        raise ValueError(
            f"variables {unknown} are not in the catalogue; add a row to CATALOGUE with the "
            "CDS name, kind, native units, GRIB step type and daily statistics"
        )
    if grouping not in ("variable", "steptype", "all"):
        raise ValueError(f"unknown request grouping {grouping!r}")
    out = []
    parts = _parts(area)
    day_start = datetime.combine(start, datetime.min.time())
    day_end = datetime.combine(end, datetime.min.time()) + timedelta(hours=23)
    for group in _groups(variables, grouping):
        for chunk in aq.plan_cds_chunks(
            start, end, dataset=DATASET, n_variables=len(group), period=chunk_period, margin=margin
        ):
            for part in parts:
                out.append(_request(part.area, group, chunk.start, chunk.end, range(24), part))
        if boundary_hours is None:
            first, last = utc_window(group, start, end, day_definition)
            for var in group:
                if var in (extra_windows or {}):
                    first = min(first, extra_windows[var][0])
                    last = max(last, extra_windows[var][1])
            edges = _edge_hours(first, last, day_start, day_end)
        else:
            edges = [(end + timedelta(1), list(range(boundary_hours)))] if boundary_hours else []
        for day, hours in edges:
            for part in parts:
                out.append(_request(part.area, group, day, day, hours, part))
    for var in dict.fromkeys([*invariants, *(v for v in variables if v in INVARIANTS)]):
        for part in parts:
            out.append(_request(part.area, [var], INVARIANT_DATE, INVARIANT_DATE, [0], part))
    return out


# --- Reading and validating one answer ---------------------------------------------------


def _open_parts(raw: Path):
    """``[(part name, dataset)]`` of a NetCDF file or of a ZIP of NetCDF parts (loaded)."""
    import xarray as xr

    try:
        if zipfile.is_zipfile(raw):
            parts = []
            with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(raw) as archive:
                names = [n for n in archive.namelist() if n.endswith(".nc")]
                if not names:
                    raise aq.ValidationError("ZIP answer without NetCDF parts")
                archive.extractall(tmp, members=names)
                for name in sorted(names):
                    with xr.open_dataset(Path(tmp) / name, engine="h5netcdf") as opened:
                        parts.append((name, opened.load()))
            return parts
        with xr.open_dataset(raw, engine="h5netcdf") as opened:
            return [(raw.name, opened.load())]
    except aq.ValidationError:
        raise
    except Exception as exc:  # not NetCDF, truncated, bad ZIP...
        raise aq.ValidationError(f"unreadable answer: {type(exc).__name__}: {exc}") from exc


def expver_status(values) -> tuple[str, list[str]]:
    """``final`` (all 0001), ``preliminary`` (any other value) or ``unknown`` (absent)."""
    if values is None:
        return "unknown", []
    import numpy as np

    values = np.atleast_1d(values)  # a one-hour answer (an invariant) carries a scalar expver
    found = sorted({str(v.decode() if isinstance(v, bytes) else v).strip() for v in values})
    if not found or found == [""]:
        return "unknown", found
    if all(v.lstrip("0") == "1" for v in found):
        return "final", found
    return "preliminary", found


@dataclass
class ExpverPolicy:
    """Revision 2, 5.1 and decision 4: reject ERA5T and unknown status unless asked."""

    allow_era5t: bool = False
    accept_unknown_before: date | None = None
    justification: str | None = None

    def allows(self, request, status: str) -> bool:
        """Whether a recorded status is acceptable under this policy (for reused pieces)."""
        if status == "final" or self.allow_era5t:
            return True
        return (
            status == "unknown"
            and self.accept_unknown_before is not None
            and request.end < self.accept_unknown_before
        )

    def record(self) -> dict:
        return {
            "era5t_policy": "accept-marked" if self.allow_era5t else "reject",
            "accept_unknown_expver_before": (
                f"{self.accept_unknown_before}" if self.accept_unknown_before else None
            ),
            "expver_justification": self.justification,
        }

    def check(self, request: Request, status: str, found: list[str]) -> str:
        if status == "final":
            return "final"
        if status == "preliminary" and self.allow_era5t:
            return "preliminary"
        if status == "unknown":
            if self.allow_era5t:
                return "unknown"
            if self.accept_unknown_before is not None and request.end < self.accept_unknown_before:
                return "unknown"
            raise Era5tError(
                f"{request.id}: no expver in the answer, so final data are not proven; use "
                "--accept-unknown-expver-before DATE with a justification, or --allow-era5t"
            )
        raise Era5tError(
            f"{request.id}: preliminary ERA5T data (expver {found}); --allow-era5t accepts them"
        )


def validate(raw: Path, request: Request, box: aq.Box, policy: str, expver: ExpverPolicy):
    """Open an answer and check it against its request; returns the merged data and checks.

    Raises ``ValidationError`` for a damaged answer or one that does not match the request
    (variables, units, step types, hours, duplicates, box) and ``Era5tError`` when the
    ``expver`` policy refuses it.
    """
    import numpy as np
    import xarray as xr

    parts = _open_parts(raw)
    expected = {CATALOGUE[v].cds: v for v in request.variables}
    checks = {"zip_members": [n for n, _ in parts] if len(parts) > 1 else []}
    wanted = np.array(request.stamps(), dtype="datetime64[s]")
    merged, statuses, found = [], [], []
    for name, ds in parts:  # every part is checked on its own before merging
        ds = ds.rename({"valid_time": "time"}) if "valid_time" in ds.dims else ds
        step = re.search(r"stepType-(\w+)", name)
        for var in ds.data_vars:
            if var not in expected.values():
                raise aq.ValidationError(f"{request.id}: unexpected variable {var!r} in {name}")
            if step and CATALOGUE[var].step_type != step.group(1):
                raise aq.ValidationError(
                    f"{request.id}: {var} arrived in the {step.group(1)} part, expected "
                    f"{CATALOGUE[var].step_type}"
                )
            wanted_units, got = CATALOGUE[var].units, _units(ds[var].attrs.get("units", ""))
            if wanted_units and got != wanted_units:
                raise aq.ValidationError(
                    f"{request.id}: {var} in {got!r}, expected {wanted_units!r}"
                )
        status, values = expver_status(ds["expver"].values if "expver" in ds.variables else None)
        statuses.append(status)
        found += values
        ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])
        if request.invariant:
            if "time" in ds.dims:
                ds = ds.isel(time=0, drop=True)
            ds = ds.drop_vars([v for v in ("time",) if v in ds.variables])
        else:
            times = ds["time"].values.astype("datetime64[s]")
            duplicates = int(times.size - np.unique(times).size)
            if (
                duplicates
                or times.size != wanted.size
                or not np.array_equal(np.sort(times), wanted)
            ):
                raise aq.ValidationError(
                    f"{request.id}: {name} has {times.size} hours ({duplicates} duplicated), "
                    f"expected {wanted.size} from {wanted[0]} to {wanted[-1]}"
                )
            ds = ds.sortby("time")
        merged.append(ds)
    try:
        ds = xr.merge(merged, join="exact", compat="override", combine_attrs="drop_conflicts")
    except ValueError as exc:
        raise aq.ValidationError(f"{request.id}: parts on different grids: {exc}") from exc
    missing = sorted(set(expected.values()) - set(ds.data_vars))
    if missing:
        raise aq.ValidationError(f"{request.id}: variables {missing} missing from the answer")
    # the worst status of any part decides: preliminary, then unknown, then final
    status = next(s for s in ("preliminary", "unknown", "final") if s in statuses)
    checks["expver_status"] = expver.check(request, status, sorted(set(found)))
    checks["expver_values"] = sorted(set(found))
    if not request.invariant:
        checks["n_times"] = int(wanted.size)
    try:
        report = aq.check_coverage(
            request.box or box, ds["latitude"].values, ds["longitude"].values, policy,
            lat_step=GRID_STEP, lon_step=GRID_STEP,
        )  # fmt: skip
    except aq.CoverageError as exc:
        raise aq.ValidationError(f"{request.id}: {exc}") from exc
    checks["coverage"] = report
    checks["variables"] = sorted(expected.values())
    return ds, checks


def write_outputs(ds, request: Request, root: Path, compress: int | None, checks: dict):
    """Split a validated answer into one file per variable and month, with sidecars."""
    paths = []
    history = ds.attrs.get("history", "")
    for (var, first, last), path in request.outputs(root).items():
        one = ds[[var]]
        if not request.invariant:
            one = one.sel(time=slice(f"{first}", f"{last}T23:59:59"))
        note = f"{aq.utc_now()} split from CDS request {request.id} by download_era5.py"
        one.attrs = {
            **ds.attrs,
            "source": SOURCE,
            "history": f"{history}\n{note}".strip(),
            "era5_expver_status": checks["expver_status"],
            "era5_expver_values": " ".join(checks["expver_values"]) or "absent",
        }
        aq.write_netcdf_atomic(aq.with_layout(one, compress), path)
        _sidecar(path, request, checks)
        paths.append(path)
    return paths


def _sidecar(path: Path, request: Request, checks: dict) -> None:
    aq.write_sidecar(
        path, request.body, dataset=DATASET, request_id=request.id,
        expver_status=checks["expver_status"], catalogue=CATALOGUE_VERSION,
    )  # fmt: skip


def reencode_pieces(store: Store, requests, compress: int | None) -> list[str]:
    """Rewrite the files of validated pieces stored with another encoding than asked
    (``--compress``, the default, or ``--no-compress``).

    The values do not change (same ``content_sha256``, checked); the sidecars and the ledger
    get the new SHA-256, and products built from these pieces are rewritten in place by the
    next ``--daily`` (``Ledger.refresh_product``). Nothing is downloaded again. Returns the
    ids of the pieces rewritten.
    """
    done = []
    for request in requests:
        checks = (store.ledger.get(request.id) or {}).get("checks")

        def sidecar(path: Path, request=request, checks=checks) -> None:
            _sidecar(path, request, checks)

        if store.ledger.reencode_piece(request.id, compress, sidecar):
            done.append(request.id)
    return done


# --- Download ----------------------------------------------------------------------------


@dataclass
class Store:
    """The folder of one box: ledger, raw answers and the files of every variable."""

    root: Path
    ledger: aq.Ledger = field(init=False)

    def __post_init__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = aq.Ledger(self.root / "ledger.json")

    def raw(self, request: Request) -> Path:
        return self.root / "raw" / f"{request.id}.download"


def download(
    requests: list[Request],
    store: Store,
    box: aq.Box,
    *,
    policy: str,
    client_factory,
    workers: int = 4,
    retries: int = 3,
    keep_failed: bool = False,
    expver: ExpverPolicy | None = None,
    compress: int | None = None,
    sleep=None,
) -> aq.RunSummary:
    """Submit every request not yet validated; returns the run summary (never raises)."""
    expver = expver or ExpverPolicy()
    local = threading.local()

    def client():
        if not hasattr(local, "client"):  # one CDS client per worker thread
            local.client = client_factory()
        return local.client

    def task(request: Request):
        def fetch(raw: Path):
            client().retrieve(DATASET, request.body, str(raw))

        def finish(raw: Path):
            ds, checks = validate(raw, request, box, policy, expver)
            return write_outputs(ds, request, store.root, compress, checks), checks

        def run():
            extra = {} if sleep is None else {"sleep": sleep}
            entry = store.ledger.get(request.id)
            recorded = (entry.get("checks") or {}).get("expver_status") or "unknown"
            if entry["status"] == "validated" and not expver.allows(request, recorded):
                store.ledger.update(  # fetch again: the data may be final by now
                    request.id, "planned", error=f"expver {recorded} refused by the current policy"
                )
            entry = aq.acquire(
                store.ledger, request.id, fetch=fetch, finish=finish, raw=store.raw(request),
                keep_failed=keep_failed, retries=retries, **extra,
            )  # fmt: skip
            print(f"{request.id}: {entry['status']}", flush=True)
            return entry

        return run

    for request in requests:
        store.ledger.plan(request.id, request.body, store.raw(request))
    outcomes = aq.run_parallel({r.id: task(r) for r in requests}, workers)
    return aq.RunSummary.from_ledger(store.ledger, [r.id for r in requests], outcomes)


# --- Daily products ----------------------------------------------------------------------


def plan_inputs(store: Store, requests: list, expver: ExpverPolicy | None = None) -> dict:
    """``{request id: sha256}`` of the plan, refusing pieces not validated or not allowed.

    The current ``expver`` policy applies to pieces downloaded under another one.
    """
    expver = expver or ExpverPolicy()
    missing = [r.id for r in requests if not store.ledger.reusable(r.id)]
    if missing:
        raise SystemExit(
            f"{len(missing)} requests of this plan are not downloaded and validated "
            f"(first: {missing[0]}); run --download with the same options first."
        )
    refused = [
        r.id
        for r in requests
        if not expver.allows(r, store.ledger.get(r.id)["checks"].get("expver_status", "unknown"))
    ]
    if refused:
        raise SystemExit(
            f"{len(refused)} pieces are preliminary or of unknown status, refused by the "
            f"current expver policy (first: {refused[0]}); see --allow-era5t"
        )
    return {r.id: store.ledger.get(r.id)["piece"]["sha256"] for r in requests}


def expver_attrs(store: Store, ids) -> dict:
    """``era5_expver_status`` of a product: ``final`` only if every piece is final."""
    by_status: dict[str, list[str]] = {}
    for i in ids:
        status = (store.ledger.get(i).get("checks") or {}).get("expver_status", "unknown")
        by_status.setdefault(status, []).append(i)
    marks = [s for s in ("preliminary", "unknown") if s in by_status]
    attrs = {"era5_expver_status": "+".join(marks) if marks else "final"}
    for mark in marks:
        attrs[f"era5_{mark}_pieces"] = " ".join(by_status[mark])
    return attrs


def hourly(store: Store, requests: list[Request], var: str, first: datetime, last: datetime):
    """Hourly ``var`` from ``first`` to ``last`` (UTC), read from the files of the plan only.

    Only the hours of the window are read from each file, so memory holds the window.
    """
    import numpy as np
    import xarray as xr

    pieces: dict[str, list] = {}
    for request in requests:
        if var not in request.variables:
            continue
        for (name, a, b), path in request.outputs(store.root).items():
            if name != var:
                continue
            span_a = datetime.combine(a, datetime.min.time()) + timedelta(hours=request.hours[0])
            span_b = datetime.combine(b, datetime.min.time()) + timedelta(hours=request.hours[-1])
            if span_b >= first and span_a <= last:
                with xr.open_dataset(path, engine="h5netcdf") as opened:
                    window = opened.sel(time=slice(first, last)).load()
                    pieces.setdefault(request.part, []).append(window)
    if not pieces:
        raise SystemExit(f"no planned file holds {var} between {first} and {last}")
    wanted = np.arange(
        np.datetime64(first, "s"),
        np.datetime64(last, "s") + np.timedelta64(1, "s"),
        np.timedelta64(3600, "s"),
    )
    parts = []
    for label in sorted(pieces, key=PART_ORDER.index):
        series = xr.concat(pieces[label], dim="time").sortby("time").sel(time=slice(first, last))
        got = series["time"].values.astype("datetime64[s]")
        if got.size != wanted.size or not np.array_equal(got, wanted):
            raise SystemExit(
                f"{var}: the planned files give {got.size} hours between {first} and {last}, "
                f"expected {wanted.size} (missing or duplicated hours)"
            )
        parts.append(series)
    return join_parts(parts)


PART_ORDER = ("", "w", "e")


def join_parts(parts):
    """Join the west and east parts of a box along longitude, keeping stored longitudes.

    Both parts are put in west-to-east order whatever their stored order, and a meridian
    present in both (+-180) is kept once, from the west part. Longitudes are then not
    monotonic (``..., 179.75, 180, -179.75, ...``), as in ``download_sst_cci.py``.
    """
    import numpy as np
    import xarray as xr

    if len(parts) == 1:
        return parts[0]
    west, east = parts
    lon_w = np.asarray(west["longitude"].values, dtype=float)
    origin = np.min(np.where(lon_w > -180, lon_w, lon_w + 360))  # -180 is the 180 meridian
    try:
        both = xr.concat([west, east], dim="longitude", join="exact")
    except ValueError as exc:
        raise SystemExit(f"the two parts of the box do not share their grid: {exc}") from None
    key = np.round((np.asarray(both["longitude"].values, dtype=float) - origin) % 360, 6)
    order = np.argsort(key, kind="stable")  # the west copy of a shared meridian comes first
    keep = order[np.concatenate([[True], np.diff(key[order]) > 0])]
    return both.isel(longitude=keep)


def invariant(store: Store, requests, var: str):
    """An invariant field (``lsm``, ``z``) of the plan, its parts joined."""
    import xarray as xr

    parts = []
    for request in sorted(requests, key=lambda r: PART_ORDER.index(r.part)):
        if request.variables == (var,) and request.invariant:
            (path,) = request.outputs(store.root).values()
            with xr.open_dataset(path, engine="h5netcdf") as opened:
                parts.append(opened.load())
    if not parts:
        raise SystemExit(f"{var} is not in the plan")
    return join_parts(parts)


#: Blocks of local days for the daily products (``--daily-block``).
DAILY_BLOCKS = ("month", "year")


def _blocks(start: date, end: date, period: str = "month"):
    """Blocks of local days ``start..end``, by calendar month (default) or year.

    Each block is read, aggregated and written before the next (``aq.BlockWriter``), so
    memory holds the hours of one block of one variable plus its daily values: a month of
    0.25 degree data over 260 000 cells is 0.76 GB per variable in float32, a year 9.1 GB.
    """
    if period not in DAILY_BLOCKS:
        raise ValueError(f"unknown block {period!r}; one of {', '.join(DAILY_BLOCKS)}")
    a = start
    while a <= end:
        if period == "year":
            b = date(a.year, 12, 31)
        else:
            b = date(a.year, a.month, calendar.monthrange(a.year, a.month)[1])
        b = min(b, end)
        yield a, b
        a = b + timedelta(1)


def registered_daily_blocks(
    store: Store, requests, start: date, end: date, day_definition: str, contrast: bool,
    period: str = "month",
):  # fmt: skip
    """TX/TN (and the t2m contrast) per local day, block by block; the land-sea mask
    comes with the first block."""
    import xarray as xr

    import xhw

    stats = [("mx2t", "max", "tx"), ("mn2t", "min", "tn")]
    if contrast:
        stats += [("t2m", "max", "tx_t2m"), ("t2m", "min", "tn_t2m")]
    for i, (a, b) in enumerate(_blocks(start, end, period)):
        out = []
        for var, stat, name in stats:
            series = hourly(store, requests, var, *utc_window([var], a, b, day_definition))
            out.append(
                xhw.aggregate_daily(
                    series,
                    variable=var,
                    statistic=stat,
                    sampling=CATALOGUE[var].sampling,
                    input_clock="UTC",
                    day_definition=day_definition,
                    name=name,
                )  # fmt: skip
            )
        days = xr.merge(out, compat="override", join="outer")  # halo days differ per variable
        days = days.sel(time=slice(f"{a}", f"{b}"))
        if i == 0:
            lsm = invariant(store, requests, "lsm")["lsm"]
            keep = ("latitude", "longitude")
            days["lsm"] = lsm.drop_vars([c for c in lsm.coords if c not in keep])
        print(f"{a}..{b}: daily aggregation done", flush=True)
        yield days


def variable_daily_blocks(
    store: Store, requests, var: str, stat: str, start: date, end: date, day_definition: str,
    period: str = "month",
):  # fmt: skip
    """One statistic of one variable per local day, block by block, with the conversions
    of revision 2, V3."""
    for a, b in _blocks(start, end, period):
        yield _variable_daily(store, requests, var, stat, a, b, day_definition)


def _variable_daily(store: Store, requests, var, stat, a: date, b: date, day_definition):
    import xhw

    spec = CATALOGUE[var]
    series = hourly(store, requests, var, *utc_window([var], a, b, day_definition))
    days = xhw.aggregate_daily(
        series, variable=var, statistic="mean" if stat == "total" else stat,
        sampling=spec.sampling, input_clock="UTC", day_definition=day_definition,
    ).sel(time=slice(f"{a}", f"{b}"))  # fmt: skip
    if spec.kind == "accumulated":
        attrs = dict(days[var].attrs)
        if stat == "total":  # a complete day holds 24 hourly accumulations ending in it
            days[var] = days[var] * 24
            attrs["conversion"] = "daily total = mean of 24 hourly accumulations x 24"
        else:
            days[var] = days[var] / 3600.0
            attrs.update(
                units="W m-2",
                conversion="mean hourly accumulation (J m-2) / 3600 s; positive downwards",
            )
        days[var].attrs = attrs
    days.attrs = {"source": SOURCE}
    return days


def product_inputs(inputs, compress: int | None, recipe: dict | None = None) -> dict:
    """The identity of a product: its pieces, its ``recipe`` and its physical encoding.

    The recipe (day definition, variables, statistics, conversions, policies) decides the
    values; the encoding (``--compress``) only the layout, but a product written with
    another one is rewritten in place (``Ledger.refresh_product``), without downloading again.
    """
    return {
        **inputs,
        "recipe": aq.request_sha256(recipe or {}),
        "encoding": aq.encoding_label(compress),
    }


def product_status(store: Store, name: str, path: Path, inputs, compress: int | None) -> str:
    """``current``, ``reencoded``, ``missing`` or ``stale``; a file that is not the recorded
    one stops. A product that differs only in its encoding is rewritten in place first.
    """
    status = store.ledger.refresh_product(name, path, inputs, compress)
    if status in ("changed", "unrecorded"):
        raise SystemExit(
            f"{path.name} exists but is not the recorded product ({status}); move it away "
            "or use another --data-dir."
        )
    return status


def write_daily(
    store: Store, blocks, path: Path, inputs, compress: int | None, recipe: dict | None = None,
    attrs: dict | None = None,
) -> str:  # fmt: skip
    """Write a daily product block by block unless an identical one is current.

    ``blocks`` is an iterable of datasets that follow each other in time, such as the
    generators ``registered_daily_blocks`` and ``variable_daily_blocks``: it is consumed
    only if the product must be (re)built, and one block at a time. ``attrs`` are the global
    attributes of the file. Returns ``current``, ``written`` or ``rebuilt``.
    """
    inputs = product_inputs(inputs, compress, recipe)
    status = product_status(store, path.name, path, inputs, compress)
    if status in aq.UP_TO_DATE:
        return status
    with aq.BlockWriter(path, compress=compress) as writer:
        for block in blocks:
            if attrs is not None:
                block.attrs = dict(attrs)
            writer.append(block)
        writer.commit()
    store.ledger.record_product(path.name, path, inputs, recipe=recipe or {})
    return "written" if status == "missing" else "rebuilt"


def write_derived(
    store: Store, blocks, path_of, recipe_of, inputs, compress: int | None, attrs: dict
) -> dict[Path, str]:
    """The files of one derived product, written together from the same blocks.

    Each file holds one daily variable (``path_of(var)``, ``recipe_of(var)``) with the
    sample counts and indicators of its inputs. The first block names the variables; if
    every file is current, no further block is computed. Returns ``{path: status}``.
    """
    from contextlib import ExitStack

    results: dict[Path, str] = {}
    todo: dict | None = None
    counts: list = []
    with ExitStack() as stack:
        for block in blocks:
            if todo is None:
                ends = ("_n_samples", "_fraction")
                counts = [v for v in block.data_vars if str(v).endswith(ends)]
                todo = {}
                for var in [v for v in block.data_vars if v not in counts]:
                    path, recipe = path_of(var), recipe_of(var)
                    identity = product_inputs(inputs, compress, recipe)
                    results[path] = product_status(store, path.name, path, identity, compress)
                    if results[path] not in aq.UP_TO_DATE:
                        writer = stack.enter_context(aq.BlockWriter(path, compress=compress))
                        todo[var] = (path, identity, recipe, writer)
                if not todo:
                    break
            for var, (*_, writer) in todo.items():
                one = block[[var, *counts]]  # each file keeps the sample counts of its inputs
                one.attrs = dict(attrs)
                writer.append(one)
        for path, identity, recipe, writer in (todo or {}).values():
            writer.commit()
            store.ledger.record_product(path.name, path, identity, recipe=recipe)
            results[path] = "written" if results[path] == "missing" else "rebuilt"
    return results


def _span(requests, var: str, year: int) -> tuple[datetime, datetime] | None:
    """First and last planned hour of ``var`` within UTC ``year``."""
    stamps = [
        t
        for r in requests
        if var in r.variables and r.start.year <= year <= r.end.year
        for t in (
            datetime.combine(r.start, datetime.min.time()) + timedelta(hours=r.hours[0]),
            datetime.combine(r.end, datetime.min.time()) + timedelta(hours=r.hours[-1]),
        )
    ]
    if not stamps:
        return None
    return max(min(stamps), datetime(year, 1, 1)), min(max(stamps), datetime(year, 12, 31, 23))


def _hour_blocks(first: datetime, last: datetime):
    """Hourly ``first..last`` cut at the start of each UTC month."""
    a = first
    while a <= last:
        following = date(a.year + a.month // 12, a.month % 12 + 1, 1)
        b = min(last, datetime.combine(following, datetime.min.time()) - timedelta(hours=1))
        yield a, b
        a = b + timedelta(hours=1)


def consolidate_years(
    store: Store, requests, target: Path, variables, inputs, compress, marks=None
):
    """One hourly file per UTC year with the variables together, as the registered input.

    Only for compatibility with ``hourly_files_sha256`` of the registered manifest (revision
    2, D4): the per-variable monthly files are the primary storage. Each year holds the
    hours planned for every variable; the time dimension is ``valid_time``, as in the CDS
    answers that the registered files kept. A year file is rebuilt when its pieces or its
    encoding change, and is written month by month: memory holds one month of the variables.
    """
    import xarray as xr

    timed = [v for v in variables if CATALOGUE[v].kind != "invariant"]
    spans = [r for r in requests if not r.invariant]
    years = sorted({y for r in spans for y in range(r.start.year, r.end.year + 1)})
    paths = []
    for year in years:
        ranges = [_span(spans, v, year) for v in timed]
        if any(r is None for r in ranges):
            continue
        first, last = max(r[0] for r in ranges), min(r[1] for r in ranges)
        path = target / f"{year}.nc"
        used = {r.id: inputs[r.id] for r in spans if r.start.year <= year <= r.end.year}
        used["encoding"] = aq.encoding_label(compress)
        status = product_status(store, f"hourly/{path.name}", path, used, compress)
        if status not in aq.UP_TO_DATE:
            with aq.BlockWriter(path, dim="valid_time", compress=compress) as writer:
                for a, b in _hour_blocks(first, last):
                    ds = xr.merge([hourly(store, requests, v, a, b) for v in timed])
                    ds = ds.rename({"time": "valid_time"})
                    ds.attrs = {"source": SOURCE, **(marks or {})}
                    writer.append(ds)
                writer.commit()
            store.ledger.record_product(f"hourly/{path.name}", path, used)
        hours = int((last - first) / timedelta(hours=1)) + 1
        print(f"{path.name}: {hours} hours ({status})", flush=True)
        paths.append(path)
    if any(r.variables == ("lsm",) for r in requests):
        lsm = aq.with_layout(invariant(store, requests, "lsm"), compress)
        aq.write_netcdf_atomic(lsm, target / "lsm.nc")
        paths.append(target / "lsm.nc")
    return paths


# --- Command line ------------------------------------------------------------------------


def _key(region: str | None, area) -> str:
    if area and isinstance(area[0], Part):
        area = area[0].area if len(area) == 1 else [part.area for part in area]
    return region or "box_" + aq.request_id({"area": area}, length=8)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    aq.add_box_arguments(parser)
    aq.add_period_arguments(parser, "1991-01-01", "2025-12-31")
    parser.add_argument("--years", help="shortcut for --start/--end, e.g. 1991-2025")
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--day-definition", default=DAY_DEFINITION)
    parser.add_argument("--chunk-period", choices=list(aq.PERIODS), default="month")
    parser.add_argument(
        "--request-grouping", choices=("variable", "steptype", "all"), default="steptype",
        help="variables per request: one, one GRIB step type (default) or all",
    )  # fmt: skip
    parser.add_argument("--group-all", action="store_true", help="same as --request-grouping all")
    parser.add_argument(
        "--boundary-hours", type=int, help="request hours 00..N-1 after the last day instead"
    )
    parser.add_argument("--margin", type=float, default=0.9, help="share of the CDS cost limit")
    aq.add_run_arguments(parser)
    parser.add_argument("--allow-era5t", action="store_true", help="accept preliminary ERA5T")
    parser.add_argument("--accept-unknown-expver-before", type=date.fromisoformat)
    parser.add_argument("--expver-justification", help="required with the previous option")
    aq.compression_arguments(parser)
    parser.add_argument(
        "--daily-block", choices=DAILY_BLOCKS, default="month",
        help="local days read, aggregated and written at a time (memory: one block of one "
        "variable); the products do not depend on it",
    )  # fmt: skip
    parser.add_argument(
        "--derived",
        nargs="+",
        default=[],
        choices=list(_derived.SURFACE),
        help="derived daily products (examples/_derived.py); their inputs must be in --variables",
    )
    parser.add_argument(
        "--no-contrast", dest="contrast", action="store_false", help="omit tx_t2m and tn_t2m"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="submit the requests")
    group.add_argument("--daily", action="store_true", help="build the local-day products")
    group.add_argument(
        "--consolidate-years",
        action="store_true",
        help="also write one hourly file per year (era5_<region>_hourly/), as registered",
    )
    return parser


def main(argv=None, client_factory=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.years:
        try:
            first, last = map(int, args.years.split("-"))
            args.start, args.end = date(first, 1, 1), date(last, 12, 31)
        except ValueError:
            parser.error("--years must be an inclusive range, for example 1991-2025")
    try:
        start, end = aq.period_from_args(args, first=date(1940, 1, 1))
        missing = sorted(_derived.required(args.derived) - set(args.variables))
        if missing:
            raise ValueError(f"--derived {' '.join(args.derived)} needs --variables {missing}")
        extra = {}
        if "tmean_9to9" in args.derived:
            definition, shift = _derived.shifted_day(args.day_definition, 9)
            days = timedelta(shift)
            for var in ("mx2t", "mn2t"):
                extra[var] = utc_window([var], start + days, end + days, definition)
        for name in args.derived:  # the hours each input needs at the product's sampling
            spec = _derived.SURFACE[name]
            window = hour_window(spec.sampling, start, end, args.day_definition)
            for var in spec.inputs:
                first, last = extra.get(var, window)
                extra[var] = (min(first, window[0]), max(last, window[1]))
        box, region = aq.box_from_args(args)
        area = box_parts(box, args.box_policy)
        grouping = "all" if args.group_all else args.request_grouping
        requests = plan(
            area, args.variables, start, end, day_definition=args.day_definition,
            chunk_period=args.chunk_period, grouping=grouping,
            boundary_hours=args.boundary_hours, margin=args.margin, extra_windows=extra,
        )  # fmt: skip
    except (ValueError, aq.CostLimitError) as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.accept_unknown_expver_before and not args.expver_justification:
        parser.error("--accept-unknown-expver-before needs --expver-justification")
    compress = aq.compression(args)
    key = _key(region, area)
    areas = [part.area for part in area]
    store_dir = args.data_dir / "era5_single" / key
    record = {
        "dataset": DATASET,
        "region": region,
        "box": box.as_dict(),
        "box_policy": args.box_policy,
        "area_nwse": areas[0] if len(areas) == 1 else areas,
        "start": f"{start}",
        "end": f"{end}",
        "variables": list(args.variables),
        "day_definition": args.day_definition,
        "chunk_period": args.chunk_period,
        "request_grouping": grouping,
        "boundary_hours": args.boundary_hours,
        "derived": list(args.derived),
        "catalogue": CATALOGUE_VERSION,
    }
    expver = ExpverPolicy(
        args.allow_era5t, args.accept_unknown_expver_before, args.expver_justification
    )
    policy = expver.record()

    if not (args.download or args.daily or args.consolidate_years):
        for request in requests:
            print(request.id, json.dumps(request.body))
        print(
            f"Print only: {len(requests)} requests (CDS cost {sum(r.cost for r in requests)}), "
            f"area {record['area_nwse']} for the box {box.as_dict()} "
            f"with --box-policy {args.box_policy}; "
            f"files in {store_dir}. Add --download to submit them."
        )
        return 0

    store = Store(store_dir)
    ids = [r.id for r in requests]
    if args.download:
        if client_factory is None:
            try:
                import cdsapi
            except ImportError as exc:
                raise ImportError(
                    "Install examples/requirements-download.txt in an acquisition environment; "
                    "see the README. It is not an xHW runtime dependency."
                ) from exc

            def client_factory():
                return cdsapi.Client(quiet=True)

        summary = download(
            requests, store, box, policy=args.box_policy, client_factory=client_factory,
            workers=args.workers, retries=args.retries, keep_failed=args.keep_failed,
            expver=expver, compress=compress,
        )  # fmt: skip
        summary.report()
        rewritten = reencode_pieces(store, requests, compress)
        if rewritten:
            print(f"{len(rewritten)} pieces rewritten with {aq.encoding_label(compress)}")
        valid = [i for i in ids if summary.statuses[i] == "validated"]
        files = {}
        for i in valid:
            piece = store.ledger.get(i)["piece"]
            for stored in piece.get("files") or [piece["path"]]:
                files[stored] = store.ledger.resolve(stored)
        manifest = aq.build_manifest(
            source="era5_single_levels", request=record, files=files, pieces=ids,
            coverage=(
                store.ledger.get(valid[0])["checks"]["coverage"]
                if valid and len(area) == 1
                else None
            ),
            coverage_by_request={i: store.ledger.get(i)["checks"]["coverage"] for i in valid},
            retrieval_window=store.ledger.retrieval_window(ids),
            requests={r.id: r.body for r in requests}, statuses=summary.statuses,
            expver={i: store.ledger.get(i)["checks"].get("expver_status") for i in valid},
            **policy,
        )  # fmt: skip
        path = aq.write_manifest(store.root, "era5_download", manifest)
        print(f"Manifest {path.name}. Next: --daily builds the local-day products.")
        return summary.exit_code

    inputs = plan_inputs(store, requests, expver)
    marks = expver_attrs(store, ids)
    if args.consolidate_years:
        target = args.data_dir / f"era5_{key}_hourly"
        paths = consolidate_years(store, requests, target, args.variables, inputs, compress, marks)
        manifest = aq.build_manifest(
            source="era5_single_levels", request=record, files={p.name: p for p in paths},
            pieces=ids, retrieval_window=store.ledger.retrieval_window(ids), expver=marks,
            **policy,
        )  # fmt: skip
        print(f"Manifest {aq.write_manifest(target, 'era5_hourly', manifest).name}")
        return 0
    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31):
        stamp = f"{start.year}_{end.year}"
    products = {}
    base = {
        "day_definition": args.day_definition,
        "catalogue": CATALOGUE_VERSION,
        **policy,
    }
    combined = set(DEFAULT_VARIABLES) <= set(args.variables)
    block = args.daily_block
    attrs = {"source": SOURCE, **marks}
    if combined:
        blocks = registered_daily_blocks(
            store, requests, start, end, args.day_definition, args.contrast, block
        )
        path = args.data_dir / f"era5_{key}_daily_{stamp}.nc"
        recipe = {**base, "product": "tx/tn", "contrast": args.contrast}
        status = write_daily(store, blocks, path, inputs, compress, recipe, attrs)
        print(f"{path.name}: {status}")
        products[path.name] = path
    for var in args.variables:
        if (combined and var in DEFAULT_VARIABLES) or CATALOGUE[var].kind == "invariant":
            continue
        for stat in CATALOGUE[var].daily:
            blocks = variable_daily_blocks(
                store, requests, var, stat, start, end, args.day_definition, block
            )
            path = args.data_dir / f"era5_{key}_daily_{var}_{stat}_{stamp}.nc"
            recipe = {**base, "product": var, "statistic": stat}
            status = write_daily(store, blocks, path, inputs, compress, recipe, attrs)
            print(f"{path.name}: {status}")
            products[path.name] = path

    def get_hourly(var, sampling, a, b, definition):
        # the sampling of the derived product: UTCI pairs instantaneous t2m with radiation
        return hourly(store, requests, var, *hour_window(sampling, a, b, definition))

    for name in args.derived:
        blocks = (
            _derived.daily(name, get_hourly, a, b, args.day_definition)
            for a, b in _blocks(start, end, block)
        )
        written = write_derived(
            store, blocks, lambda var: args.data_dir / f"era5_{key}_daily_{var}_{stamp}.nc",
            lambda var, name=name: {**base, "product": var, "derived": name}, inputs, compress,
            {"source": SOURCE, "catalogue": CATALOGUE_VERSION, **marks},
        )  # fmt: skip
        for path, status in written.items():
            print(f"{path.name}: {status}")
            products[path.name] = path
    # the box check of the joined grid (both parts of a box crossing the antimeridian)
    grid = (
        invariant(store, requests, "lsm")
        if any(r.variables == ("lsm",) for r in requests)
        else None
    )
    joined_coverage = (
        aq.coverage_report(
            box,
            grid["latitude"].values,
            grid["longitude"].values,
            args.box_policy,
            lat_step=GRID_STEP,
            lon_step=GRID_STEP,
        )  # fmt: skip
        if grid is not None
        else None
    )
    manifest = aq.build_manifest(
        source="era5_single_levels", request=record, files=products, pieces=ids,
        coverage=joined_coverage,
        retrieval_window=store.ledger.retrieval_window(ids), fingerprint=True,
        expver={i: store.ledger.get(i)["checks"].get("expver_status") for i in ids},
        **policy,
    )  # fmt: skip
    registered_name = f"era5_{key}_daily_{stamp}.nc"
    if region == "peru" and registered_name in manifest["files"]:
        manifest["matches_registered_input"] = aq.matches_registered(
            manifest["files"][registered_name], registered("era5")
        )
    path = aq.write_manifest(args.data_dir, "era5_daily", manifest)
    print(f"Manifest {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
