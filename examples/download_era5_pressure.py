# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional ERA5 pressure-levels acquisition and daily products; default prints the plan.

Requests hourly upper-air variables from the CDS dataset ``reanalysis-era5-pressure-levels``
(same account, key and licence steps as ``download_era5.py``). Kept apart from the
single-levels script: the levels add a dimension, the cost limit is half (60 000) and the
daily values need a terrain mask.

    python examples/download_era5_pressure.py --data-dir data --years 1991-2025       # plan
    python examples/download_era5_pressure.py --data-dir data --years 1991-2025 \\
        --variables temperature geopotential --levels 1000 925 850 700 500 --download
    python examples/download_era5_pressure.py --data-dir data --years 1991-2025 --daily \\
        --surface-pressure data/era5_single/peru

Defaults (revision 1, decisions 1, 4 and 5): ``temperature`` at 850, 700 and 500 hPa over
the Peru box; one request per variable, month and group of levels under the cost limit;
daily mean, maximum and minimum per level for local days at UTC-05:00.

ERA5 gives values at every level, also below the terrain (44 % of the Peru box at 1000 hPa
on 2023-03-01 00 UTC; ECMWF documents them as extrapolations). ``--daily`` therefore masks
every hour before aggregating: a sample counts only where the surface pressure is finite
and ``100 x level_hPa <= sp_Pa``; elsewhere it becomes missing, not "no event". The hourly
surface pressure comes from ``download_era5.py --variables sp`` over the same box and
hours (``--surface-pressure`` is its folder); without it, no daily file is built. Each
level is a separate 2-D surface for xHW; detection on levels is deferred (decision 2).

Requests, ledger, damage policy, ERA5T policy, box policy and manifest work as in
``download_era5.py``. Design after bonino2024xt (request per variable and month, sidecar,
bounded pool), rewritten, not copied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import _acquire as aq
import _derived
import download_era5 as single

DATASET = "reanalysis-era5-pressure-levels"
GRID_STEP = single.GRID_STEP
DEFAULT_LEVELS = ("850", "700", "500")
STATISTICS = ("mean", "max", "min")
SOURCE = f"Copernicus Climate Data Store {DATASET} (ERA5, Copernicus licence)"


@dataclass(frozen=True)
class Variable:
    cds: str
    units: str  # normalised native units


#: Short names and units seen in CDS answers on 2026-11-03 (revision 1, section 3 and L10).
CATALOGUE = {
    "t": Variable("temperature", "K"),
    "z": Variable("geopotential", "m2 s-2"),
    "u": Variable("u_component_of_wind", "m s-1"),
    "v": Variable("v_component_of_wind", "m s-1"),
    "r": Variable("relative_humidity", "%"),
    "q": Variable("specific_humidity", "kg kg-1"),
}
BY_CDS = {spec.cds: name for name, spec in CATALOGUE.items()}

#: Derived products per level (revision 1, L10), from masked hourly values.
DERIVED = {
    "gh": (("z",), "m", "z / 9.80665"),
    "ws": (("u", "v"), "m s-1", "sqrt(u**2 + v**2)"),
    "theta": (("t",), "K", "T (1000 / p) ** 0.285691, p in hPa"),
    "thickness_1000_500": (("z",), "m", "(z500 - z1000) / 9.80665 where 1000 hPa is above ground"),
}


def short_name(name: str) -> str:
    if name in CATALOGUE:
        return name
    if name in BY_CDS:
        return BY_CDS[name]
    raise ValueError(f"unknown pressure-level variable {name!r}; known: {', '.join(BY_CDS)}")


def levels_label(levels) -> str:
    """``p850-700-500``, or a short hash for more than four levels (revision 1, L2)."""
    if len(levels) <= 4:
        return "p" + "-".join(levels)
    return "p" + aq.request_id({"levels": list(levels)}, length=6) + f"x{len(levels)}"


# --- Plan --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    id: str
    body: dict
    variable: str
    levels: tuple[str, ...]
    start: date
    end: date
    hours: tuple[int, ...]
    part: str = ""  # "w" or "e" for the two parts of a box crossing the antimeridian
    box: aq.Box | None = None  # the box of this part (None: the whole requested box)

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.variable,)

    @property
    def invariant(self) -> bool:
        return False

    @property
    def cost(self) -> int:
        days = (self.end - self.start).days + 1
        return aq.cds_cost(1, len(self.hours), days, len(self.levels))

    def stamps(self) -> list[datetime]:
        midnight = datetime.combine(self.start, datetime.min.time())
        days = (self.end - self.start).days + 1
        return [midnight + timedelta(days=d, hours=h) for d in range(days) for h in self.hours]

    def output(self, root: Path) -> Path:
        full = self.hours == tuple(range(24))
        month_end = (self.start.replace(day=28) + timedelta(4)).replace(day=1) - timedelta(1)
        if full and self.start.day == 1 and self.end == month_end:
            span = f"{self.start:%Y-%m}"
        else:
            span = f"{self.start:%Y-%m-%d}_{self.end:%Y-%m-%d}"
            if not full:
                span += f"_h{self.hours[0]:02d}-{self.hours[-1]:02d}"
        suffix = f"_{self.part}" if self.part else ""
        name = f"{self.variable}_{levels_label(self.levels)}_{span}{suffix}.nc"
        return root / self.variable / f"{self.start:%Y}" / name


def utc_window(start: date, end: date, day_definition: str) -> tuple[datetime, datetime]:
    """Hours that local days ``start..end`` need for instantaneous values (revision 1, L8)."""
    return single.hour_window("instantaneous", start, end, day_definition)


def _request(area, var, levels, start, end, hours, part=None) -> Request:
    body = {
        "product_type": ["reanalysis"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
        "variable": [CATALOGUE[var].cds],
        "pressure_level": list(levels),
        **aq.cds_date_fields(start, end),
        "time": [f"{h:02d}:00" for h in hours],
    }
    label = f"pl_{start:%Y%m%d}_{var}_{levels_label(levels)}"
    if part is not None and part.label:
        label += f"_{part.label}"
    rid = aq.request_id(body, label)
    if part is None:
        return Request(rid, body, var, tuple(levels), start, end, tuple(hours))
    return Request(rid, body, var, tuple(levels), start, end, tuple(hours), part.label, part.box)


def plan(
    area,
    variables,
    levels,
    start: date,
    end: date,
    *,
    day_definition: str = single.DAY_DEFINITION,
    level_group: int | None = None,
    margin: float = 0.9,
) -> list[Request]:
    """One request per variable, month (or block of days) and group of levels (L1, L9).

    Plus the hours of the neighbouring UTC days that the local days need (L8). ``area`` is
    a CDS ``[N, W, S, E]`` or the parts of ``download_era5.box_parts`` for a box crossing
    the antimeridian: every request is then made once per part.
    """
    parts = single._parts(area)
    variables = [short_name(v) for v in variables]
    levels = [str(int(float(level))) for level in levels]
    if len(set(levels)) != len(levels):
        raise ValueError("repeated pressure levels")
    first, last = utc_window(start, end, day_definition)
    day_start = datetime.combine(start, datetime.min.time())
    day_end = datetime.combine(end, datetime.min.time()) + timedelta(hours=23)
    before = [h for h in range(24) if day_start - timedelta(hours=24 - h) >= first]
    after = [h for h in range(24) if day_end + timedelta(hours=h + 1) <= last]
    if level_group is None:  # the same groups every month, sized for 31 days
        budget = int(aq.CDS_COST_LIMITS[DATASET] * margin)
        level_group = max(1, min(len(levels), budget // aq.cds_cost(1, 24, 31)))
    out = []
    for var in variables:
        chunks = aq.plan_cds_chunks(
            start, end, dataset=DATASET, n_variables=1, levels=levels, margin=margin,
            level_group=level_group,
        )  # fmt: skip
        groups = sorted({c.levels for c in chunks}, key=lambda g: levels.index(g[0]))
        for part in parts:
            for chunk in chunks:
                out.append(
                    _request(part.area, var, chunk.levels, chunk.start, chunk.end, range(24), part)
                )
            for group in groups:
                for day, hours in ((start - timedelta(1), before), (end + timedelta(1), after)):
                    if hours:
                        out.append(_request(part.area, var, group, day, day, hours, part))
    return out


# --- Validation --------------------------------------------------------------------------


def validate(raw: Path, request: Request, box: aq.Box, policy: str, expver):
    """Variables, units, levels (L4), expver (L5), hours and box of one answer."""
    import numpy as np

    parts = single._open_parts(raw)
    if len(parts) != 1:
        raise aq.ValidationError(f"{request.id}: expected one NetCDF, got {len(parts)} parts")
    ds = parts[0][1]
    ds = ds.rename({"valid_time": "time"}) if "valid_time" in ds.dims else ds
    var = request.variable
    if set(ds.data_vars) != {var}:
        raise aq.ValidationError(f"{request.id}: variables {sorted(ds.data_vars)}, expected {var}")
    got_units = single._units(ds[var].attrs.get("units", ""))
    if got_units != CATALOGUE[var].units:
        raise aq.ValidationError(f"{request.id}: {var} in {got_units!r}")
    if "pressure_level" not in ds.dims:
        raise aq.ValidationError(f"{request.id}: no pressure_level dimension")
    level_units = single._units(ds["pressure_level"].attrs.get("units", ""))
    if level_units not in ("hPa", "millibars"):
        raise aq.ValidationError(f"{request.id}: pressure_level in {level_units!r}, expected hPa")
    values = np.asarray(ds["pressure_level"].values, dtype=float)
    wanted_levels = np.sort(np.asarray(request.levels, dtype=float))
    received = [f"{v:g}" for v in values]
    if (
        not np.all(np.isfinite(values))
        or values.size != wanted_levels.size
        or np.unique(values).size != values.size
        or not np.array_equal(np.sort(values), wanted_levels)
    ):
        raise aq.ValidationError(
            f"{request.id}: levels {received}, requested {list(request.levels)}"
        )
    status, found = single.expver_status(ds["expver"].values if "expver" in ds.variables else None)
    checks = {
        "expver_status": expver.check(request, status, found),
        "expver_values": found,
        "levels": received,
    }
    ds.attrs.update(
        era5_expver_status=checks["expver_status"], era5_expver_values=" ".join(found) or "absent"
    )
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])
    times = ds["time"].values.astype("datetime64[s]")
    wanted = np.array(request.stamps(), dtype="datetime64[s]")
    duplicates = int(times.size - np.unique(times).size)
    if duplicates or times.size != wanted.size or not np.array_equal(np.sort(times), wanted):
        raise aq.ValidationError(f"{request.id}: {times.size} hours, expected {wanted.size}")
    try:
        checks["coverage"] = aq.check_coverage(
            request.box or box, ds["latitude"].values, ds["longitude"].values, policy,
            lat_step=GRID_STEP, lon_step=GRID_STEP,
        )  # fmt: skip
    except aq.CoverageError as exc:
        raise aq.ValidationError(f"{request.id}: {exc}") from exc
    checks["n_times"] = int(times.size)
    return ds.sortby("time"), checks


def write_output(ds, request: Request, root: Path, compress, checks) -> Path:
    path = request.output(root)
    note = f"{aq.utc_now()} written from CDS request {request.id} by download_era5_pressure.py"
    history = ds.attrs.get("history", "")
    ds.attrs = {**ds.attrs, "source": SOURCE, "history": f"{history}\n{note}".strip()}
    var = request.variable
    encoding = {var: {"zlib": True, "complevel": compress}} if compress else None
    aq.write_netcdf_atomic(ds, path, encoding=encoding)
    aq.write_sidecar(
        path, request.body, dataset=DATASET, request_id=request.id,
        expver_status=checks["expver_status"], levels=list(request.levels),
    )  # fmt: skip
    return path


def download(
    requests,
    store,
    box,
    *,
    policy,
    client_factory,
    workers=4,
    retries=3,
    keep_failed=False,
    expver=None,
    compress=None,
    sleep=None,
) -> aq.RunSummary:
    """Submit every request not yet validated; returns the run summary (never raises)."""
    expver = expver or single.ExpverPolicy()
    local = threading.local()

    def client():
        if not hasattr(local, "client"):
            local.client = client_factory()
        return local.client

    def task(request):
        def fetch(raw):
            client().retrieve(DATASET, request.body, str(raw))

        def finish(raw):
            ds, checks = validate(raw, request, box, policy, expver)
            return write_output(ds, request, store.root, compress, checks), checks

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


def hourly(store, requests, var: str, levels, first: datetime, last: datetime):
    """Hourly ``var`` on ``levels`` over ``first..last`` from the planned files only."""
    import numpy as np
    import xarray as xr

    by_part: dict[str, dict[tuple, list]] = {}
    for request in requests:
        if request.variable != var:
            continue
        a = datetime.combine(request.start, datetime.min.time()) + timedelta(hours=request.hours[0])
        b = datetime.combine(request.end, datetime.min.time()) + timedelta(hours=request.hours[-1])
        if b >= first and a <= last:
            with xr.open_dataset(request.output(store.root), engine="h5netcdf") as opened:
                by_group = by_part.setdefault(request.part, {})
                by_group.setdefault(request.levels, []).append(opened.load())
    if not by_part:
        raise SystemExit(f"no planned file holds {var} between {first} and {last}")
    joined = []
    for label in sorted(by_part, key=single.PART_ORDER.index):
        groups = [
            xr.concat(parts, dim="time").sortby("time").sel(time=slice(first, last))
            for parts in by_part[label].values()
        ]
        joined.append(xr.concat(groups, dim="pressure_level", join="exact"))
    series = single.join_parts(joined)
    order = [float(level) for level in levels]
    series = series.sel(pressure_level=order)
    wanted = np.arange(
        np.datetime64(first, "s"), np.datetime64(last, "s") + np.timedelta64(1, "s"),
        np.timedelta64(3600, "s"),
    )  # fmt: skip
    got = series["time"].values.astype("datetime64[s]")
    if got.size != wanted.size or not np.array_equal(got, wanted):
        raise SystemExit(
            f"{var}: {got.size} hours between {first} and {last}, expected {wanted.size}"
        )
    return series


def surface_pressure(folder: Path, first: datetime, last: datetime, expver=None, used=None):
    """Hourly ``sp`` over ``first..last`` from a ``download_era5.py`` store (L7).

    Every file used must have its provenance sidecar with the same SHA-256, and an
    ``expver`` status that the current policy accepts; the hours must be complete and
    unique. ``used`` collects ``{file: (sha256, expver status)}`` for the product identity.
    """
    expver = expver or single.ExpverPolicy()
    import numpy as np
    import xarray as xr

    pieces: dict[str, list] = {}
    for path in sorted((folder / "sp").glob("*/sp_*.nc")):
        side = aq.sidecar_path(path)
        if not side.exists():
            raise SystemExit(f"{path} has no provenance sidecar")
        record = json.loads(side.read_text(encoding="utf-8"))
        if record.get("sha256") != aq.file_sha256(path):
            raise SystemExit(f"{path} does not match its sidecar")
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            times = opened["time"].values
            if times[-1] < np.datetime64(first) or times[0] > np.datetime64(last):
                continue
            status = record.get("expver_status", "unknown")
            span = SimpleNamespace(end=date.fromisoformat(str(times[-1])[:10]))
            if not expver.allows(span, status):
                raise SystemExit(
                    f"{path.name}: surface pressure with expver {status}, refused by the "
                    "current policy (see --allow-era5t)"
                )
            part = re.search(r"_(w|e)\.nc$", path.name)
            pieces.setdefault(part.group(1) if part else "", []).append(opened.load())
            if used is not None:
                used[path.relative_to(folder).as_posix()] = (record["sha256"], status)
    if not pieces:
        raise SystemExit(
            f"no surface pressure in {folder} for {first}..{last}; run download_era5.py "
            "--variables sp over the same box and period first (revision 1, L7)"
        )
    wanted = np.arange(
        np.datetime64(first, "s"), np.datetime64(last, "s") + np.timedelta64(1, "s"),
        np.timedelta64(3600, "s"),
    )  # fmt: skip
    joined = []
    for label in sorted(pieces, key=single.PART_ORDER.index):
        sp = xr.concat(pieces[label], dim="time").sortby("time").sel(time=slice(first, last))
        got = sp["time"].values.astype("datetime64[s]")
        if got.size != wanted.size or not np.array_equal(got, wanted):
            raise SystemExit(
                f"surface pressure: {got.size} hours between {first} and {last}, expected "
                f"{wanted.size} (missing, duplicated or from two downloads)"
            )
        joined.append(sp)
    sp = single.join_parts(joined)
    if single._units(sp["sp"].attrs.get("units", "")) != "Pa":
        raise SystemExit("surface pressure is not in Pa")
    return sp["sp"]


def masked(series, var: str, sp):
    """Values where the level is above the terrain; NaN below it or where ``sp`` is missing."""
    import numpy as np
    import xarray as xr

    try:
        xr.align(series, sp, join="exact", exclude=["pressure_level"])
    except ValueError as exc:
        raise SystemExit(f"surface pressure is on another grid or other hours: {exc}") from None
    above = xr.apply_ufunc(_derived.above_ground, series["pressure_level"], sp)
    out = series.copy()
    out[var] = series[var].where(above)
    below = (~above & np.isfinite(sp)).astype("float32")
    return out, above, below, (~np.isfinite(sp)).astype("float32")


def daily_block(
    store, requests, var, levels, sp_folder, a: date, b: date, day_definition, derived,
    expver=None, used=None,
):  # fmt: skip
    """Masked daily mean/max/min per level for local days ``a..b`` (L6, L8)."""
    import xarray as xr

    import xhw

    first, last = utc_window(a, b, day_definition)
    sp = surface_pressure(sp_folder, first, last, expver, used)
    series, above, below, missing = masked(
        hourly(store, requests, var, levels, first, last), var, sp
    )

    def reduce(ds, name, variable, stat="mean"):
        out = xhw.aggregate_daily(
            ds, variable=variable, statistic=stat, sampling="instantaneous", input_clock="UTC",
            day_definition=day_definition, name=name,
        )  # fmt: skip
        return out.sel(time=slice(f"{a}", f"{b}"))

    pieces = [reduce(series, f"{var}_{stat}", var, stat) for stat in STATISTICS]
    extra = xr.Dataset({"below": below, "sp_missing": missing})
    pieces.append(
        reduce(extra, f"{var}_below_ground_fraction", "below")[[f"{var}_below_ground_fraction"]]
    )
    pieces.append(reduce(extra, "sp_missing_fraction", "sp_missing")[["sp_missing_fraction"]])
    for name in derived:
        inputs, units, formula = DERIVED[name]
        if inputs != (var,) and not (name == "ws" and var == "u"):
            continue
        if name == "gh":
            values = series[var] / _derived.G0
        elif name == "theta":
            values = _derived.potential_temperature(series[var], series["pressure_level"])
        elif name == "ws":
            v = masked(hourly(store, requests, "v", levels, first, last), "v", sp)[0]["v"]
            values = _derived.wind_speed(series["u"], v)
        else:  # thickness 1000-500, only where 1000 hPa is above the terrain
            z = series[var]
            values = (z.sel(pressure_level=500.0) - z.sel(pressure_level=1000.0)) / _derived.G0
        ds = values.rename(name).to_dataset()
        ds[name].attrs = {"units": units, "formula": formula, "mask": "terrain mask applied first"}
        pieces.append(reduce(ds, f"{name}_mean", name))
    days = xr.merge(pieces, compat="override")
    labels = days["time"].values
    if len(set(labels.tolist())) != labels.size or labels.size != (b - a).days + 1:
        raise SystemExit(f"{a}..{b}: daily labels are not exactly the requested local days")
    for stat in STATISTICS:
        days[f"{var}_{stat}"].attrs["mask"] = "100 * pressure_level_hPa <= sp_Pa, sp finite"
    return days


def build_daily(
    store, requests, var, levels, sp_folder, start, end, day_definition, derived=(),
    expver=None, used=None,
):  # fmt: skip
    import xarray as xr

    blocks = [
        daily_block(
            store, requests, var, levels, sp_folder, a, b, day_definition, derived, expver, used
        )
        for a, b in single._blocks(start, end)
    ]
    days = xr.concat(blocks, dim="time")
    days.attrs = {"source": SOURCE, "surface_pressure": "ERA5 single levels sp (download_era5.py)"}
    return days


# --- Command line ------------------------------------------------------------------------


def main(argv=None, client_factory=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    aq.add_box_arguments(parser)
    aq.add_period_arguments(parser, "1991-01-01", "2025-12-31")
    parser.add_argument("--years", help="shortcut for --start/--end, e.g. 1991-2025")
    parser.add_argument("--variables", nargs="+", default=["temperature"])
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument(
        "--level-group", default="auto", help="levels per request: auto (largest that fits) or N"
    )
    parser.add_argument("--day-definition", default=single.DAY_DEFINITION)
    parser.add_argument("--margin", type=float, default=0.9, help="share of the CDS cost limit")
    parser.add_argument(
        "--surface-pressure", type=Path, help="download_era5.py folder holding sp (for --daily)"
    )
    parser.add_argument(
        "--derived", nargs="+", default=[], choices=list(DERIVED), help="derived daily means"
    )
    aq.add_run_arguments(parser)
    parser.add_argument("--allow-era5t", action="store_true")
    parser.add_argument("--accept-unknown-expver-before", type=date.fromisoformat)
    parser.add_argument("--expver-justification")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--compression-level", type=int, default=4, choices=range(1, 10))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--daily", action="store_true")
    args = parser.parse_args(argv)
    if args.years:
        try:
            first, last = map(int, args.years.split("-"))
            args.start, args.end = date(first, 1, 1), date(last, 12, 31)
        except ValueError:
            parser.error("--years must be an inclusive range, for example 1991-2025")
    try:
        start, end = aq.period_from_args(args, first=date(1940, 1, 1))
        box, region = aq.box_from_args(args)
        area = single.box_parts(box, args.box_policy)
        variables = [short_name(v) for v in args.variables]
        levels = [str(int(float(level))) for level in args.levels]
        level_group = None if args.level_group == "auto" else int(args.level_group)
        needs = {"ws": {"u", "v"}, "thickness_1000_500": {"z"}, "gh": {"z"}, "theta": {"t"}}
        for name in args.derived:
            if not needs[name] <= set(variables):
                raise ValueError(f"--derived {name} needs --variables {sorted(needs[name])}")
        if "thickness_1000_500" in args.derived and not {"1000", "500"} <= set(levels):
            raise ValueError("--derived thickness_1000_500 needs --levels 1000 and 500")
        requests = plan(
            area, variables, levels, start, end, day_definition=args.day_definition,
            level_group=level_group, margin=args.margin,
        )  # fmt: skip
    except (ValueError, aq.CostLimitError) as exc:
        parser.error(str(exc))
    if args.accept_unknown_expver_before and not args.expver_justification:
        parser.error("--accept-unknown-expver-before needs --expver-justification")
    key = single._key(region, area)
    areas = [part.area for part in area]
    area_record = areas[0] if len(areas) == 1 else areas
    store = single.Store.__new__(single.Store)
    store.root = args.data_dir / "era5_pressure" / key
    record = {
        "dataset": DATASET, "region": region, "box": box.as_dict(),
        "box_policy": args.box_policy, "area_nwse": area_record,
        "start": f"{start}", "end": f"{end}",
        "variables": variables, "levels": levels, "level_group": args.level_group,
        "day_definition": args.day_definition, "derived": list(args.derived),
    }  # fmt: skip
    if not (args.download or args.daily):
        for request in requests:
            print(request.id, json.dumps(request.body))
        print(
            f"Print only: {len(requests)} requests (CDS cost {sum(r.cost for r in requests)}, "
            f"largest {max(r.cost for r in requests)} of 60 000), area {area_record}; files in "
            f"{store.root}. Add --download to submit them."
        )
        return 0
    store = single.Store(store.root)
    ids = [r.id for r in requests]
    expver = single.ExpverPolicy(
        args.allow_era5t, args.accept_unknown_expver_before, args.expver_justification
    )
    compress = args.compression_level if args.compress else None
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
        valid = [i for i in ids if summary.statuses[i] == "validated"]
        manifest = aq.build_manifest(
            source="era5_pressure_levels", request=record, pieces=ids,
            files={
                store.ledger.get(i)["piece"]["path"]: store.ledger.resolve(
                    store.ledger.get(i)["piece"]["path"]
                )
                for i in valid
            },
            coverage=(
                store.ledger.get(valid[0])["checks"]["coverage"]
                if valid and len(area) == 1
                else None
            ),
            coverage_by_request={i: store.ledger.get(i)["checks"]["coverage"] for i in valid},
            retrieval_window=store.ledger.retrieval_window(ids), statuses=summary.statuses,
            expver={i: store.ledger.get(i)["checks"]["expver_status"] for i in valid},
            **expver.record(),
        )  # fmt: skip
        print(f"Manifest {aq.write_manifest(store.root, 'era5p_download', manifest).name}")
        return summary.exit_code

    if args.surface_pressure is None:
        raise SystemExit(
            "--daily needs --surface-pressure: the download_era5.py folder holding hourly sp "
            "over the same box and hours (revision 1, L7); without it no daily file is built"
        )
    inputs = single.plan_inputs(store, requests, expver)
    marks = single.expver_attrs(store, ids)
    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31):
        stamp = f"{start.year}_{end.year}"
    products = {}
    used: dict[str, tuple[str, str]] = {}
    for var in variables:
        ds = build_daily(
            store, requests, var, levels, args.surface_pressure, start, end,
            args.day_definition, args.derived, expver, used,
        )  # fmt: skip
        sp_marks = sorted({status for _, status in used.values()} - {"final"})
        product_marks = dict(marks)
        if sp_marks:  # the mask depends on sp: its status marks the product too
            status = product_marks["era5_expver_status"]
            joined = [m for m in status.split("+") if m != "final"] + sp_marks
            product_marks["era5_expver_status"] = "+".join(dict.fromkeys(joined))
            product_marks["era5_sp_expver"] = " ".join(sp_marks)
        ds.attrs.update(product_marks)
        joined_grid = ds
        product_inputs = {**inputs, **{f"sp:{name}": sha for name, (sha, _) in used.items()}}
        path = args.data_dir / f"era5p_{key}_daily_{var}_{levels_label(levels)}_{stamp}.nc"
        recipe = {
            "day_definition": args.day_definition, "statistics": list(STATISTICS),
            "derived": list(args.derived), "surface_pressure": str(args.surface_pressure),
            "mask": "100 * level_hPa <= sp_Pa", **expver.record(),
        }  # fmt: skip
        status = single.write_daily(store, ds, path, product_inputs, compress, recipe)
        print(f"{path.name}: {status}")
        products[path.name] = path
    manifest = aq.build_manifest(
        source="era5_pressure_levels", request=record, files=products, pieces=ids,
        retrieval_window=store.ledger.retrieval_window(ids), fingerprint=True,
        surface_pressure=str(args.surface_pressure), expver=marks, **expver.record(),
        surface_pressure_files={n: {"sha256": h, "expver_status": e} for n, (h, e) in used.items()},
        coverage=aq.coverage_report(  # the joined grid of the product
            box, joined_grid["latitude"].values, joined_grid["longitude"].values,
            args.box_policy, lat_step=GRID_STEP, lon_step=GRID_STEP,
        ),
    )  # fmt: skip
    print(f"Manifest {aq.write_manifest(args.data_dir, 'era5p_daily', manifest).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
