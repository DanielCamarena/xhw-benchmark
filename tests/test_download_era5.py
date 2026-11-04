# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of download_era5.py with a fake CDS client (revisions 0 and 2, F2.5)."""

from __future__ import annotations

import json
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5netcdf")
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _acquire as aq  # noqa: E402
import download_era5 as era5  # noqa: E402
from content_hash import content_sha256  # noqa: E402

# A small box whose edges fall between nodes: the enclosing area has 5 x 5 nodes.
SMALL = ["--box", "-12.9", "-12.1", "-82.1", "-81.4"]
ECMWF_UNITS = {"K": "K", "Pa": "Pa", "m s-1": "m s**-1", "J m-2": "J m**-2", "m": "m"}
ECMWF_UNITS["m2 s-2"] = "m**2 s**-2"
SHORT = {spec.cds: name for name, spec in era5.CATALOGUE.items()}


def value(var: str, times, lat, lon) -> np.ndarray:
    """Deterministic field of absolute time, so overlapping requests agree."""
    hours = (pd.DatetimeIndex(times) - pd.Timestamp("2000-01-01")) / pd.Timedelta("1h")
    h = np.asarray(hours)[:, None, None]
    base = {"t2m": 290.0, "mx2t": 291.0, "mn2t": 289.0, "lsm": 0.0, "z": 9.8}.get(var, 100.0)
    field = (
        base
        + 6 * np.sin(2 * np.pi * h / 24 + lat[None, :, None])
        + 2 * np.cos(h / 300 + lon[None, None, :])
    )
    return field.astype("float32")


class FakeCDS:
    """Writes what the CDS would answer to a request body; failures are injectable."""

    def __init__(
        self, *, expver="0001", fail=(), corrupt=(), inner=False, values=None, units=None,
        step_expver=None, short_steps=(),
    ):  # fmt: skip
        self.expver, self.fail, self.corrupt = expver, dict.fromkeys(fail, 1), list(corrupt)
        self.inner, self.values, self.units = inner, values or {}, units or {}
        self.step_expver, self.short_steps = step_expver or {}, set(short_steps)
        self.calls = []

    def retrieve(self, dataset, body, target):
        tag = f"{body['year'][0]}-{body['month'][0]}-{body['day'][0]}"
        self.calls.append(tag)
        if tag in self.fail:
            raise RuntimeError(f"fake CDS failure {tag}")  # unknown errors are permanent
        if tag in self.corrupt:
            self.corrupt.remove(tag)
            Path(target).write_bytes(b"not a NetCDF file")
            return
        north, west, south, east = body["area"]
        lat = np.arange(north, south - 1e-9, -0.25)
        lon = np.arange(west, east + 1e-9, 0.25)
        if self.inner:  # only the nodes strictly inside: what an unrounded area gives
            lat, lon = lat[1:-1], lon[1:-1]
        times = [
            pd.Timestamp(f"{y}-{m}-{d} {t}")
            for y in body["year"]
            for m in body["month"]
            for d in body["day"]
            if int(d) <= pd.Timestamp(f"{y}-{m}-01").days_in_month
            for t in body["time"]
        ]
        parts = {}
        for cds in body["variable"]:
            var = SHORT[cds]
            spec = era5.CATALOGUE[var]
            data = self.values.get(var, value)(var, times, lat, lon)
            units = self.units.get(var, ECMWF_UNITS.get(spec.units, "(0 - 1)"))
            arr = xr.DataArray(data, dims=("valid_time", "latitude", "longitude"))
            parts.setdefault(spec.step_type, {})[var] = arr.assign_attrs(units=units)
        datasets = {}
        for step, variables in parts.items():
            ds = xr.Dataset(
                variables, coords={"valid_time": times, "latitude": lat, "longitude": lon}
            )
            ds = ds.assign_coords(number=0)
            expver = self.step_expver.get(step, self.expver)
            if expver is not None and len(times) == 1:  # the real CDS: a scalar (F3, lsm)
                ds = ds.assign_coords(expver=expver)
            elif expver is not None:
                ds = ds.assign_coords(expver=("valid_time", [expver] * len(times)))
            if step in self.short_steps:  # a part with one hour missing
                ds = ds.isel(valid_time=slice(0, -1))
            datasets[step] = ds
        if len(datasets) == 1:
            next(iter(datasets.values())).to_netcdf(target, engine="h5netcdf")
            return
        with zipfile.ZipFile(target, "w") as archive:
            for step, ds in datasets.items():
                archive.writestr(
                    f"data_stream-oper_stepType-{step}.nc", ds.to_netcdf(engine="h5netcdf")
                )


def run(tmp_path, argv, fake=None):
    fake = fake or FakeCDS()
    code = era5.main(["--data-dir", str(tmp_path), *argv], client_factory=lambda: fake)
    return code, fake


def legacy_requests(first: int, last: int) -> list[dict]:
    """The 72-request plan of the registered input (download_era5.py at 58d2def)."""
    names = [era5.CATALOGUE[v].cds for v in era5.DEFAULT_VARIABLES]
    days = [f"{d:02d}" for d in range(1, 32)]
    out = []
    for year in range(first, last + 1):
        for months in (range(1, 7), range(7, 13)):
            out.append(
                {"variable": names, "year": [str(year)], "month": [f"{m:02d}" for m in months],
                 "day": days, "time": [f"{h:02d}:00" for h in range(24)]}
            )  # fmt: skip
    out.append(
        {"variable": names, "year": [str(last + 1)], "month": ["01"], "day": ["01"],
         "time": [f"{h:02d}:00" for h in range(7)]}
    )  # fmt: skip
    out.append(
        {"variable": ["land_sea_mask"], "year": ["2020"], "month": ["01"], "day": ["01"],
         "time": ["00:00"]}
    )  # fmt: skip
    common = {"product_type": ["reanalysis"], "data_format": "netcdf"}
    common.update(download_format="unarchived", area=[-4, -82, -12, -77])
    return [{**common, **body} for body in out]


# --- plan ------------------------------------------------------------------------------


def test_historical_plan_is_reproduced_exactly():
    area = aq.cds_area(aq.REGIONS["peru"], "enclose")
    requests = era5.plan(
        area, era5.DEFAULT_VARIABLES, date(1991, 1, 1), date(2025, 12, 31),
        chunk_period="half-year", grouping="all", boundary_hours=7,
    )  # fmt: skip
    assert [r.body for r in requests] == legacy_requests(1991, 2025)
    assert len({r.id for r in requests}) == 72


def test_default_plan_monthly_by_step_type_under_the_limit():
    area = aq.cds_area(aq.REGIONS["peru"], "enclose")
    requests = era5.plan(area, era5.DEFAULT_VARIABLES, date(1991, 1, 1), date(2025, 12, 31))
    assert len(requests) == 420 * 3 + 3 + 1
    assert all(r.cost <= 121_000 * 0.9 for r in requests)
    assert {len(r.variables) for r in requests} == {1}
    # 15 variables of the first new case (revision 2, decision 3), grouped by step type
    many = ["t2m", "mx2t", "mn2t", "sp", "u10", "v10", "fg10", "d2m", "msl"]
    many += ["ssr", "str", "sshf", "slhf", "ssrd", "strd"]
    requests = era5.plan(area, many, date(2001, 1, 1), date(2001, 1, 31))
    groups = {r.variables for r in requests if r.start == date(2001, 1, 1)}
    assert groups == {
        ("t2m", "sp", "u10", "v10", "d2m", "msl"),
        ("ssr", "str", "sshf", "slhf", "ssrd", "strd"),
        ("mx2t", "fg10"),
        ("mn2t",),
    }
    with pytest.raises(ValueError, match="not in the catalogue"):
        era5.plan(area, ["t2m", "cape"], date(2001, 1, 1), date(2001, 1, 31))


@pytest.mark.parametrize(
    "variables, day_definition, before, after",
    [
        (("mx2t",), "offset:-05:00", [], list(range(6))),  # D5: 6 stamps, not 7
        (("t2m",), "offset:-05:00", [], list(range(5))),
        (("t2m",), "offset:+05:00", list(range(19, 24)), []),
        (("mx2t",), "offset:+05:00", list(range(20, 24)), []),
        (("t2m",), "UTC", [], []),
        (("tp",), "UTC", [], [0]),  # accumulation ending at D+1 00:00 belongs to D
    ],
)
def test_boundary_hours_follow_the_day_definition(variables, day_definition, before, after):
    requests = era5.plan(
        [-12.0, -82.0, -12.5, -81.5], variables, date(2001, 3, 1), date(2001, 3, 31),
        day_definition=day_definition, invariants=(),
    )  # fmt: skip
    edge = [r for r in requests if r.start != date(2001, 3, 1)]
    assert [list(r.hours) for r in edge if r.start < date(2001, 3, 1)] == (
        [before] if before else []
    )
    assert [list(r.hours) for r in edge if r.start > date(2001, 3, 31)] == (
        [after] if after else []
    )


def test_print_only_rounds_the_area_and_touches_nothing(tmp_path, capsys):
    code, fake = run(tmp_path, [*SMALL, "--start", "2023-03-01", "--end", "2023-03-02"])
    assert code == 0 and fake.calls == [] and list(tmp_path.iterdir()) == []
    out = capsys.readouterr().out
    assert '"area": [-12.0, -82.25, -13.0, -81.25]' in out
    code, _ = run(tmp_path, [*SMALL, "--box-policy", "inside", "--years", "2023-2023"])
    assert '"area": [-12.1, -82.1, -12.9, -81.4]' in capsys.readouterr().out


# --- download, damage, expver -----------------------------------------------------------


def statuses(tmp_path, key=None):
    folder = next((tmp_path / "era5_single").iterdir()) if key is None else key
    ledger = json.loads((folder / "ledger.json").read_text())
    return {i: e["status"] for i, e in ledger["pieces"].items()}, ledger


def test_download_validates_splits_and_records(tmp_path):
    code, fake = run(
        tmp_path, [*SMALL, "--start", "2023-01-30", "--end", "2023-02-02", "--download"]
    )
    assert code == 0
    states, ledger = statuses(tmp_path)
    assert set(states.values()) == {"validated"}
    root = next((tmp_path / "era5_single").iterdir())
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.nc"))
    assert "t2m/2023/t2m_2023-01-30_2023-01-31.nc" in files
    assert "mx2t/2023/mx2t_2023-02-03_2023-02-03_h00-05.nc" in files
    assert "lsm/lsm.nc" in files
    with xr.open_dataset(root / "t2m/2023/t2m_2023-02-01_2023-02-02.nc", engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 48 and "expver" not in ds.variables
    sidecar = json.loads((root / "t2m/2023/t2m_2023-02-01_2023-02-02.provenance.json").read_text())
    assert sidecar["expver_status"] == "final"
    entry = next(e for i, e in ledger["pieces"].items() if i.startswith("sl_20230130_t2m"))
    assert entry["checks"]["coverage"]["ok"] and entry["checks"]["coverage"]["policy"] == "enclose"
    (manifest,) = root.glob("era5_download_manifest_*.json")
    stored = json.loads(manifest.read_text())
    assert stored["request"]["area_nwse"] == [-12.0, -82.25, -13.0, -81.25]
    assert stored["retrieval_window_utc"] and len(stored["files"]) == len(files)
    # a second run submits nothing
    calls = len(fake.calls)
    assert (
        run(tmp_path, [*SMALL, "--start", "2023-01-30", "--end", "2023-02-02", "--download"], fake)[
            0
        ]
        == 0
    )
    assert len(fake.calls) == calls


def test_every_failure_is_reported_and_the_rest_downloaded(tmp_path, capsys):
    # Revision 2, H2: two failures, both reported; the other requests are not cancelled.
    fake = FakeCDS(fail=["2023-01-30", "2023-02-03"])
    code, _ = run(
        tmp_path,
        [
            *SMALL,
            "--start",
            "2023-01-30",
            "--end",
            "2023-02-02",
            "--download",
            "--variables",
            "t2m",
        ],
        fake,
    )  # noqa: E501
    assert code == 1
    states, _ = statuses(tmp_path)
    assert sorted(states.values()) == ["failed", "failed", "validated", "validated"]
    out = capsys.readouterr().out
    assert "fake CDS failure 2023-01-30" in out and "fake CDS failure 2023-02-03" in out


@pytest.mark.parametrize("keep_failed", [False, True])
def test_damaged_answer_leaves_evidence(tmp_path, keep_failed):
    # Revision 2, H3 and decision 1.
    fake = FakeCDS(corrupt=["2023-03-01"])
    argv = [
        *SMALL,
        "--start",
        "2023-03-01",
        "--end",
        "2023-03-01",
        "--variables",
        "t2m",
        "--download",
    ]
    code, _ = run(tmp_path, argv + (["--keep-failed"] if keep_failed else []), fake)
    states, ledger = statuses(tmp_path)
    entry = next(e for i, e in ledger["pieces"].items() if i.startswith("sl_20230301"))
    (damage,) = entry["damage"]
    assert damage["bytes"] == 17 and len(damage["sha256"]) == 64 and "unreadable" in damage["error"]
    if keep_failed:
        assert code == 1 and entry["status"] == "quarantined"
        quarantined = list((tmp_path / "era5_single").rglob("quarantine/*"))
        assert len(quarantined) == 1 and quarantined[0].read_bytes() == b"not a NetCDF file"
    else:
        assert code == 0 and entry["status"] == "validated" and entry["attempts"] == 2


def test_scalar_expver_of_a_one_hour_answer():
    # F3, real CDS: the land-sea mask (one hour) came with a 0-d expver, which failed with
    # "iteration over a 0-d array".
    assert era5.expver_status(np.array("0001")) == ("final", ["0001"])
    assert era5.expver_status(np.array(b"0005")) == ("preliminary", ["0005"])
    assert era5.expver_status(np.array(["0001", "0001"])) == ("final", ["0001"])


@pytest.mark.parametrize(
    "expver, extra, status",
    [
        ("0005", [], None),
        ("0005", ["--allow-era5t"], "preliminary"),
        (None, [], None),
        (None, ["--accept-unknown-expver-before", "2024-01-01", "--expver-justification", "x"],
         "unknown"),
        (None, ["--accept-unknown-expver-before", "2023-03-01", "--expver-justification", "x"],
         None),
        ("1", [], "final"),
    ],
)  # fmt: skip
def test_expver_policy(tmp_path, expver, extra, status):
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--variables", "t2m"]
    code, _ = run(tmp_path, [*argv, "--download", *extra], FakeCDS(expver=expver))
    _, ledger = statuses(tmp_path)
    entry = next(e for i, e in ledger["pieces"].items() if i.startswith("sl_20230301"))
    if status is None:
        assert code == 1 and entry["status"] == "failed" and "Era5tError" in entry["error"]
    else:
        assert entry["status"] == "validated" and entry["checks"]["expver_status"] == status


def test_unknown_expver_exception_needs_a_justification(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, [*SMALL, "--accept-unknown-expver-before", "2024-01-01", "--download"])


def test_answer_that_does_not_enclose_the_box_is_rejected(tmp_path):
    # Revision 0, B6: only the inner nodes arrive; both attempts fail, with evidence.
    argv = [
        *SMALL,
        "--start",
        "2023-03-01",
        "--end",
        "2023-03-01",
        "--variables",
        "t2m",
        "--download",
    ]
    code, fake = run(tmp_path, argv, FakeCDS(inner=True))
    _, ledger = statuses(tmp_path)
    entry = next(e for i, e in ledger["pieces"].items() if i.startswith("sl_20230301"))
    assert code == 1 and entry["status"] == "failed" and len(entry["damage"]) == 2
    assert "do not enclose" in entry["damage"][0]["error"]


def test_wrong_units_are_rejected(tmp_path):
    argv = [
        *SMALL,
        "--start",
        "2023-03-01",
        "--end",
        "2023-03-01",
        "--variables",
        "t2m",
        "--download",
    ]
    code, _ = run(tmp_path, argv, FakeCDS(units={"t2m": "degC"}))
    _, ledger = statuses(tmp_path)
    entry = next(e for i, e in ledger["pieces"].items() if i.startswith("sl_20230301"))
    assert code == 1 and "expected 'K'" in entry["damage"][0]["error"]


def test_mixed_step_types_arrive_as_zip_parts(tmp_path):
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--group-all", "--download"]
    code, _ = run(tmp_path, argv)
    _, ledger = statuses(tmp_path)
    entry = next(e for i, e in ledger["pieces"].items() if i.startswith("sl_20230301"))
    assert code == 0 and len(entry["checks"]["zip_members"]) == 3
    assert len(entry["piece"]["files"]) == 3  # one file per variable


# --- daily products ---------------------------------------------------------------------


def legacy_daily(tmp_path, first: date, last: date) -> Path:
    """The registered pipeline (build_daily at 58d2def) on the same hourly values."""
    import xhw

    area = [-12.0, -82.25, -13.0, -81.25]
    lat = np.arange(area[0], area[2] - 1e-9, -0.25)
    lon = np.arange(area[1], area[3] + 1e-9, 0.25)
    pieces = []
    for year in range(first.year, last.year + 1):
        times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01 07:00", freq="1h")
        ds = xr.Dataset(
            {v: (("time", "latitude", "longitude"), value(v, times, lat, lon))
             for v in era5.DEFAULT_VARIABLES},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )  # fmt: skip
        for v in era5.DEFAULT_VARIABLES:
            ds[v].attrs["units"] = "K"
        path = tmp_path / f"legacy_{year}.nc"
        ds.to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            hourly = opened.load()
        out = [
            xhw.aggregate_daily(
                hourly, variable=v, statistic=s, sampling=m, input_clock="UTC",
                day_definition="offset:-05:00", name=n,
            )
            for v, s, m, n in (
                ("mx2t", "max", "period_ending", "tx"),
                ("mn2t", "min", "period_ending", "tn"),
                ("t2m", "max", "instantaneous", "tx_t2m"),
                ("t2m", "min", "instantaneous", "tn_t2m"),
            )
        ]  # fmt: skip
        pieces.append(
            xr.merge(out, compat="override").sel(time=slice(f"{max(first, date(year, 1, 1))}",
                                                            f"{min(last, date(year, 12, 31))}"))
        )  # fmt: skip
    days = xr.concat(pieces, dim="time")
    lsm = xr.DataArray(value("lsm", [pd.Timestamp("2020-01-01")], lat, lon)[0],
                       dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon},
                       attrs={"units": "(0 - 1)"}, name="lsm")  # fmt: skip
    days["lsm"] = lsm
    days.attrs = {"source": era5.SOURCE}
    target = tmp_path / "legacy_daily.nc"
    days.to_netcdf(target, engine="h5netcdf")
    return target


def test_daily_product_equals_the_registered_pipeline(tmp_path):
    argv = [*SMALL, "--start", "2022-12-30", "--end", "2023-01-02"]
    assert run(tmp_path / "new", [*argv, "--download"])[0] == 0
    assert run(tmp_path / "new", [*argv, "--daily"])[0] == 0
    (product,) = (tmp_path / "new").glob("era5_box_*_daily_20221230_20230102.nc")
    legacy = legacy_daily(tmp_path, date(2022, 12, 30), date(2023, 1, 2))
    assert content_sha256(product) == content_sha256(legacy)
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 4 and int(ds["tx_n_samples"].min()) == 24
    # same content through the historical half-year plan
    hist = [*argv, "--chunk-period", "half-year", "--group-all", "--boundary-hours", "7"]
    assert run(tmp_path / "hist", [*hist, "--download"])[0] == 0
    assert run(tmp_path / "hist", [*hist, "--daily"])[0] == 0
    (other,) = (tmp_path / "hist").glob("era5_box_*_daily_*.nc")
    assert content_sha256(other) == content_sha256(product)
    # a repeated --daily leaves the current product alone
    before = product.stat().st_mtime_ns
    assert run(tmp_path / "new", [*argv, "--daily"])[0] == 0
    assert product.stat().st_mtime_ns == before


def test_extending_the_period_never_uses_stale_pieces(tmp_path):
    # Revision 2, H1: the old boundary piece of the first period is not mixed in.
    first = [*SMALL, "--start", "2023-01-30", "--end", "2023-01-31"]
    assert run(tmp_path, [*first, "--download"])[0] == 0
    assert run(tmp_path, [*first, "--daily"])[0] == 0
    longer = [*SMALL, "--start", "2023-01-30", "--end", "2023-02-02"]
    with pytest.raises(SystemExit, match="not downloaded"):
        run(tmp_path, [*longer, "--daily"])
    assert run(tmp_path, [*longer, "--download"])[0] == 0
    assert run(tmp_path, [*longer, "--daily"])[0] == 0
    (product,) = tmp_path.glob("era5_box_*_daily_20230130_20230202.nc")
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 4 and int(ds["tn_n_samples"].min()) == 24
    # a re-downloaded piece with other content makes the product stale, not current
    root = next((tmp_path / "era5_single").iterdir())
    ledger = aq.Ledger(root / "ledger.json")
    rid = next(i for i in ledger.pieces if i.startswith("sl_20230201_t2m"))
    inputs = {i: ledger.get(i)["piece"]["sha256"] for i in ledger.pieces}
    assert ledger.product_status(product.name, product, {**inputs, rid: "0" * 64}) == "stale"


def test_accumulated_daily_conversions(tmp_path):
    # Revision 2, V3: 3600 J m-2 per hour is 1 W m-2; 0.001 m per hour is 0.024 m per day.
    def constant(level):
        return lambda var, times, lat, lon: np.full((len(times), lat.size, lon.size), level, "f4")

    fake = FakeCDS(values={"ssrd": constant(3600.0), "tp": constant(0.001)})
    for day_definition in ("offset:-05:00", "offset:+05:00"):
        argv = [*SMALL, "--start", "2023-03-02", "--end", "2023-03-03", "--variables", "ssrd", "tp",
                "--day-definition", day_definition]  # fmt: skip
        folder = tmp_path / day_definition.replace(":", "")
        assert run(folder, [*argv, "--download"], fake)[0] == 0
        assert run(folder, [*argv, "--daily"], fake)[0] == 0
        (flux,) = folder.glob("era5_*_daily_ssrd_mean_*.nc")
        (rain,) = folder.glob("era5_*_daily_tp_total_*.nc")
        with xr.open_dataset(flux, engine="h5netcdf") as ds:
            assert np.allclose(ds["ssrd"], 1.0) and ds["ssrd"].attrs["units"] == "W m-2"
        with xr.open_dataset(rain, engine="h5netcdf") as ds:
            assert np.allclose(ds["tp"], 0.024)


def test_accumulation_ending_at_midnight_belongs_to_the_previous_day(tmp_path):
    # Revision 2, 7.2: 0 at D 00:00, 86 400 J m-2 at D+1 00:00, 0 otherwise -> 1 W m-2 for D.
    def edges(var, times, lat, lon):
        out = np.zeros((len(times), lat.size, lon.size), "f4")
        for i, t in enumerate(times):
            if t == pd.Timestamp("2023-03-03 00:00"):
                out[i] = 86400.0
        return out

    argv = [*SMALL, "--start", "2023-03-02", "--end", "2023-03-02", "--variables", "ssrd",
            "--day-definition", "UTC"]  # fmt: skip
    fake = FakeCDS(values={"ssrd": edges})
    assert run(tmp_path, [*argv, "--download"], fake)[0] == 0
    assert run(tmp_path, [*argv, "--daily"], fake)[0] == 0
    (flux,) = tmp_path.glob("era5_*_daily_ssrd_mean_*.nc")
    with xr.open_dataset(flux, engine="h5netcdf") as ds:
        assert np.allclose(ds["ssrd"].sel(time="2023-03-02"), 1.0)


def test_consolidated_years_for_the_registered_layout(tmp_path):
    argv = [*SMALL, "--start", "2022-12-31", "--end", "2023-01-01"]
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--consolidate-years"])[0] == 0
    (folder,) = tmp_path.glob("era5_box_*_hourly")
    assert sorted(p.name for p in folder.glob("*.nc")) == ["2022.nc", "2023.nc", "lsm.nc"]
    with xr.open_dataset(folder / "2023.nc", engine="h5netcdf") as ds:
        assert set(ds.data_vars) == {"t2m", "mx2t", "mn2t"}
        assert ds.sizes["valid_time"] == 24 + 5  # hours planned for all three variables
        assert ds["valid_time"].values[-1] == np.datetime64(datetime(2023, 1, 2, 4))


def test_derived_products_end_to_end(tmp_path):
    argv = [*SMALL, "--start", "2023-03-02", "--end", "2023-03-03",
            "--variables", "t2m", "mx2t", "mn2t", "u10", "v10",
            "--derived", "ws10", "wdir10", "tmean_9to9"]  # fmt: skip
    with pytest.raises(SystemExit):  # inputs of a derived product must be requested
        run(tmp_path, [*SMALL, "--variables", "t2m", "--derived", "ws10"])
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    names = sorted(p.name.split("_daily_")[1] for p in tmp_path.glob("era5_*_daily_*.nc"))
    assert names == [
        "20230302_20230303.nc",
        "tmean_9to9_20230302_20230303.nc",
        "u10_mean_20230302_20230303.nc",
        "v10_mean_20230302_20230303.nc",
        "wdir10_20230302_20230303.nc",
        "ws10_max_20230302_20230303.nc",
        "ws10_mean_20230302_20230303.nc",
    ]
    (nine,) = tmp_path.glob("era5_*_daily_tmean_9to9_*.nc")
    with xr.open_dataset(nine, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 2 and bool(ds["tmean_9to9"].notnull().all())


# --- review findings (GPT, F1-F6) -----------------------------------------------------------


def piece_entry(tmp_path, prefix):
    _, ledger = statuses(tmp_path)
    return next(e for i, e in ledger["pieces"].items() if i.startswith(prefix))


def test_zip_parts_are_checked_one_by_one(tmp_path):
    # F1: a part with a missing hour, or with ERA5T data, is not hidden by the merge.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--group-all", "--download"]
    code, _ = run(tmp_path / "short", argv, FakeCDS(short_steps={"max"}))
    entry = piece_entry(tmp_path / "short", "sl_20230301")
    assert code == 1 and "stepType-max.nc has 23 hours" in entry["damage"][0]["error"]
    code, _ = run(tmp_path / "mixed", argv, FakeCDS(step_expver={"max": "0005"}))
    entry = piece_entry(tmp_path / "mixed", "sl_20230301")
    assert code == 1 and "preliminary ERA5T" in entry["error"]


def test_current_expver_policy_applies_to_reused_pieces(tmp_path):
    # F2: pieces accepted with --allow-era5t are not reused under the default policy.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--variables", "t2m"]
    preliminary = FakeCDS(expver="0005")
    assert run(tmp_path, [*argv, "--download", "--allow-era5t"], preliminary)[0] == 0
    with pytest.raises(SystemExit, match="refused by the"):
        run(tmp_path, [*argv, "--daily"])
    assert run(tmp_path, [*argv, "--daily", "--allow-era5t"])[0] == 0
    (product,) = tmp_path.glob("era5_*_daily_t2m_max_*.nc")
    with xr.open_dataset(product, engine="h5netcdf") as ds:  # F6: the mark travels
        assert ds.attrs["era5_expver_status"] == "preliminary"
    final = FakeCDS(expver="0001")  # fetched again, now final
    assert run(tmp_path, [*argv, "--download"], final)[0] == 0 and final.calls
    assert piece_entry(tmp_path, "sl_20230301")["checks"]["expver_status"] == "final"
    root = next((tmp_path / "era5_single").iterdir())
    (piece,) = root.glob("t2m/2023/t2m_2023-03-01_2023-03-01.nc")
    with xr.open_dataset(piece, engine="h5netcdf") as ds:
        assert ds.attrs["era5_expver_status"] == "final"


def test_product_is_rebuilt_when_its_recipe_changes(tmp_path):
    # F3: --no-contrast and another day definition change the content, so it is rebuilt.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--boundary-hours", "7"]
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    (product,) = tmp_path.glob("era5_*_daily_20230301_20230301.nc")
    assert run(tmp_path, [*argv, "--daily", "--no-contrast"])[0] == 0
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        assert "tx_t2m" not in ds
    assert run(tmp_path, [*argv, "--daily", "--day-definition", "UTC"])[0] == 0
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        assert ds["tx"].attrs.get("cell_methods") and "tx_t2m" in ds


def test_daily_of_part_of_the_default_variables(tmp_path):
    # F4: --variables t2m gives its own daily files instead of nothing.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--variables", "t2m"]
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    names = sorted(p.name.split("_daily_")[1] for p in tmp_path.glob("era5_*_daily_*.nc"))
    assert names == [f"t2m_{s}_20230301_20230301.nc" for s in ("max", "mean", "min")]


def test_offsets_with_minutes(tmp_path):
    # F5: +05:30 days hold 24 whole hourly stamps.
    first, last = era5.utc_window(["t2m"], date(2023, 3, 1), date(2023, 3, 1), "offset:+05:30")
    assert (first, last) == (datetime(2023, 2, 28, 19), datetime(2023, 3, 1, 18))
    first, last = era5.utc_window(["mx2t"], date(2023, 3, 1), date(2023, 3, 1), "offset:+05:30")
    assert (first, last) == (datetime(2023, 2, 28, 20), datetime(2023, 3, 1, 19))
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-02", "--day-definition",
            "offset:+05:30"]  # fmt: skip
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    (product,) = tmp_path.glob("era5_*_daily_20230301_20230302.nc")
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        assert int(ds["tx_t2m_n_samples"].min()) == 24 and int(ds["tx_n_samples"].min()) == 24


def test_nine_to_nine_at_extreme_offsets_plans_every_needed_hour(tmp_path):
    # Re-review R1: at -14:30 the 09:00-09:00 day needs hours two UTC dates later.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01",
            "--day-definition", "offset:-14:30", "--variables", "t2m", "mx2t", "mn2t",
            "--derived", "tmean_9to9"]  # fmt: skip
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    (nine,) = tmp_path.glob("era5_*_daily_tmean_9to9_*.nc")
    with xr.open_dataset(nine, engine="h5netcdf") as ds:
        assert bool(ds["tmean_9to9"].notnull().all())
        assert int(ds["tmean_9to9_mx2t_n_samples"].min()) == 24


# --- boxes crossing the antimeridian (revision 0, B5) ---------------------------------------

CROSSING = ["--box", "25.1", "25.4", "179.6", "-179.6"]


def test_crossing_box_is_requested_in_two_parts():
    box = aq.REGIONS["npacific"]
    parts = era5.box_parts(box, "enclose")
    assert [(p.label, p.area) for p in parts] == [
        ("w", [55.0, 160.0, 25.0, 180.0]),
        ("e", [55.0, -180.0, 25.0, -130.0]),
    ]
    requests = era5.plan(parts, ["t2m"], date(2023, 1, 1), date(2023, 1, 31))
    assert {r.part for r in requests} == {"w", "e"}
    month = [r for r in requests if r.start == date(2023, 1, 1) and r.variables == ("t2m",)]
    assert len(month) == 2 and month[0].id != month[1].id
    root = Path("r")
    names = {p.name for r in month for p in r.outputs(root).values()}
    assert names == {"t2m_2023-01_w.nc", "t2m_2023-01_e.nc"}
    # a box not crossing keeps the single area and the old file names
    (single,) = era5.box_parts(aq.REGIONS["peru"], "enclose")
    assert single.label == "" and single.area == [-4.0, -82.0, -12.0, -77.0]


def test_crossing_box_end_to_end(tmp_path):
    argv = [*CROSSING, "--start", "2023-03-01", "--end", "2023-03-02"]
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    (product,) = tmp_path.glob("era5_box_*_daily_20230301_20230302.nc")
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        lon = ds["longitude"].values.tolist()
        assert lon == [179.5, 179.75, 180.0, -179.75, -179.5]  # one copy of the 180 meridian
        assert bool(ds["tx"].notnull().all()) and int(ds["tx_n_samples"].min()) == 24
        assert ds["lsm"].sizes["longitude"] == 5
    (manifest,) = tmp_path.glob("era5_daily_manifest_*.json")
    stored = json.loads(manifest.read_text())
    assert stored["box"]["ok"] and stored["box"]["longitude"]["nodes"] == [179.5, 180.5]
    assert len(stored["request"]["area_nwse"]) == 2


def test_crossing_box_part_failing_is_reported(tmp_path):
    fake = FakeCDS(inner=True)  # every answer misses its edge nodes
    argv = [*CROSSING, "--start", "2023-03-01", "--end", "2023-03-01", "--variables", "t2m"]
    code, _ = run(tmp_path, [*argv, "--download"], fake)
    states, _ = statuses(tmp_path)
    assert code == 1 and "failed" in states.values()
    with pytest.raises(SystemExit, match="not downloaded"):
        run(tmp_path, [*argv, "--daily"])


def test_utci_and_wbgt_end_to_end(tmp_path):
    pytest.importorskip("thermofeel")
    radiation = ["ssrd", "ssr", "strd", "str", "fdir"]
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-02", "--day-definition",
            "offset:+05:00", "--variables", "t2m", "d2m", "u10", "v10", *radiation,
            "--derived", "utci2m", "wbgt2m", "mrt"]  # fmt: skip
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    for name in ("utci2m_max", "utci2m_mean", "wbgt2m_max", "mrt_mean"):
        (path,) = tmp_path.glob(f"era5_*_daily_{name}_*.nc")
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            assert ds.sizes["time"] == 2 and bool(ds[name].notnull().all())
            count = next(v for v in ds.data_vars if str(v).endswith("_n_samples"))
            assert int(ds[count].min()) == 24
            assert "thermofeel" in ds[name].attrs["constants"]
            if name.startswith("utci"):  # review C4: the validity indicator travels along
                assert "utci2m_outside_domain_fraction" in ds


def test_box_not_crossing_keeps_its_identity():
    # Review H1: one part has the same key (and folder) as before the parts existed.
    area = aq.cds_area(aq.Box(-12.9, -12.1, -82.1, -81.4), "enclose")
    parts = era5.box_parts(aq.Box(-12.9, -12.1, -82.1, -81.4), "enclose")
    assert era5._key(None, parts) == era5._key(None, area) == "box_037c972b"


def test_join_parts_whatever_the_stored_order():
    # Review H2: descending longitudes in each part still give one 180 meridian, west to east.
    def part(lons):
        return xr.Dataset({"v": ("longitude", np.zeros(len(lons)))}, coords={"longitude": lons})

    for west, east in (
        ([179.5, 179.75, 180.0], [-180.0, -179.75, -179.5]),
        ([180.0, 179.75, 179.5], [-179.5, -179.75, -180.0]),
        ([179.5, 179.75, -180.0], [-180.0, -179.75, -179.5]),
    ):
        joined = era5.join_parts([part(west), part(east)])
        lon = joined["longitude"].values.tolist()
        assert len(lon) == 5 and lon[:2] == [179.5, 179.75] and lon[3:] == [-179.75, -179.5]


def test_download_manifest_keeps_the_coverage_format(tmp_path):
    # Review H3: a box without parts keeps manifest["box"] as one coverage report.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--variables", "t2m"]
    assert run(tmp_path, [*argv, "--download"])[0] == 0
    root = next((tmp_path / "era5_single").iterdir())
    (manifest,) = root.glob("era5_download_manifest_*.json")
    stored = json.loads(manifest.read_text())
    assert stored["box"]["ok"] and stored["box"]["policy"] == "enclose"
    assert len(stored["coverage_by_request"]) == 3
