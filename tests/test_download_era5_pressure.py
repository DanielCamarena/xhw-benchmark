# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of download_era5_pressure.py with a fake CDS client (revision 1, section 6)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5netcdf")
pytest.importorskip("xhw")
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _acquire as aq  # noqa: E402
import download_era5 as single  # noqa: E402
import download_era5_pressure as pl  # noqa: E402
from content_hash import content_sha256  # noqa: E402

SMALL = ["--box", "-12.9", "-12.1", "-82.1", "-81.4"]  # 5 x 5 nodes after rounding outward
UNITS = {"t": "K", "z": "m**2 s**-2", "u": "m s**-1", "v": "m s**-1", "sp": "Pa", "lsm": "1"}


def temperature(times, levels, lat, lon):
    h = np.asarray((pd.DatetimeIndex(times) - pd.Timestamp("2000-01-01")) / pd.Timedelta("1h"))
    p = np.asarray(levels, dtype=float)
    field = (
        300.0 - 0.05 * (1000.0 - p)[None, :, None, None]
        + 3 * np.sin(2 * np.pi * h / 24)[:, None, None, None]
        + 0.1 * lat[None, None, :, None] + 0.01 * lon[None, None, None, :]
    )  # fmt: skip
    return field.astype("float32")


def surface_pressure_high(times, lat, lon):
    """Low ground everywhere (sp 101 325 Pa) except the first longitude (70 000 Pa)."""
    sp = np.full((len(times), lat.size, lon.size), 101325.0, "float32")
    sp[:, :, 0] = 70000.0
    return sp


class FakeCDS:
    """Answers single-level sp/lsm and pressure-level requests; damage is injectable."""

    def __init__(self, *, expver="0001", levels_out=None, level_units="hPa", corrupt=(), sp=None):
        self.expver, self.levels_out, self.level_units = expver, levels_out, level_units
        self.corrupt, self.sp = list(corrupt), sp or surface_pressure_high
        self.calls = []

    def retrieve(self, dataset, body, target):
        tag = f"{body['year'][0]}-{body['month'][0]}-{body['day'][0]}"
        self.calls.append((dataset, tag))
        if tag in self.corrupt:
            self.corrupt.remove(tag)
            Path(target).write_bytes(b"cut")
            return
        north, west, south, east = body["area"]
        lat = np.arange(north, south - 1e-9, -0.25)
        lon = np.arange(west, east + 1e-9, 0.25)
        times = [
            pd.Timestamp(f"{y}-{m}-{d} {t}")
            for y in body["year"]
            for m in body["month"]
            for d in body["day"]
            if int(d) <= pd.Timestamp(f"{y}-{m}-01").days_in_month
            for t in body["time"]
        ]
        coords = {"valid_time": times, "latitude": lat, "longitude": lon}
        if dataset == pl.DATASET:
            var = pl.BY_CDS[body["variable"][0]]
            levels = [float(x) for x in (self.levels_out or body["pressure_level"])]
            data = temperature(times, levels, lat, lon)
            if var == "z":
                data = (data * 30).astype("float32")
            ds = xr.Dataset(
                {var: (("valid_time", "pressure_level", "latitude", "longitude"), data)},
                coords={**coords, "pressure_level": levels},
            )
            ds["pressure_level"].attrs["units"] = self.level_units
        else:
            var = single.SHORT_NAMES[body["variable"][0]]
            data = (
                self.sp(times, lat, lon)
                if var == "sp"
                else np.ones((len(times), lat.size, lon.size), "f4")
            )
            ds = xr.Dataset({var: (("valid_time", "latitude", "longitude"), data)}, coords=coords)
        ds[var].attrs["units"] = UNITS[var]
        if self.expver is not None:
            ds = ds.assign_coords(expver=("valid_time", [self.expver] * len(times)))
        ds.to_netcdf(target, engine="h5netcdf")


single.SHORT_NAMES = {spec.cds: name for name, spec in single.CATALOGUE.items()}


def run(folder, argv, fake=None):
    fake = fake or FakeCDS()
    code = pl.main(["--data-dir", str(folder), *argv], client_factory=lambda: fake)
    return code, fake


def run_sp(folder, argv, fake=None):
    fake = fake or FakeCDS()
    argv = [*argv, "--variables", "sp"]
    code = single.main(["--data-dir", str(folder), *argv], client_factory=lambda: fake)
    return code, fake


def entry(folder, prefix):
    root = next((folder / "era5_pressure").iterdir())
    ledger = json.loads((root / "ledger.json").read_text())
    return next(e for i, e in ledger["pieces"].items() if i.startswith(prefix))


# --- plan (tests 1, 2) -------------------------------------------------------------------


def test_plan_per_variable_month_and_level_group():
    area = [-4.0, -82.0, -12.0, -77.0]
    requests = pl.plan(area, ["temperature", "z"], ["850", "700", "500"], date(2023, 1, 1),
                       date(2023, 2, 28))  # fmt: skip
    main = [r for r in requests if r.hours == tuple(range(24))]
    assert len(main) == 4 and {r.variable for r in main} == {"t", "z"}
    assert main[0].body["pressure_level"] == ["850", "700", "500"]
    assert main[0].body["variable"] == ["temperature"]
    assert main[0].cost == 13_392  # decision 1: one monthly request of 60 000
    assert main[0].output(Path("r")).as_posix() == "r/t/2023/t_p850-700-500_2023-01.nc"
    boundary = [r for r in requests if r.hours != tuple(range(24))]
    assert [list(r.hours) for r in boundary] == [list(range(5))] * 2  # -05:00, instantaneous
    # 37 levels: groups under the margin, each level once per month
    levels = [str(p) for p in range(1000, 99, -25)]
    requests = pl.plan(area, ["t"], levels, date(2023, 1, 1), date(2023, 1, 31))
    groups = [r.levels for r in requests if r.hours == tuple(range(24))]
    assert sorted(lev for g in groups for lev in g) == sorted(levels)
    assert all(r.cost <= 54_000 for r in requests)
    assert "x10" in requests[0].output(Path("r")).name  # hashed label of a 10-level group


def test_other_levels_are_another_request_and_file():
    area = [-4.0, -82.0, -12.0, -77.0]
    a = pl.plan(area, ["t"], ["850", "700", "500"], date(2023, 1, 1), date(2023, 1, 31))[0]
    b = pl.plan(area, ["t"], ["850", "500"], date(2023, 1, 1), date(2023, 1, 31))[0]
    assert a.id != b.id and a.output(Path("r")) != b.output(Path("r"))
    ledger = aq.Ledger(Path("unused") / "ledger.json", autosave=False)
    ledger.plan(a.id, a.body, "x")
    with pytest.raises(aq.LedgerError):
        ledger.plan(a.id, b.body, "x")


def test_single_levels_plan_is_unchanged():
    # Test 10: the single-levels script keeps its registered plan.
    area = aq.cds_area(aq.REGIONS["peru"], "enclose")
    requests = single.plan(area, single.DEFAULT_VARIABLES, date(1991, 1, 1), date(2025, 12, 31),
                           chunk_period="half-year", grouping="all", boundary_hours=7)  # fmt: skip
    assert len(requests) == 72


# --- validation (tests 3, 4, 9) -----------------------------------------------------------


@pytest.mark.parametrize(
    "fake, message",
    [
        (FakeCDS(levels_out=["850", "700"]), "levels"),
        (FakeCDS(levels_out=["850", "700", "500", "300"]), "levels"),
        (FakeCDS(level_units="Pa"), "expected hPa"),
    ],
)
def test_received_levels_must_be_the_requested_ones(tmp_path, fake, message):
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--download"]
    code, _ = run(tmp_path, argv, fake)
    piece = entry(tmp_path, "pl_20230301_t")
    assert code == 1 and piece["status"] == "failed" and message in piece["damage"][0]["error"]


@pytest.mark.parametrize(
    "expver, ok", [("0001", True), ("0005", False), (None, False)], ids=["final", "era5t", "none"]
)
def test_expver_as_a_coordinate(tmp_path, expver, ok):
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--download"]
    code, _ = run(tmp_path, argv, FakeCDS(expver=expver))
    piece = entry(tmp_path, "pl_20230301_t")
    assert (code == 0) is ok and (piece["status"] == "validated") is ok
    if ok:
        assert piece["checks"]["expver_status"] == "final"


def test_cut_answer_is_retried_and_leaves_no_file_without_sidecar(tmp_path):
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--download"]
    code, fake = run(tmp_path, argv, FakeCDS(corrupt=["2023-03-01"]))
    assert code == 0
    piece = entry(tmp_path, "pl_20230301_t")
    assert piece["attempts"] == 2 and piece["damage"][0]["bytes"] == 3
    root = next((tmp_path / "era5_pressure").iterdir())
    for path in root.rglob("t_*.nc"):
        assert aq.sidecar_matches(path, json.loads(aq.sidecar_path(path).read_text())["request"])


# --- daily with the terrain mask (tests 5-8) -----------------------------------------------


def download_both(folder, start, end, day_definition="offset:-05:00", fake=None):
    period = ["--start", start, "--end", end, "--day-definition", day_definition]
    assert run_sp(folder, [*SMALL, *period, "--download"], fake)[0] == 0
    assert run(folder, [*SMALL, *period, "--download"], fake)[0] == 0
    sp_folder = next((folder / "era5_single").iterdir())
    return period, sp_folder


def daily_file(folder):
    (path,) = folder.glob("era5p_*_daily_t_p850-700-500_*.nc")
    return xr.open_dataset(path, engine="h5netcdf")


def test_daily_masks_levels_below_the_terrain(tmp_path):
    period, sp_folder = download_both(tmp_path, "2023-03-01", "2023-03-02")
    argv = [*SMALL, *period, "--daily", "--surface-pressure", str(sp_folder)]
    assert run(tmp_path, argv)[0] == 0
    with daily_file(tmp_path) as ds:
        mean = ds["t_mean"]
        # first longitude: sp 70 000 Pa, so 850 hPa (85 000 Pa) is below ground; 700 is not
        assert bool(mean.sel(pressure_level=850.0).isel(longitude=0).isnull().all())
        assert bool(mean.sel(pressure_level=700.0).isel(longitude=0).notnull().all())
        assert bool(mean.isel(longitude=slice(1, None)).notnull().all())
        frac = ds["t_below_ground_fraction"]
        assert float(frac.sel(pressure_level=850.0).isel(longitude=0).min()) == 1.0
        assert float(ds["sp_missing_fraction"].max()) == 0.0
        assert ds.sizes["time"] == 2 and int(ds["t_mean_n_samples"].max()) == 24
        # test 7: per level, the same as aggregating that level alone
        assert "pressure_level" in ds["t_max"].dims and ds.attrs["era5_expver_status"] == "final"


def test_one_hour_below_ground_or_missing_sp_invalidates_the_day(tmp_path):
    def sp(times, lat, lon):
        out = np.full((len(times), lat.size, lon.size), 101325.0, "float32")
        for i, t in enumerate(times):
            if t == pd.Timestamp("2023-03-01 12:00"):
                out[i, :, 0] = 70000.0  # one hour below ground at 850 hPa
                out[i, :, 1] = np.nan  # one hour of missing sp
        return out

    period, sp_folder = download_both(tmp_path, "2023-03-01", "2023-03-02", fake=FakeCDS(sp=sp))
    assert run(tmp_path, [*SMALL, *period, "--daily", "--surface-pressure", str(sp_folder)])[0] == 0
    with daily_file(tmp_path) as ds:
        day = ds["t_mean"].sel(time="2023-03-01", pressure_level=850.0)
        assert bool(day.isel(longitude=0).isnull().all())  # 23 of 24 samples
        assert bool(day.isel(longitude=1).isnull().all())  # NaN sp is never above ground
        assert bool(day.isel(longitude=2).notnull().all())
        assert float(ds["sp_missing_fraction"].sel(time="2023-03-01").isel(longitude=1).max()) == (
            pytest.approx(1 / 24)
        )


@pytest.mark.parametrize("day_definition", ["offset:-05:00", "offset:+05:00"])
def test_boundary_days_are_complete_and_trimmed(tmp_path, day_definition):
    period, sp_folder = download_both(tmp_path, "2023-03-01", "2023-03-02", day_definition)
    assert run(tmp_path, [*SMALL, *period, "--daily", "--surface-pressure", str(sp_folder)])[0] == 0
    with daily_file(tmp_path) as ds:
        assert [str(t)[:10] for t in ds["time"].values] == ["2023-03-01", "2023-03-02"]
        assert int(ds["t_mean_n_samples"].isel(longitude=1).min()) == 24


def test_daily_needs_surface_pressure(tmp_path):
    period = ["--start", "2023-03-01", "--end", "2023-03-01"]
    assert run(tmp_path, [*SMALL, *period, "--download"])[0] == 0
    with pytest.raises(SystemExit, match="--surface-pressure"):
        run(tmp_path, [*SMALL, *period, "--daily"])
    with pytest.raises(SystemExit, match="no surface pressure"):
        run(tmp_path, [*SMALL, *period, "--daily", "--surface-pressure", str(tmp_path / "none")])


def test_surface_pressure_on_another_grid_or_hours_is_refused(tmp_path):
    period, sp_folder = download_both(tmp_path, "2023-03-01", "2023-03-01")
    other = ["--start", "2023-03-01", "--end", "2023-03-01"]
    # another box: another grid
    elsewhere = tmp_path / "elsewhere"
    assert (
        run_sp(elsewhere, ["--box", "-12.9", "-12.1", "-80.1", "-79.4", *other, "--download"])[0]
        == 0
    )
    with pytest.raises(SystemExit, match="another grid"):
        run(tmp_path, [*SMALL, *period, "--daily", "--surface-pressure",
                       str(next((elsewhere / "era5_single").iterdir()))])  # fmt: skip
    # a file that no longer matches its sidecar
    (path,) = sp_folder.glob("sp/2023/sp_2023-03-01_2023-03-01.nc")
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="sidecar"):
        run(tmp_path, [*SMALL, *period, "--daily", "--surface-pressure", str(sp_folder)])


def test_derived_per_level(tmp_path):
    period, sp_folder = download_both(tmp_path, "2023-03-01", "2023-03-01")
    argv = [*SMALL, *period, "--surface-pressure", str(sp_folder), "--derived", "theta"]
    assert run(tmp_path, [*argv, "--daily"])[0] == 0
    with daily_file(tmp_path) as ds:
        theta = ds["theta_mean"].sel(pressure_level=500.0).isel(longitude=2)
        t = ds["t_mean"].sel(pressure_level=500.0).isel(longitude=2)
        # theta is linear in T at a fixed level, so its daily mean is theta of the mean
        assert np.allclose(theta, t * 2**0.285691, rtol=1e-4)
    with pytest.raises(SystemExit):
        run(tmp_path, [*SMALL, *period, "--derived", "gh"])  # needs z


# --- review findings (GPT, H1-H4) ------------------------------------------------------------


def test_level_groups_are_the_same_every_month():
    # H1: January and February must share the partition of the levels.
    levels = [str(p) for p in range(1000, 99, -25)]
    requests = pl.plan([-4.0, -82.0, -12.0, -77.0], ["t"], levels, date(2023, 1, 1),
                       date(2023, 2, 28))  # fmt: skip
    months = {}
    for r in requests:
        if r.hours == tuple(range(24)):
            months.setdefault(r.start.month, []).append(r.levels)
    assert months[1] == months[2] and all(r.cost <= 54_000 for r in requests)
    halo = [r.levels for r in requests if r.hours != tuple(range(24))]
    assert sorted(lev for g in halo for lev in g) == sorted(levels)  # no level repeated


def test_level_close_to_but_not_the_requested_one_is_refused(tmp_path):
    # H4: 850.4 hPa is not 850 hPa.
    argv = [*SMALL, "--start", "2023-03-01", "--end", "2023-03-01", "--download"]
    code, _ = run(tmp_path, argv, FakeCDS(levels_out=["850.4", "700", "500"]))
    piece = entry(tmp_path, "pl_20230301_t")
    assert code == 1 and "levels" in piece["damage"][0]["error"]


def test_surface_pressure_files_are_inputs_and_follow_the_expver_policy(tmp_path):
    # H2 and H3.
    period = ["--start", "2023-03-01", "--end", "2023-03-01"]
    fake_sp = FakeCDS(expver="0005")
    assert run_sp(tmp_path, [*SMALL, *period, "--download", "--allow-era5t"], fake_sp)[0] == 0
    assert run(tmp_path, [*SMALL, *period, "--download"])[0] == 0
    sp_folder = next((tmp_path / "era5_single").iterdir())
    daily = [*SMALL, *period, "--daily", "--surface-pressure", str(sp_folder)]
    with pytest.raises(SystemExit, match="refused by the"):
        run(tmp_path, daily)
    assert run(tmp_path, [*daily, "--allow-era5t"])[0] == 0
    with daily_file(tmp_path) as ds:
        assert "preliminary" in ds.attrs["era5_expver_status"]
    # the sp now final: a new download changes the sp file, so the product is rebuilt
    assert run_sp(tmp_path, [*SMALL, *period, "--download"], FakeCDS())[0] == 0
    (path,) = tmp_path.glob("era5p_*_daily_t_*.nc")
    before = path.stat().st_mtime_ns
    assert run(tmp_path, daily)[0] == 0
    assert path.stat().st_mtime_ns != before
    with daily_file(tmp_path) as ds:
        assert ds.attrs["era5_expver_status"] == "final"
    (manifest,) = sorted(tmp_path.glob("era5p_daily_manifest_*.json"))[-1:]
    files = json.loads(manifest.read_text())["surface_pressure_files"]
    assert sorted(files) == [  # the day and the boundary hours of the next UTC day
        "sp/2023/sp_2023-03-01_2023-03-01.nc",
        "sp/2023/sp_2023-03-02_2023-03-02_h00-04.nc",
    ]
    assert {f["expver_status"] for f in files.values()} == {"final"}


def test_crossing_box_with_surface_pressure_in_two_parts(tmp_path):
    # Revision 0, B5: levels and sp are both requested in two parts and joined west to east.
    crossing = ["--box", "25.1", "25.4", "179.6", "-179.6"]
    period = ["--start", "2023-03-01", "--end", "2023-03-01"]
    assert run_sp(tmp_path, [*crossing, *period, "--download"])[0] == 0
    assert run(tmp_path, [*crossing, *period, "--download"])[0] == 0
    root = next((tmp_path / "era5_pressure").iterdir())
    names = sorted(p.name for p in root.rglob("t_*.nc"))
    assert any(n.endswith("_w.nc") for n in names) and any(n.endswith("_e.nc") for n in names)
    sp_folder = next((tmp_path / "era5_single").iterdir())
    argv = [*crossing, *period, "--daily", "--surface-pressure", str(sp_folder)]
    assert run(tmp_path, argv)[0] == 0
    with daily_file(tmp_path) as ds:
        assert ds["longitude"].values.tolist() == [179.5, 179.75, 180.0, -179.75, -179.5]
        assert bool(ds["t_mean"].sel(pressure_level=500.0).notnull().all())


def test_daily_blocks_and_compression_do_not_change_the_product(tmp_path, capsys):
    # External review 2026-11-04: month blocks (default) and year blocks give the same
    # values across a month end; another encoding rewrites the product with the same content.
    period, sp_folder = download_both(tmp_path, "2023-02-27", "2023-03-02")
    argv = [*SMALL, *period, "--daily", "--surface-pressure", str(sp_folder), "--derived", "theta"]
    assert run(tmp_path, [*argv, "--daily-block", "year"])[0] == 0
    (path,) = tmp_path.glob("era5p_*_daily_t_*.nc")
    content = content_sha256(path)
    path.rename(tmp_path / "year.nc")
    ledger = aq.Ledger(next((tmp_path / "era5_pressure").iterdir()) / "ledger.json")
    ledger.data["products"] = {}
    ledger.save()
    assert run(tmp_path, argv)[0] == 0
    assert content_sha256(path) == content and aq.file_encoding(path) == "zlib-4"
    capsys.readouterr()
    assert run(tmp_path, [*argv, "--no-compress"])[0] == 0
    assert ": reencoded" in capsys.readouterr().out
    assert content_sha256(path) == content and aq.file_encoding(path) == "none"
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 4 and ds.attrs["era5_expver_status"] == "final"
