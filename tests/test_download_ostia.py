# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of download_ostia.py and the yearly join, with a fake copernicusmarine.

The fake follows what revision 5 verified with copernicusmarine 2.5.0: a period the product
covers in part is clipped (``inside``/``outside``) or rejected with
``CoordinatesOutOfDatasetBounds`` (``strict-inside``); ``outside`` returns the smallest grid
containing the box. Cases R1-R3 of revision 5 (appendix A) are asserted here.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _acquire as aq  # noqa: E402
import download_ostia as d  # noqa: E402
import make_test_datasets as m  # noqa: E402
from content_hash import content_sha256  # noqa: E402

LAT = np.round(-89.975 + 0.05 * np.arange(3600), 3)
LON = np.round(-179.975 + 0.05 * np.arange(7200), 3)
SMALL = ["--box", "-5", "-4", "-78", "-77"]  # aligned: 20 x 20 centres inside
UNALIGNED = ["--box", "-5.01", "-3.99", "-78.01", "-76.99"]


class CoordinatesOutOfDatasetBounds(Exception):
    pass


class DatasetVersionNotFound(Exception):
    pass


class FakeClient:
    """``subset`` of copernicusmarine on the OSTIA grid, writing int16 kelvin like the service."""

    def __init__(self, first="1981-10-01", last="2026-06-30"):
        self.first, self.last = pd.Timestamp(first), pd.Timestamp(last)
        self.calls: list[dict] = []
        self.cuts = 0  # downloads that leave a truncated file and raise ConnectionError
        self.damaged = 0  # downloads that leave an unreadable file and return normally
        self.drop_day: str | None = None  # a day missing in the middle of the answer
        self.method_override: str | None = None  # answer with another selection method
        self.raise_error: Exception | None = None
        self.fail_east = False  # downloads of the part east of the antimeridian fail
        self.fail_if = lambda cfg: False  # any other download that should fail
        self.east_scale = 0.01  # scale_factor written for the east part
        self.float32 = False  # select on float32 coordinates, as stored by the service
        self.time_units = ["seconds since 1981-01-01"]  # cycled over downloads

    @property
    def downloads(self) -> list[dict]:
        return [c for c in self.calls if not c.get("dry_run")]

    def _select(self, cfg):
        method = self.method_override or cfg.get("coordinates_selection_method", "inside")
        policy = "enclose" if method == "outside" else "inside"
        if cfg["minimum_longitude"] > cfg["maximum_longitude"]:  # behaviour unverified
            raise AssertionError("the script must not rely on min > max longitude")
        grid_lat, grid_lon = (
            (LAT.astype("f4").astype(float), LON.astype("f4").astype(float))
            if self.float32
            else (LAT, LON)
        )
        tight = dict(tol=0.0) if self.float32 else {}  # a literal comparison
        lat = LAT[
            aq.select_indices(
                grid_lat, cfg["minimum_latitude"], cfg["maximum_latitude"], policy, **tight
            )
        ]
        lon = LON[
            aq.select_indices(
                grid_lon, cfg["minimum_longitude"], cfg["maximum_longitude"], policy, **tight
            )
        ]
        start = pd.Timestamp(cfg["start_datetime"][:10])
        end = pd.Timestamp(cfg["end_datetime"][:10])
        if end < self.first or start > self.last:
            raise CoordinatesOutOfDatasetBounds("the requested period is outside the dataset")
        if method == "strict-inside" and (start < self.first or end > self.last):
            raise CoordinatesOutOfDatasetBounds("strict-inside: period exceeds the dataset")
        # float64 bounds compared literally (the strictest case for re-review F3-R1)
        too_far = cfg["minimum_longitude"] < d.LON_FIRST or cfg["maximum_longitude"] > d.LON_LAST
        if method == "strict-inside" and too_far:
            raise CoordinatesOutOfDatasetBounds("strict-inside: longitude exceeds the dataset")
        return lat, lon, pd.date_range(max(start, self.first), min(end, self.last), freq="D")

    def subset(self, **cfg):
        self.calls.append(cfg)
        if self.raise_error is not None:
            raise self.raise_error
        east = cfg["minimum_longitude"] <= d.LON_FIRST + 1e-4
        if not cfg.get("dry_run") and ((self.fail_east and east) or self.fail_if(cfg)):
            raise ConnectionError("simulated cut of the east part")
        lat, lon, times = self._select(cfg)
        extent = [
            SimpleNamespace(
                coordinate_id="time",
                minimum=f"{times[0]:%Y-%m-%dT%H:%M:%SZ}",
                maximum=f"{times[-1]:%Y-%m-%dT%H:%M:%SZ}",
                unit="iso8601",
            ),
            SimpleNamespace(
                coordinate_id="latitude", minimum=lat[0], maximum=lat[-1], unit="degrees_north"
            ),
        ]
        if cfg.get("dry_run"):
            return SimpleNamespace(coordinates_extent=extent, file_size=1.0)
        out = Path(cfg["output_directory"]) / cfg["output_filename"]
        if self.cuts:
            self.cuts -= 1
            out.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 1000)
            raise ConnectionError("simulated network cut")
        if self.damaged:
            self.damaged -= 1
            out.write_bytes(b"not a netcdf")
            return SimpleNamespace(coordinates_extent=extent)
        if self.drop_day is not None:
            times = times[times != pd.Timestamp(self.drop_day)]
        scale = self.east_scale if east else 0.01
        doy = np.asarray(times.dayofyear)[:, None, None]
        values = (doy * 7 + np.arange(lat.size)[:, None] + 3 * np.arange(lon.size)) % 3000
        xr.Dataset(
            {"analysed_sst": (("time", "latitude", "longitude"), (values * scale + 273.15))},
            coords={"time": times, "latitude": lat.astype("f4"), "longitude": lon.astype("f4")},
        ).to_netcdf(
            out,
            engine="h5netcdf",
            encoding={
                "analysed_sst": {
                    "dtype": "int16",
                    "scale_factor": scale,
                    "add_offset": 273.15,
                    "_FillValue": -32768,
                },
                "time": {
                    "units": self.time_units[len(self.downloads) % len(self.time_units)],
                    "dtype": "int64",
                },
            },
        )
        return SimpleNamespace(coordinates_extent=extent)


def run(tmp_path, client, *args):
    return d.main(["--data-dir", str(tmp_path), *args], client=client)


def folder(tmp_path, policy="enclose"):
    key = "box_" + aq.request_id(aq.Box(-5.0, -4.0, -78.0, -77.0).as_dict(), length=8)
    return tmp_path / d.folder_name(key, policy)


def days_of(path):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        t = ds["time"].values.astype("datetime64[D]")
    return t.size, str(t[0]), str(t[-1])


def manifests(target):
    return [
        json.loads((target / e["file"]).read_text())
        for e in json.loads((target / "manifests.json").read_text())
    ]


# --- plan ----------------------------------------------------------------------------------


def test_print_only_needs_no_client(tmp_path, capsys):
    assert run(tmp_path, None) == 0
    out = capsys.readouterr().out
    assert '"coordinates_selection_method": "outside"' in out  # enclose, B3
    assert '"dataset_version": "202003"' in out
    assert "analysed_sst_1991.nc" in out and "analysed_sst_2025.nc" in out
    assert (
        sum(line.startswith("block_") for line in out.splitlines()) == 1
    )  # one request for the whole period (decision 1)
    assert not tmp_path.joinpath("ostia_peru_enclose").exists()


def test_plan_blocks_spans_and_names():
    assert d.blocks(date(1991, 1, 1), date(1991, 12, 31), None) == [
        (date(1991, 1, 1), date(1991, 12, 31))
    ]
    assert d.blocks(date(1991, 1, 1), date(1991, 1, 10), 4)[-1] == (
        date(1991, 1, 9),
        date(1991, 1, 10),
    )
    assert d.spans(date(1991, 3, 1), date(1992, 2, 1)) == [
        (date(1991, 3, 1), date(1991, 12, 31)),
        (date(1992, 1, 1), date(1992, 2, 1)),
    ]
    assert d.file_name(date(1991, 1, 1), date(1991, 12, 31)) == "analysed_sst_1991.nc"
    assert d.file_name(date(2026, 1, 1), date(2026, 12, 31), True) == "analysed_sst_2026_partial.nc"
    assert d.file_name(date(1991, 3, 1), date(1991, 3, 31)) == "analysed_sst_19910301_19910331.nc"
    space = d.spatial_request(aq.Box(-5, 5, 190, 230), "inside")  # 0..360 input
    assert (space["minimum_longitude"], space["maximum_longitude"]) == (-170.0, -130.0)
    assert space["coordinates_selection_method"] == "inside"


@pytest.mark.parametrize(
    "args",
    [
        ["--box", "-5", "5", "0", "360"],  # global
        ["--start", "1981-09-30", "--end", "1981-12-31"],  # before the series starts
        ["--start", "1992-01-01", "--end", "1991-12-31"],
        ["--block-days", "0"],
        ["--block-days", "100", "--allow-partial"],  # review F1: one request only
    ],
)
def test_invalid_arguments_are_rejected(tmp_path, args):
    with pytest.raises(SystemExit) as exc:
        run(tmp_path, FakeClient(), *args)
    assert exc.value.code == 2


# --- download, resumption (R1) and splitting -------------------------------------------------


def test_one_request_split_into_complete_years(tmp_path):
    client = FakeClient()
    code = run(
        tmp_path, client, *SMALL, "--start", "1991-01-01", "--end", "1992-12-31", "--download"
    )
    assert code == 0
    assert len(client.downloads) == 1  # decision 1
    assert client.downloads[0]["coordinates_selection_method"] == "outside"
    assert "skip_existing" not in client.downloads[0]  # R5
    strict = [c for c in client.calls if c.get("coordinates_selection_method") == "strict-inside"]
    assert strict and all(c["dry_run"] for c in strict)  # the time gate never downloads
    target = folder(tmp_path)
    assert days_of(target / "analysed_sst_1991.nc") == (365, "1991-01-01", "1991-12-31")
    assert days_of(target / "analysed_sst_1992.nc") == (366, "1992-01-01", "1992-12-31")
    assert not list((target / "blocks").glob("*.nc"))  # blocks removed after splitting
    (manifest,) = manifests(target)
    assert manifest["dataset_version"] == "202003" and manifest["units"] == "kelvin"
    assert manifest["box"]["ok"] and manifest["box"]["policy"] == "enclose"
    assert manifest["days"]["analysed_sst_1992.nc"]["days"] == 366
    assert manifest["partial_files"] == []
    assert sorted(manifest["files"]) == ["analysed_sst_1991.nc", "analysed_sst_1992.nc"]
    assert next(iter(manifest["client_extent"].values()))[0]["coordinate_id"] == "time"
    # The stored values keep the service's packing (int16, scale and offset).
    with xr.open_dataset(target / "analysed_sst_1991.nc", engine="h5netcdf", **d.RAW) as raw:
        assert raw["analysed_sst"].dtype == np.int16
        assert raw["analysed_sst"].attrs["scale_factor"] == pytest.approx(0.01)


def test_cut_then_rerun_resumes_and_a_third_run_downloads_nothing(tmp_path, capsys):
    client = FakeClient()
    client.cuts = 2  # the first attempt and its single retry both fail
    args = [*SMALL, "--start", "1991-01-01", "--end", "1993-12-31", "--download"]
    assert run(tmp_path, client, *args) == 1
    target = folder(tmp_path)
    assert not list(target.glob("analysed_sst_*.nc"))  # no truncated file under a final name
    ledger = json.loads((target / "ledger.json").read_text())
    (piece,) = ledger["pieces"].values()
    assert piece["status"] == "failed" and "ConnectionError" in piece["error"]
    assert "block_" in capsys.readouterr().out
    assert run(tmp_path, client, *args) == 0  # R1: the same command continues
    assert len(client.downloads) == 3
    assert sorted(p.name for p in target.glob("analysed_sst_*.nc")) == [
        "analysed_sst_1991.nc",
        "analysed_sst_1992.nc",
        "analysed_sst_1993.nc",
    ]
    before = len(client.calls)
    assert run(tmp_path, None, *args) == 0  # nothing to do: no client needed
    assert len(client.calls) == before
    assert len(json.loads((target / "manifests.json").read_text())) == 2  # never overwritten


def test_blocks_give_the_same_years_as_one_request(tmp_path):
    common = [*SMALL, "--start", "1991-01-01", "--end", "1992-12-31", "--download"]
    one, many = FakeClient(), FakeClient()
    assert run(tmp_path / "one", one, *common) == 0
    assert run(tmp_path / "many", many, *common, "--block-days", "200") == 0
    assert len(many.downloads) == 4  # 731 days in blocks of 200
    for name in ("analysed_sst_1991.nc", "analysed_sst_1992.nc"):
        a = content_sha256(folder(tmp_path / "one") / name)
        b = content_sha256(folder(tmp_path / "many") / name)
        assert a == b  # the year crossing two blocks is the same values


def test_a_lost_year_is_rebuilt_from_a_new_download(tmp_path):
    client = FakeClient()
    args = [*SMALL, "--start", "1991-01-01", "--end", "1992-12-31", "--download"]
    assert run(tmp_path, client, *args) == 0
    (folder(tmp_path) / "analysed_sst_1992.nc").unlink()
    assert run(tmp_path, client, *args) == 0
    assert len(client.downloads) == 2  # the block was deleted, so it is fetched again
    assert days_of(folder(tmp_path) / "analysed_sst_1992.nc")[0] == 366


def test_damaged_file_is_retried_once_and_recorded(tmp_path):
    client = FakeClient()
    client.damaged = 1
    args = [*SMALL, "--start", "1991-01-01", "--end", "1991-12-31", "--download"]
    assert run(tmp_path, client, *args) == 0
    (piece,) = json.loads((folder(tmp_path) / "ledger.json").read_text())["pieces"].values()
    assert piece["status"] == "validated"
    assert piece["damage"][0]["bytes"] == len(b"not a netcdf")


def test_permanent_client_error_is_not_retried(tmp_path, capsys):
    client = FakeClient()
    client.raise_error = DatasetVersionNotFound("202003 not found; available: 202512")
    args = [*SMALL, "--start", "1991-01-01", "--end", "1991-12-31", "--download"]
    assert run(tmp_path, client, *args) == 1
    assert len(client.calls) == 1  # the preflight call; no retry and no download
    assert "a new version is a new registered entry" in capsys.readouterr().out
    assert d.classify(DatasetVersionNotFound()) == aq.PERMANENT
    assert d.classify(ConnectionError()) == aq.TRANSIENT


# --- edge years (R2) and validation ------------------------------------------------------


def test_edge_years_are_rejected_before_downloading(tmp_path, capsys):
    client = FakeClient()
    with pytest.raises(SystemExit):  # the start of the series is fixed: rejected at once
        run(tmp_path, client, *SMALL, "--start", "1981-01-01", "--end", "1981-12-31")
    assert "first available day 1981-10-01" in capsys.readouterr().err
    # The end moves, so only the service knows it: strict-inside rejects without data.
    args = [*SMALL, "--start", "2026-01-01", "--end", "2026-12-31", "--download"]
    assert run(tmp_path, client, *args) == 1
    assert client.downloads == []
    out = capsys.readouterr().out
    assert "covers only 2026-01-01T00:00:00Z..2026-06-30T00:00:00Z" in out
    assert "--allow-partial" in out
    assert not list(folder(tmp_path).glob("analysed_sst_*.nc"))
    # The covered part of a year is a complete request, not a partial file.
    args = [*SMALL, "--start", "1981-10-01", "--end", "1981-12-31", "--download"]
    assert run(tmp_path, client, *args) == 0
    assert days_of(folder(tmp_path) / "analysed_sst_19811001_19811231.nc")[0] == 92


def test_allow_partial_keeps_edge_years_with_a_suffix(tmp_path):
    client = FakeClient()
    args = [*SMALL, "--start", "2025-01-01", "--end", "2026-12-31", "--download", "--allow-partial"]
    assert run(tmp_path, client, *args) == 0
    target = folder(tmp_path)
    assert days_of(target / "analysed_sst_2025.nc")[0] == 365
    assert days_of(target / "analysed_sst_2026_partial.nc") == (181, "2026-01-01", "2026-06-30")
    (manifest,) = manifests(target)
    assert manifest["partial_files"] == ["analysed_sst_2026_partial.nc"]
    assert manifest["days"]["analysed_sst_2026_partial.nc"]["missing_days"]["end"] == 184
    # Once the series grows, the same command replaces the partial file by the full year.
    client.last = pd.Timestamp("2026-12-31")
    assert run(tmp_path, client, *args) == 0
    assert days_of(target / "analysed_sst_2026.nc")[0] == 365
    assert not (target / "analysed_sst_2026_partial.nc").exists()
    # 1981 begins before the series: kept as a partial year only on request.
    args = [*SMALL, "--start", "1981-01-01", "--end", "1981-12-31", "--download"]
    assert run(tmp_path / "early", client, *args, "--allow-partial") == 0
    partial = folder(tmp_path / "early") / "analysed_sst_1981_partial.nc"
    assert days_of(partial) == (92, "1981-10-01", "1981-12-31")


def test_a_gap_inside_the_period_fails_even_with_allow_partial(tmp_path, capsys):
    client = FakeClient()
    client.drop_day = "1991-06-15"
    args = [*SMALL, "--start", "1991-01-01", "--end", "1991-12-31", "--download"]
    assert run(tmp_path, client, *args, "--allow-partial") == 1
    assert len(client.downloads) == 1  # a data gap is not a damaged file: no retry
    assert "not daily and continuous" in capsys.readouterr().out
    assert not list(folder(tmp_path).glob("analysed_sst_*.nc"))


def test_enclose_contains_an_unaligned_box_and_inside_does_not(tmp_path):
    client = FakeClient()
    period = ["--start", "1991-01-01", "--end", "1991-01-31", "--download"]
    assert run(tmp_path, client, *UNALIGNED, *period) == 0
    (manifest,) = manifests(next(tmp_path.glob("ostia_box_*_enclose")))
    assert manifest["box"]["latitude"]["nodes"] == pytest.approx([-5.025, -3.975], abs=1e-5)
    assert run(tmp_path, client, *UNALIGNED, *period, "--box-policy", "inside") == 0
    (inside,) = manifests(next(p for p in tmp_path.glob("ostia_box_*") if p.name[-7:] != "enclose"))
    assert inside["box"]["latitude"]["nodes"] == pytest.approx([-4.975, -4.025], abs=1e-5)


def test_a_grid_that_does_not_contain_the_box_fails_coverage(tmp_path, capsys):
    client = FakeClient()
    client.method_override = "inside"  # the service ignores "outside"
    args = [*UNALIGNED, "--start", "1991-01-01", "--end", "1991-01-31", "--download"]
    assert run(tmp_path, client, *args) == 1
    assert "do not enclose" in capsys.readouterr().out


def test_dry_run_checks_every_block_without_data(tmp_path, capsys):
    client = FakeClient()
    assert run(tmp_path, client, *SMALL, "--dry-run") == 0
    assert client.calls and all(c["dry_run"] for c in client.calls)
    assert (
        run(tmp_path, client, *SMALL, "--start", "2026-01-01", "--end", "2026-12-31", "--dry-run")
        == 1
    )
    assert "2026-06-30" in capsys.readouterr().err  # the real coverage


# --- yearly join (R3) ------------------------------------------------------------------------


def yearly(folder: Path, years, drop: str | None = None) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        t = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if drop is not None:
            t = t[t != pd.Timestamp(drop)]
        path = folder / f"analysed_sst_{year}.nc"
        xr.Dataset(
            {"analysed_sst": (("time", "latitude", "longitude"), np.full((len(t), 2, 2), 290.0))},
            coords={"time": t, "latitude": [-11.975, -4.025], "longitude": [-81.975, -77.025]},
        ).to_netcdf(
            path, engine="h5netcdf", encoding={"time": {"units": "seconds since 1981-01-01"}}
        )
        paths.append(path)
    return paths


def test_join_names_the_output_after_the_years_joined(tmp_path):
    parts = yearly(tmp_path / "in", (1991, 1992))
    out, info = m.join_in_time(parts, tmp_path / "out", "ostia_peru", "joined", [1991, 1992])
    assert out.name == "ostia_peru_1991_1992.nc"
    assert sorted(info["derived_from"]["files"]) == ["analysed_sst_1991.nc", "analysed_sst_1992.nc"]
    assert days_of(out) == (731, "1991-01-01", "1992-12-31")


def test_join_rejects_a_missing_year_a_gap_and_a_partial_year(tmp_path):
    parts = yearly(tmp_path / "a", (1991, 1992, 1994))
    with pytest.raises(RuntimeError, match="1993"):  # 1994 given for 1993
        m.join_in_time(parts, tmp_path, "ostia_peru", "joined", [1991, 1992, 1993])
    with pytest.raises(RuntimeError, match="not consecutive"):
        m.join_in_time(parts, tmp_path, "ostia_peru", "joined", [1991, 1992, 1994])
    missing = [*parts[:2], tmp_path / "a" / "analysed_sst_1993.nc"]
    with pytest.raises(FileNotFoundError, match="analysed_sst_1993.nc"):
        m.join_in_time(missing, tmp_path, "ostia_peru", "joined", [1991, 1992, 1993])
    gap = yearly(tmp_path / "b", (1991,), drop="1991-06-15")
    with pytest.raises(RuntimeError, match="every day of 1991"):
        m.join_in_time(gap, tmp_path, "ostia_peru", "joined", [1991])
    assert not list(tmp_path.glob("ostia_peru_*.nc"))


def test_build_requires_every_registered_ostia_year(tmp_path):
    yearly(tmp_path / "data" / "ostia_peru", (1991, 1992))
    with pytest.raises(FileNotFoundError, match="analysed_sst_1993.nc"):
        m.build(tmp_path / "data", tmp_path / "out", only=("ostia",))


def test_blocks_with_another_time_epoch_are_not_joined(tmp_path, capsys):
    client = FakeClient()  # review F2: raw times are joined as stored
    client.time_units = ["seconds since 1981-01-01", "days since 1991-01-01"]
    args = [*SMALL, "--start", "1991-01-01", "--end", "1991-12-31", "--block-days", "200"]
    assert run(tmp_path, client, *args, "--download") == 1
    assert "blocks encode time differently" in capsys.readouterr().out
    assert not (folder(tmp_path) / "analysed_sst_1991.nc").exists()


# --- antimeridian (revision 0, B5) -----------------------------------------------------------


def joined_lon(path):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return np.asarray(ds["longitude"].values, dtype=float)


def test_lon_parts_cut_at_the_outermost_centres():
    assert d.lon_parts(aq.Box(0, 1, 150, -150), "enclose") == [
        aq.Box(0, 1, 150, d.LON_LAST),
        aq.Box(0, 1, d.LON_FIRST, -150),
    ]
    assert d.lon_parts(aq.Box(0, 1, 150, 210), "enclose")[1] == aq.Box(0, 1, d.LON_FIRST, -150)
    assert d.lon_parts(aq.Box(0, 1, -82, -77), "enclose") == [aq.Box(0, 1, -82, -77)]
    # inside: a part with no centre is left out; enclose still takes the outermost centre
    assert d.lon_parts(aq.Box(0, 1, 179.99, -170), "inside") == [aq.Box(0, 1, d.LON_FIRST32, -170)]
    assert d.lon_parts(aq.Box(0, 1, 179.99, -170), "enclose")[0] == aq.Box(
        0, 1, d.LON_LAST, d.LON_LAST
    )


@pytest.mark.parametrize(
    ("args", "west", "east"),
    [
        (["--region", "npacific"], 159.975, -129.975),  # 160E-130W, enclose
        (["--box", "0", "0.5", "150", "-150"], 149.975, -149.975),
        (["--box", "0", "0.5", "150.01", "-150.01", "--box-policy", "inside"], 150.025, -150.025),
    ],
)
def test_antimeridian_box_is_joined_west_to_east(tmp_path, args, west, east):
    client = FakeClient()
    period = ["--start", "1991-01-01", "--end", "1991-01-02", "--download"]
    assert run(tmp_path, client, *args, *period) == 0
    downloads = client.downloads
    assert len(downloads) == 2  # one request per part, each with minimum < maximum
    edges = sorted((c["minimum_longitude"], c["maximum_longitude"]) for c in downloads)
    assert edges[0][0] == pytest.approx(d.LON_FIRST, abs=1e-5)
    assert edges[1][1] == pytest.approx(d.LON_LAST, abs=1e-5)
    (target,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    lon = joined_lon(target / "analysed_sst_19910101_19910102.nc")
    assert lon[0] == pytest.approx(west, abs=1e-4) and lon[-1] == pytest.approx(east, abs=1e-4)
    unwrapped = lon[0] + (lon - lon[0]) % 360
    assert np.allclose(np.diff(unwrapped), 0.05, atol=1e-4)  # no gap nor repeat at the seam
    assert np.sum(np.isclose(lon, d.LON_LAST, atol=1e-4)) == 1
    assert np.sum(np.isclose(lon, d.LON_FIRST, atol=1e-4)) == 1
    (manifest,) = manifests(target)
    assert manifest["box"]["ok"] and len(manifest["requests"]) == 2
    assert [p["ok"] for p in manifest["part_coverage"]] == [True, True]
    assert len(manifest["request"]["longitude_parts"]) == 2
    # The values of each part are kept: the joined file equals the parts side by side.
    with xr.open_dataset(target / "analysed_sst_19910101_19910102.nc", engine="h5netcdf") as ds:
        sst = ds["analysed_sst"].values
    n_west = int(np.sum(lon > 0))
    assert sst.shape[-1] == lon.size and np.isfinite(sst[..., n_west - 1 : n_west + 1]).all()


def test_one_failing_part_is_reported_and_resumed(tmp_path, capsys):
    client = FakeClient()
    client.fail_east = True
    args = ["--box", "0", "0.5", "150", "-150", "--start", "1991-01-01", "--end", "1991-01-02"]
    assert run(tmp_path, client, *args, "--download") == 1
    out = capsys.readouterr().out
    assert "simulated cut of the east part" in out
    (target,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert not list(target.glob("analysed_sst_*.nc"))
    pieces = json.loads((target / "ledger.json").read_text())["pieces"]
    statuses = sorted(p["status"] for p in pieces.values())
    assert statuses == ["failed", "validated"]
    client.fail_east = False
    before = len(client.downloads)
    assert run(tmp_path, client, *args, "--download") == 0
    assert len(client.downloads) == before + 1  # only the failed part is requested again
    assert joined_lon(target / "analysed_sst_19910101_19910102.nc").size == 1202


def test_antimeridian_blocks_and_partial_years(tmp_path):
    client = FakeClient()
    box = ["--box", "0", "0.5", "179", "-179"]
    args = [*box, "--start", "1991-12-30", "--end", "1992-01-02", "--block-days", "2", "--download"]
    assert run(tmp_path / "blocks", client, *args) == 0
    assert len(client.downloads) == 4  # 2 periods x 2 parts
    target = next((tmp_path / "blocks").iterdir())
    assert days_of(target / "analysed_sst_19920101_19920102.nc")[0] == 2
    assert joined_lon(target / "analysed_sst_19911230_19911231.nc").size == 42
    args = [*box, "--start", "2026-06-29", "--end", "2026-07-02", "--download", "--allow-partial"]
    assert run(tmp_path / "partial", client, *args) == 0
    target = next((tmp_path / "partial").iterdir())
    assert days_of(target / "analysed_sst_20260629_20260702_partial.nc") == (
        2,
        "2026-06-29",
        "2026-06-30",
    )


def test_seam_check_rejects_a_missing_or_repeated_longitude():
    run_ = SimpleNamespace(
        box=aq.Box(0, 0.5, 179.9, -179.9), args=SimpleNamespace(box_policy="enclose")
    )
    lat = np.round(np.arange(-0.025, 0.55, 0.05), 3)  # encloses 0..0.5
    west, east = np.array([179.875, 179.925, 179.975]), np.array([-179.975, -179.925, -179.875])
    assert d.Run.joined_coverage(run_, lat, np.concatenate([west, east]))["ok"]
    for lon in (np.concatenate([west, east[1:]]), np.concatenate([west, [179.975], east])):
        with pytest.raises(aq.ValidationError, match="seam"):
            d.Run.joined_coverage(run_, lat, lon)


def test_parts_stored_differently_are_not_joined(tmp_path, capsys):
    client = FakeClient()  # review 3, F1: raw values joined as stored need the same packing
    client.east_scale = 0.02
    args = ["--box", "0", "0.5", "179", "-179", "--start", "1991-01-01", "--end", "1991-01-02"]
    assert run(tmp_path, client, *args, "--download") == 1
    assert "store analysed_sst differently" in capsys.readouterr().out
    assert not list(next(tmp_path.iterdir()).glob("analysed_sst_*.nc"))


def test_part_coverage_comes_from_the_blocks_behind_the_written_files(tmp_path):
    client = FakeClient()  # review 3, F2: the east part of the first period fails
    client.fail_if = lambda cfg: (
        cfg["start_datetime"].startswith("1991-12-31") and cfg["minimum_longitude"] < 0
    )
    args = ["--box", "0", "0.5", "179", "-179", "--start", "1991-12-31", "--end", "1992-01-01"]
    assert run(tmp_path, client, *args, "--block-days", "1", "--download") == 1
    target = next(tmp_path.iterdir())
    (manifest,) = manifests(target)
    assert list(manifest["files"]) == ["analysed_sst_19920101_19920101.nc"]
    assert [c is not None and c["ok"] for c in manifest["part_coverage"]] == [True, True]


@pytest.mark.parametrize("policy", ["inside", "enclose"])
def test_float32_service_coordinates_keep_the_seam_centres(tmp_path, policy):
    client = FakeClient()  # review 3, F3: 179.975 is stored as 179.97500610...
    client.float32 = True
    args = ["--box", "0", "0.5", "179", "-179", "--box-policy", policy]
    assert (
        run(tmp_path, client, *args, "--start", "1991-01-01", "--end", "1991-01-01", "--download")
        == 0
    )
    lon = joined_lon(next(tmp_path.iterdir()) / "analysed_sst_19910101_19910101.nc")
    assert np.sum(np.isclose(lon, d.LON_LAST, atol=1e-4)) == 1
    assert np.sum(np.isclose(lon, d.LON_FIRST, atol=1e-4)) == 1
