# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""tools/regress_data_v1.py: the CDS replay and the offline regression of data-v1.

The full regression reads the retained originals (``../xhw/data``, several GB) and runs for
tens of minutes; it runs only when ``XHW_DATA_V1_ORIGINALS`` names that folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "examples"))

import regress_data_v1 as rd  # noqa: E402
from content_hash import content_sha256  # noqa: E402

LAT = np.arange(-4.0, -12.0 - 1e-9, -0.25)
LON = np.arange(-82.0, -77.0 + 1e-9, 0.25)


def fake_originals(folder: Path, first="2023-03-01", last="2023-03-05 23:00") -> Path:
    """Year files in the registered layout (valid_time, the Peru grid), with a few days only."""
    hourly = folder / rd.ERA5_FOLDER
    hourly.mkdir(parents=True)
    times = pd.date_range(first, last, freq="1h")
    rng = np.random.default_rng(7)
    shape = (times.size, LAT.size, LON.size)
    ds = xr.Dataset(
        {v: (("valid_time", "latitude", "longitude"),
             (290 + 5 * rng.standard_normal(shape)).astype("float32"), {"units": "K"})
         for v in ("t2m", "mx2t", "mn2t")},
        coords={"valid_time": times, "latitude": LAT, "longitude": LON},
    )  # fmt: skip
    ds.to_netcdf(hourly / f"{times[0].year}.nc", engine="h5netcdf")
    lsm = xr.Dataset(
        {"lsm": (("latitude", "longitude"), rng.random((LAT.size, LON.size)).astype("float32"),
                 {"units": "(0 - 1)"})},
        coords={"latitude": LAT, "longitude": LON},
    )  # fmt: skip
    lsm.to_netcdf(hourly / "lsm.nc", engine="h5netcdf")
    return hourly


def legacy_daily(hourly: Path, first: str, last: str, target: Path) -> Path:
    """The registered pipeline (build_daily at 58d2def) applied to the same year file."""
    import xhw

    (year,) = [p for p in hourly.glob("2*.nc")]
    with xr.open_dataset(year, engine="h5netcdf") as opened:
        ds = opened.rename({"valid_time": "time"}).load()
    out = [
        xhw.aggregate_daily(
            ds, variable=v, statistic=s, sampling=m, input_clock="UTC",
            day_definition="offset:-05:00", name=n,
        )
        for v, s, m, n in (
            ("mx2t", "max", "period_ending", "tx"),
            ("mn2t", "min", "period_ending", "tn"),
            ("t2m", "max", "instantaneous", "tx_t2m"),
            ("t2m", "min", "instantaneous", "tn_t2m"),
        )
    ]  # fmt: skip
    days = xr.merge(out, compat="override").sel(time=slice(first, last))
    with xr.open_dataset(hourly / "lsm.nc", engine="h5netcdf") as lsm:
        days["lsm"] = lsm["lsm"].load()
    days.to_netcdf(target, engine="h5netcdf")
    return target


def test_replayed_cds_gives_the_registered_pipeline(tmp_path):
    hourly = fake_originals(tmp_path / "originals")
    product = rd.replay_era5(
        tmp_path / "originals", tmp_path / "work", "2023-03-02", "2023-03-04", workers=2
    )
    assert product.name == "era5_peru_daily_20230302_20230304.nc"
    legacy = legacy_daily(hourly, "2023-03-02", "2023-03-04", tmp_path / "legacy.nc")
    assert content_sha256(product) == content_sha256(legacy)
    with xr.open_dataset(product, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 3 and int(ds["tx_n_samples"].min()) == 24
    # GPT review F1: a reused folder would compare the old product, not the current code
    with pytest.raises(SystemExit, match="previous replay"):
        rd.replay_era5(tmp_path / "originals", tmp_path / "work", "2023-03-02", "2023-03-04")


def test_replay_answers_only_what_the_originals_hold(tmp_path):
    client = rd.ReplayCDS(fake_originals(tmp_path))
    body = {
        "variable": ["2m_temperature"], "year": ["2023"], "month": ["03"], "day": ["01"],
        "time": ["00:00", "01:00"], "area": [-4.0, -82.0, -12.0, -77.0],
    }  # fmt: skip
    ds = client.answer(body)
    assert ds.sizes == {"valid_time": 2, "latitude": 33, "longitude": 21}
    with xr.open_dataset(tmp_path / rd.ERA5_FOLDER / "2023.nc", engine="h5netcdf") as src:
        assert np.array_equal(ds["t2m"].values, src["t2m"].values[:2])
    with pytest.raises(RuntimeError, match="hours missing"):
        client.answer({**body, "day": ["06"]})
    with pytest.raises(RuntimeError, match="do not hold the area"):
        client.answer({**body, "area": [-3.0, -82.0, -12.0, -77.0]})
    with pytest.raises(FileNotFoundError):
        client.answer({**body, "year": ["2024"]})
    client.close()


def test_integrity_reports_missing_originals_as_failures(tmp_path):
    results = rd.integrity(tmp_path)
    assert results and all(r["status"] == "fail" and r["got"] is None for r in results)


@pytest.mark.skipif(
    not os.environ.get("XHW_DATA_V1_ORIGINALS"),
    reason="set XHW_DATA_V1_ORIGINALS to the retained originals (slow, several GB)",
)
def test_data_v1_is_rebuilt_from_the_originals(tmp_path):
    report = rd.run(Path(os.environ["XHW_DATA_V1_ORIGINALS"]), tmp_path)
    assert report["failed"] == 0, [r for r in report["results"] if r["status"] == "fail"]
