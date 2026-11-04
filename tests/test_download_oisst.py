# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of download_oisst.py with synthetic snapshots (revision 3, appendix A).

No network and no icechunk: ``opener`` returns an in-memory ``Dataset`` with the store's
layout (``time``, ``zlev``, ``lat``, ``lon``, ``sst``, ``preliminary``), and icechunk errors
are imitated by classes with the same name and ``kind``.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _acquire as aq  # noqa: E402
import download_oisst as d  # noqa: E402
from content_hash import registered  # noqa: E402

LAT, LON = d.OISST_LAT, d.OISST_LON
#: snapshot -> (last day, sst value, first preliminary day or None)
SNAPSHOTS = {
    "S1": ("2000-12-31", 1.0, None),
    "S2": ("2001-12-31", 2.0, None),
    "S_short": ("2000-06-30", 3.0, None),
    "S_prelim": ("2000-12-31", 4.0, "2000-12-02"),
    "S_final": ("2000-12-31", 5.0, None),
}
NOAA_ATTRS = {"title": "NOAA OISST v2.1", "source": "ICOADS, NCEP_GTS", "history": "NOAA"}
SST_ATTRS = {"units": "Celsius", "long_name": "Daily sea surface temperature", "valid_min": -300}
FAST = d.Retry(retries=3, wait=lambda attempt: 0.0, sleep=lambda seconds: None)


def synthetic(snapshot: str, lat=None, lon=None, first: str = "1999-12-01") -> xr.Dataset:
    last, value, prelim = SNAPSHOTS[snapshot]
    t = pd.date_range(first, last, freq="D")
    lat = LAT[(LAT > -14) & (LAT < -2)] if lat is None else lat
    lon = LON[(LON > 276) & (LON < 285)] if lon is None else lon
    flags = np.zeros(t.size, "i1") if prelim is None else (t >= pd.Timestamp(prelim)).astype("i1")
    return xr.Dataset(
        {
            "sst": (
                ("time", "zlev", "lat", "lon"),
                np.full((t.size, 1, lat.size, lon.size), value, "f4"),
                SST_ATTRS,
            ),
            "preliminary": (("time",), flags),
        },
        coords={"time": t, "zlev": [0.0], "lat": lat, "lon": lon},
        attrs=NOAA_ATTRS,
    )


class Opener:
    """Counts store openings; ``fail`` lists errors to raise on successive openings."""

    def __init__(self, make=synthetic, fail=()):
        self.make, self.fail, self.calls, self.cache = make, list(fail), 0, {}

    def __call__(self, snapshot):
        self.calls += 1
        if self.fail:
            error = self.fail.pop(0)
            if error is not None:
                raise error
        if snapshot not in self.cache:
            self.cache[snapshot] = self.make(snapshot)
        return self.cache[snapshot]


class StorageError(Exception):
    """Same class name and ``kind`` attribute as ``icechunk.StorageError``."""

    def __init__(self, kind: str):
        super().__init__(f"simulated {kind}")
        self.kind = kind


class SnapshotNotFoundError(KeyError):
    pass


@pytest.fixture
def reads(monkeypatch):
    """Count read_strip calls; set ``errors`` to raise before successive reads."""
    state = {"n": 0, "errors": []}
    original = d.read_strip

    def counting(*args, **kwargs):
        state["n"] += 1
        if state["errors"]:
            error = state["errors"].pop(0)
            if error is not None:
                raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(d, "read_strip", counting)
    return state


def run(tmp_path, snapshot, start, end, regions=("peru",), policy="enclose", opener=None, **kw):
    return d.download(
        d.make_targets(regions),
        date.fromisoformat(start),
        date.fromisoformat(end),
        data_dir=tmp_path,
        snapshot=snapshot,
        policy=policy,
        concurrency=None,
        retry=kw.pop("retry", FAST),
        opener=opener or Opener(),
        **kw,
    )


def product_values(result, label="peru") -> dict:
    with xr.open_dataset(result.products[label]["path"], engine="h5netcdf") as ds:
        years = ds["time"].dt.year.values
        return {
            "days": ds.sizes["time"],
            "snapshot": ds.attrs["store_snapshot"],
            "sst": {int(y): float(ds["sst"].values[years == y].mean()) for y in np.unique(years)},
        }


def ledger(tmp_path) -> dict:
    return json.loads((tmp_path / d.LEDGER).read_text(encoding="utf-8"))


# --- selection (B4) and longitudes ------------------------------------------------------


@pytest.mark.parametrize("region", ["peru", "equatorial", "npacific"])
def test_inside_selection_equals_historical_sel_on_the_real_grid(region):
    reference = registered(d.REGISTERED_KEYS[region])
    query = reference["query"]
    grid = xr.Dataset(coords={"lat": LAT, "lon": LON})
    old = grid.sel(lat=slice(*query["lat"]), lon=slice(*query["lon_0_360"]))
    sel = d.select_box(aq.REGIONS[region], LAT, LON, "inside")
    np.testing.assert_array_equal(LAT[sel.lat], old["lat"].values)
    np.testing.assert_array_equal(LON[sel.lon_index()], old["lon"].values)
    assert (LAT[sel.lat].size, LON[sel.lon_index()].size) == (
        reference["sizes"]["lat"],
        reference["sizes"]["lon"],
    )


def test_enclose_adds_the_nodes_beyond_every_edge():
    sel = d.select_box(aq.REGIONS["peru"], LAT, LON, "enclose")
    lat, lon = LAT[sel.lat], LON[sel.lon_index()]
    assert (lat[0], lat[-1], lon[0], lon[-1]) == (-12.125, -3.875, 277.875, 283.125)


def test_saved_longitudes_follow_the_registered_shifts():
    for region, shift, convention in [
        ("peru", -360.0, "-180..180"),
        ("equatorial", -360.0, "-180..180"),
        ("npacific", 0.0, "0..360"),
    ]:
        reference = registered(d.REGISTERED_KEYS[region])["query"].get("lon_shift_applied", -360.0)
        lon = LON[d.select_box(aq.REGIONS[region], LAT, LON, "inside").lon_index()]
        shifts, name = d.output_longitudes(lon)
        assert set(shifts) == {shift} == {reference} and name == convention
    greenwich = LON[d.select_box(aq.Box(0, 1, -10, 10), LAT, LON, "inside").lon_index()]
    shifts, name = d.output_longitudes(greenwich)
    assert name == "-180..180" and np.all(np.diff(greenwich + shifts) > 0)


# --- R1: identity with snapshot ---------------------------------------------------------


def test_r1_another_snapshot_never_reuses_pieces(tmp_path, reads):
    first = run(tmp_path, "S1", "2000-01-01", "2000-12-31")
    assert first.exit_code == 0 and product_values(first)["sst"] == {2000: 1.0}
    second = run(tmp_path, "S2", "2000-01-01", "2001-12-31")
    assert second.exit_code == 0
    values = product_values(second)
    assert values["snapshot"] == "S2" and values["sst"] == {2000: 2.0, 2001: 2.0}
    assert reads["n"] == 12 + 24
    # The same product name from another snapshot is not overwritten.
    third = run(tmp_path, "S2", "2000-01-01", "2000-12-31")
    assert third.exit_code == 1 and "delete it" in third.problems[0]
    assert product_values(first)["sst"] == {2000: 1.0}


def test_r1_same_plan_reuses_validated_pieces_without_opening_the_store(tmp_path, reads):
    run(tmp_path, "S1", "2000-01-01", "2000-12-31")
    opener = Opener()
    again = run(tmp_path, "S1", "2000-01-01", "2000-12-31", opener=opener)
    assert again.exit_code == 0 and opener.calls == 0 and reads["n"] == 12
    assert again.products["peru"]["status"] == "current"


def test_r5_changed_piece_is_read_again(tmp_path, reads):
    run(tmp_path, "S1", "2000-01-01", "2000-12-31", regions=("peru",))
    entry = next(iter(ledger(tmp_path)["pieces"].values()))
    with (tmp_path / entry["piece"]["path"]).open("ab") as handle:
        handle.write(b"x")
    run(tmp_path, "S1", "2000-01-01", "2000-12-31")
    entry = next(iter(ledger(tmp_path)["pieces"].values()))
    assert reads["n"] == 24 and entry["status"] == "validated"
    assert entry["reverted"][0]["reason"].startswith("sha256 changed")
    assert aq.file_sha256(tmp_path / entry["piece"]["path"]) == entry["piece"]["sha256"]


# --- R2: incomplete years ---------------------------------------------------------------


def test_r2_year_beyond_the_snapshot_stops_before_any_transfer(tmp_path, reads):
    short = run(tmp_path, "S_short", "2000-01-01", "2000-12-31")
    assert short.exit_code == 1 and reads["n"] == 0 and not short.products
    assert "lacks 184 of 366 days" in short.problems[0] and "S_short" in short.problems[0]
    assert not list((tmp_path / d.PARTIALS).rglob("*.nc"))
    full = run(tmp_path, "S1", "2000-01-01", "2000-12-31")
    assert full.exit_code == 0 and product_values(full)["days"] == 366


def test_r2_validation_rejects_gaps_and_partial_years():
    piece = d.plan_pieces(d.make_targets(), date(2000, 1, 1), date(2000, 12, 31), "S1",
                          "enclose", Path("."))[0]  # fmt: skip
    ds = synthetic("S1").squeeze("zlev", drop=True).sel(time="2000").drop_vars("preliminary")
    sel = d.select_box(piece.target.box, ds["lat"].values, ds["lon"].values, "enclose")
    ds = ds.isel(lat=sel.lat, lon=sel.lon_index())
    assert d.validate_piece(ds, piece, "enclose")["days"] == 366
    with pytest.raises(aq.ValidationError, match="365 days instead of 366"):
        d.validate_piece(ds.drop_isel(time=100), piece, "enclose")
    with pytest.raises(aq.ValidationError, match="fails the box policy"):
        d.validate_piece(ds.isel(lat=slice(1, None)), piece, "enclose")


# --- R3: preliminary days ---------------------------------------------------------------


def test_r3_preliminary_days_stop_before_transfer_and_name_the_month(tmp_path, reads):
    prelim = run(tmp_path, "S_prelim", "2000-01-01", "2000-12-31")
    assert prelim.exit_code == 1 and reads["n"] == 0
    assert "preliminary days in 2000-12" in prelim.problems[0]
    assert not list((tmp_path / d.PARTIALS).rglob("*.nc"))
    final = run(tmp_path, "S_final", "2000-01-01", "2000-12-31")
    assert final.exit_code == 0 and product_values(final)["sst"] == {2000: 5.0}


def test_r3_month_with_preliminary_days_is_never_saved(reads):
    pieces = d.plan_pieces(d.make_targets(), date(2000, 1, 1), date(2000, 12, 31), "S_prelim",
                           "enclose", Path("."))  # fmt: skip
    store = d.Store("S_prelim", Opener())
    ds = store.ds
    sels = {"peru": d.select_box(aq.REGIONS["peru"], ds["lat"].values, ds["lon"].values,
                                 "enclose")}  # fmt: skip
    with pytest.raises(aq.ValidationError, match="2000-12: 30 preliminary days"):
        d.fetch_year(store, pieces, sels, FAST)
    assert reads["n"] == 12


# --- R4 and R7: retries by icechunk kind ------------------------------------------------


def test_r4_object_store_errors_are_retried_and_the_store_reopened(tmp_path, reads):
    opener = Opener()
    reads["errors"] = [StorageError("object-store"), StorageError("OBJECT_STORE")]
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31", opener=opener)
    assert result.exit_code == 0 and reads["n"] == 3 and opener.calls == 3


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (StorageError("virtual-chunk-modified"), 1),
        (StorageError("unauthorized-virtual-chunk-container"), 1),
        (StorageError("virtual-chunk-fetch"), 2),  # retried once, may hide a 5xx
        (OSError("disk"), 1),  # unknown errors are permanent
    ],
)
def test_r4_permanent_kinds_are_not_retried(tmp_path, reads, error, attempts):
    reads["errors"] = [error] * 3
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    assert result.exit_code == 1 and reads["n"] == attempts
    assert str(error) in result.problems[0]
    assert set(aq.Ledger(tmp_path / d.LEDGER).statuses().values()) == {"failed"}


def test_r4_object_store_retries_are_bounded(tmp_path, reads):
    reads["errors"] = [StorageError("object-store")] * 10
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    assert result.exit_code == 1 and reads["n"] == FAST.retries + 1


def test_r4_wrapped_icechunk_errors_are_classified_by_their_cause(tmp_path, reads):
    try:
        raise RuntimeError("zarr wrapper") from StorageError("object-store")
    except RuntimeError as exc:
        wrapped = exc
    assert d.classify(wrapped) == aq.TRANSIENT
    reads["errors"] = [wrapped]
    assert run(tmp_path, "S1", "2000-01-01", "2000-01-31").exit_code == 0


def test_r7_a_transient_error_while_reopening_is_retried(tmp_path, reads):
    opener = Opener(fail=[None, StorageError("object-store")])
    reads["errors"] = [StorageError("object-store")]
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31", opener=opener)
    assert result.exit_code == 0 and opener.calls == 3 and reads["n"] == 2


def test_r4_missing_snapshot_fails_at_once(tmp_path, reads):
    opener = Opener(fail=[SnapshotNotFoundError("S9")] * 5)
    result = run(tmp_path, "S9", "2000-01-01", "2000-01-31", opener=opener)
    assert result.exit_code == 1 and opener.calls == 1 and reads["n"] == 0
    assert "SnapshotNotFoundError" in result.problems[0]


# --- R5: local write faults --------------------------------------------------------------


def test_r5_unreadable_written_piece_is_read_once_more(tmp_path, reads, monkeypatch):
    original, calls = d._verify_written, {"n": 0}

    def flaky(path, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("truncated file (simulated)")
        original(path, data)

    monkeypatch.setattr(d, "_verify_written", flaky)
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    assert result.exit_code == 0 and reads["n"] == 2
    entry = next(iter(ledger(tmp_path)["pieces"].values()))
    assert entry["status"] == "validated" and "truncated" in entry["damage"][0]["error"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_r5_written_piece_with_other_values_is_read_once_more(tmp_path, reads, monkeypatch):
    original, calls = d.write_netcdf_atomic, {"n": 0}

    def corrupting(data, path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1 and "sst" in data:
            data = data.copy()
            data["sst"] = data["sst"] + 1  # same sizes, other values
        return original(data, path, **kwargs)

    monkeypatch.setattr(d, "write_netcdf_atomic", corrupting)
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    assert result.exit_code == 0 and reads["n"] == 2
    entry = next(iter(ledger(tmp_path)["pieces"].values()))
    assert "sst differs" in entry["damage"][0]["error"]
    assert product_values(result)["sst"] == {2000: 1.0}


def test_invalid_piece_goes_to_quarantine_with_keep_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "validate_piece", _raise_validation)
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31", keep_failed=True)
    entry = next(iter(ledger(tmp_path)["pieces"].values()))
    assert result.exit_code == 1 and entry["status"] == "quarantined"
    assert (tmp_path / entry["quarantine"]).exists()


def _raise_validation(data, piece, policy):
    raise aq.ValidationError("box mismatch (simulated)")


# --- R6: product and manifest -----------------------------------------------------------


def test_r6_product_without_manifest_gets_it_on_the_next_run(tmp_path, monkeypatch):
    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(d, "write_manifest", crash)
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    product = tmp_path / "oisst_v21_peru_enclose_20000101_20000131.nc"
    assert product.exists() and not list(tmp_path.glob("oisst_peru_manifest_*.json"))
    opener = Opener()
    again = run(tmp_path, "S1", "2000-01-01", "2000-01-31", opener=opener)
    assert again.exit_code == 0 and opener.calls == 0
    assert len(list(tmp_path.glob("oisst_peru_manifest_*.json"))) == 1


def test_r6_changed_product_stops_and_names_the_file(tmp_path):
    first = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    path = first.products["peru"]["path"]
    with path.open("ab") as handle:
        handle.write(b"x")
    again = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    assert again.exit_code == 1 and path.name in again.problems[0]


# --- P1-P7: provenance ------------------------------------------------------------------


def test_manifest_has_the_registered_schema_and_compares_content(tmp_path):
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31", policy="inside")
    manifest_file = tmp_path / result.products["peru"]["manifest"]
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    reference = registered("oisst")
    assert set(reference) <= set(manifest)
    assert set(reference["query"]) <= set(manifest["query"])
    assert manifest["version"] == "icechunk snapshot S1"
    assert manifest["sizes"] == {"time": 31, "lat": 32, "lon": 20}
    assert manifest["content_sha256"] == aq.file_record(result.products["peru"]["path"])[
        "content_sha256"
    ]
    assert manifest["matches_registered"]["content_sha256"] is False
    assert manifest["matches_registered"]["same_request"] is False
    assert manifest["box"]["ok"] and manifest["box"]["policy"] == "inside"
    assert {"xarray", "zarr", "h5netcdf", "numpy", "icechunk"} <= set(manifest["software"])
    assert "virtual_refs_checksummed" in manifest
    assert manifest["query"]["lon_shift_applied"] == -360.0


def test_attributes_are_kept_and_provenance_added(tmp_path):
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    with xr.open_dataset(result.products["peru"]["path"], engine="h5netcdf") as ds:
        assert ds.attrs["title"] == NOAA_ATTRS["title"]
        assert ds.attrs["store_snapshot"] == "S1" and ds.attrs["store"] == d.STORE
        assert ds.attrs["source"].startswith(d.SOURCE) and "ICOADS" in ds.attrs["source"]
        assert ds.attrs["history"].startswith("NOAA\n") and "snapshot S1" in ds.attrs["history"]
        assert ds["sst"].attrs["units"] == "degC" and ds["sst"].attrs["valid_min"] == -300
        assert "preliminary" not in ds and "zlev" not in ds.dims
        assert ds["lon"].values.min() == -82.125 and ds["lon"].values.max() == -76.875


def test_another_snapshot_writes_another_manifest(tmp_path):
    run(tmp_path, "S1", "2000-01-01", "2000-01-31")
    run(tmp_path, "S2", "2001-01-01", "2001-01-31")
    names = {m.name for m in tmp_path.glob("oisst_peru_manifest_*.json")}
    index = json.loads((tmp_path / "manifests.json").read_text(encoding="utf-8"))
    assert len(names) == 2 and {entry["file"] for entry in index} == names


# --- E1 and A1-A3: several boxes in one pass, antimeridian ------------------------------


def wide(snapshot):
    return synthetic(snapshot, lat=LAT[(LAT > -14) & (LAT < 7)], lon=LON[(LON > 188) & (LON < 285)])


def test_e1_each_month_is_read_once_for_all_regions(tmp_path, reads):
    result = run(tmp_path, "S1", "2000-01-01", "2000-02-29", regions=("peru", "equatorial"),
                 policy="inside", opener=Opener(make=wide))  # fmt: skip
    assert result.exit_code == 0 and reads["n"] == 2
    sizes = {k: product_values(result, k)["days"] for k in result.products}
    assert sizes == {"peru": 60, "equatorial": 60}
    with xr.open_dataset(result.products["equatorial"]["path"], engine="h5netcdf") as ds:
        assert ds.sizes["lat"] == 40 and ds.sizes["lon"] == 200
        assert ds["lon"].values[0] == -169.875 and ds["lon"].values[-1] == -120.125


def coarse(snapshot):
    """A global 2-degree grid, enough for a box that crosses the antimeridian."""
    return synthetic(
        snapshot, lat=np.arange(-89.0, 90, 2, "f4"), lon=np.arange(1.0, 360, 2, "f4")
    )


def test_a1_npacific_crosses_the_antimeridian_and_keeps_0_360(tmp_path):
    result = run(tmp_path, "S1", "2000-01-01", "2000-01-31", regions=("npacific",),
                 opener=Opener(make=coarse))  # fmt: skip
    assert result.exit_code == 0
    with xr.open_dataset(result.products["npacific"]["path"], engine="h5netcdf") as ds:
        lon = ds["lon"].values
    assert lon[0] == 159.0 and lon[-1] == 231.0 and np.all(np.diff(lon) > 0)


def test_a2_box_crossing_greenwich_is_continuous(tmp_path):
    targets = d.make_targets(boxes=[(-5, 5, -10, 10)])
    result = d.download(targets, date(2000, 1, 1), date(2000, 1, 31), data_dir=tmp_path,
                        snapshot="S1", concurrency=None, retry=FAST,
                        opener=Opener(make=coarse))  # fmt: skip
    assert result.exit_code == 0
    with xr.open_dataset(result.products[targets[0].label]["path"], engine="h5netcdf") as ds:
        lon = ds["lon"].values
    assert lon[0] == -11.0 and lon[-1] == 11.0 and np.all(np.diff(lon) > 0)


# --- Command line -----------------------------------------------------------------------


def test_main_without_download_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    code = d.main(["--data-dir", str(tmp_path), "--region", "peru", "npacific"])
    out = capsys.readouterr().out
    plan = json.loads(out[: out.rindex("}") + 1])
    assert code == 0 and not list(tmp_path.iterdir())
    assert plan["snapshot"] == d.REGISTERED_SNAPSHOT and plan["box_policy"] == "enclose"
    assert plan["transfer_estimate_gb"] == pytest.approx(12784 * 0.669 / 1000, abs=0.01)
    peru, npacific = plan["boxes"]
    assert peru["selection"]["lat"] == [-12.125, -3.875, 34]
    assert npacific["selection"]["lon_convention"] == "0..360"
    # the closing line gives the estimate of the requested period, not a fixed 1991-2025 one
    d.main(["--data-dir", str(tmp_path), "--box", "-12.1", "-4.1", "-81.9", "-77.1",
            "--start", "2020-03-01", "--end", "2020-03-31"])  # fmt: skip
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.endswith(
        f"(about {round(31 * d.MB_PER_DAY / 1000, 2)} GB for 2020-03-01..2020-03-31)."
    )


def test_main_inside_plan_matches_the_registered_sizes(capsys):
    d.main(["--region", "peru", "equatorial", "npacific", "--box-policy", "inside",
            "--years", "1991-2025"])  # fmt: skip
    out = capsys.readouterr().out
    plan = json.loads(out[: out.rindex("}") + 1])
    for entry in plan["boxes"]:
        sizes = registered(entry["registered"])["sizes"]
        assert entry["selection"]["lat"][2] == sizes["lat"]
        assert entry["selection"]["lon"][2] == sizes["lon"]
        assert entry["product"] == registered(entry["registered"])["file"]


def test_main_download_with_fake_store(tmp_path):
    code = d.main(
        ["--data-dir", str(tmp_path), "--start", "2000-01-01", "--end", "2000-01-31",
         "--snapshot", "S1", "--download"],
        opener=Opener(),
    )  # fmt: skip
    assert code == 0 and (tmp_path / "oisst_v21_peru_enclose_20000101_20000131.nc").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--years", "1991-2025", "--start", "1991-01-01"],
        ["--years", "1991"],
        ["--start", "1980-01-01"],
        ["--start", "2001-01-01", "--end", "2000-01-01"],
        ["--concurrency", "0"],
    ],
)
def test_main_rejects_bad_periods(argv):
    with pytest.raises(SystemExit):
        d.main(argv)


def test_free_boxes_get_a_stable_label():
    a = d.make_targets(boxes=[(-60, 30, 150, -30)])
    b = d.make_targets(boxes=[(-60, 30, 150, -30)])
    assert a == b and a[0].label.startswith("box_") and a[0].key is None
    assert [t.label for t in d.make_targets(["peru", "peru"])] == ["peru"]
