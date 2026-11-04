# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of the shared acquisition module: box coverage, ledger, damage, cost plan."""

from __future__ import annotations

import io
import json
import sys
import threading
from datetime import date
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _acquire as aq  # noqa: E402

# Box of revision 0 (section 3.1): its edges fall between grid nodes.
UNALIGNED = aq.Box(-12.1, -3.9, -82.1, -76.9)
ALIGNED = aq.Box(-12.0, -4.0, -82.0, -77.0)
OISST_LAT = np.arange(-89.875, 90, 0.25)  # cell centres, ascending
OISST_LON = np.arange(0.125, 360, 0.25)  # 0..360
ERA5_LAT = np.arange(90, -90.001, -0.25)  # nodes, descending
ERA5_LON = np.arange(-180, 180, 0.25)  # -180..180


def picked(values, s: slice):
    return np.asarray(values)[s]


# --- box coverage ----------------------------------------------------------------------


def test_cds_area_rounds_outward_and_keeps_inside_unchanged():
    assert aq.cds_area(UNALIGNED) == [-3.75, -82.25, -12.25, -76.75]
    assert aq.cds_area(ALIGNED) == [-4.0, -82.0, -12.0, -77.0]
    # cover-cells: the nearest node whose +/-0.125 cell still reaches the edge
    assert aq.cds_area(UNALIGNED, "cover-cells") == [-4.0, -82.0, -12.0, -77.0]
    assert aq.cds_area(aq.Box(-12.2, -3.8, -82.2, -76.8), "cover-cells") == [
        -3.75,
        -82.25,
        -12.25,
        -76.75,
    ]
    # inside: the request exactly as the registered inputs were made
    assert aq.cds_area(UNALIGNED, "inside") == [-3.9, -82.1, -12.1, -76.9]
    with pytest.raises(ValueError, match="antimeridian"):
        aq.cds_area(aq.Box(-10, 10, 150, -150))
    with pytest.raises(ValueError, match="box policy"):
        aq.cds_area(UNALIGNED, "outside")


@pytest.mark.parametrize("lat", [OISST_LAT, OISST_LAT[::-1]], ids=["ascending", "descending"])
def test_select_indices_oisst_centres(lat):
    # The historical sel(slice) keeps the same centres for both boxes: the cells stop at -12/-4.
    inside = picked(lat, aq.select_indices(lat, -12.1, -3.9, "inside"))
    assert (inside.min(), inside.max()) == (-11.875, -4.125)
    enclose = picked(lat, aq.select_indices(lat, -12.1, -3.9))
    assert (enclose.min(), enclose.max()) == (-12.125, -3.875)
    assert np.all(np.diff(enclose) < 0) if lat[0] > lat[-1] else np.all(np.diff(enclose) > 0)
    # Aligned box: enclosing nodes need one more centre per side, covering cells do not.
    enclose = picked(lat, aq.select_indices(lat, -12.0, -4.0))
    assert (enclose.min(), enclose.max()) == (-12.125, -3.875)
    cells = picked(lat, aq.select_indices(lat, -12.0, -4.0, "cover-cells"))
    assert (cells.min(), cells.max()) == (-11.875, -4.125)
    cells = picked(lat, aq.select_indices(lat, -12.1, -3.9, "cover-cells"))
    assert (cells.min(), cells.max()) == (-12.125, -3.875)


def test_select_indices_on_nodes_and_float32_noise():
    lat = ERA5_LAT.astype("float32")  # node exactly on the edge counts, despite float32
    s = aq.select_indices(lat, -12.0, -4.0)
    assert (picked(lat, s).max(), picked(lat, s).min()) == (-4.0, -12.0)
    s = aq.select_indices(lat, -12.1, -3.9)
    assert (picked(lat, s).max(), picked(lat, s).min()) == (-3.75, -12.25)
    sst_cci = np.arange(-89.975, 90, 0.05).astype("float32")
    s = aq.select_indices(sst_cci, -12.11, -3.9, "cover-cells")
    assert picked(sst_cci, s).min() == pytest.approx(-12.125, abs=1e-5)


def test_select_indices_refuses_instead_of_returning_less():
    with pytest.raises(aq.CoverageError, match="no node reaches -12.3"):
        aq.select_indices(np.arange(-12.25, 0, 0.25), -12.3, -4)
    with pytest.raises(aq.CoverageError, match="no node reaches 0.1"):
        aq.select_indices(np.arange(-12.25, 0.01, 0.25), -12, 0.1)
    with pytest.raises(aq.CoverageError, match="monotonic"):
        aq.select_indices([0.0, 1.0, 0.5], 0, 1)
    with pytest.raises(aq.CoverageError, match="no coordinate inside"):
        aq.select_indices(OISST_LAT, -12.05, -12.0, "inside")
    with pytest.raises(aq.CoverageError, match="grid step"):
        aq.select_indices([5.0], 4.9, 5.1, "cover-cells")
    assert aq.select_indices([5.0], 4.9, 5.1, "cover-cells", step=0.25) == slice(0, 1)


def test_select_lon_indices_converts_conventions():
    # Peru box given in -180..180 against OISST 0..360 longitudes
    (s,) = aq.select_lon_indices(OISST_LON, UNALIGNED.west, UNALIGNED.east)
    lons = picked(OISST_LON, s)
    assert (lons.min(), lons.max()) == (277.875, 283.125)
    # and the same box in 0..360 against ERA5 -180..180 longitudes
    (s,) = aq.select_lon_indices(ERA5_LON, 277.9, 283.1)
    assert (picked(ERA5_LON, s).min(), picked(ERA5_LON, s).max()) == (-82.25, -76.75)


@pytest.mark.parametrize(
    "lons, west, east, parts, nodes",
    [
        (ERA5_LON, 150.1, -150.1, 2, [150.0, 210.0]),
        (OISST_LON, 150.1, 209.9, 1, [149.875, 210.125]),
        (OISST_LON, -10.1, 10.1, 2, [-10.125, 10.125]),
    ],
    ids=["antimeridian-in-180", "same-box-in-360", "greenwich-in-360"],
)
def test_select_lon_indices_seam_split_joins_continuously(lons, west, east, parts, nodes):
    slices = aq.select_lon_indices(lons, west, east)
    joined = np.concatenate([picked(lons, s) for s in slices])
    box = aq.Box(-1, 1, west, east)
    report = aq.check_coverage(box, np.arange(-1, 1.01, 0.25), joined, lat_step=0.25)
    assert report["ok"] and len(slices) == parts
    assert report["longitude"]["nodes"] == nodes  # in the frame of the box


def test_select_lon_indices_takes_nodes_across_the_seam():
    # Review F1: enclosing 0.01..10 on a global 0..360 grid needs 359.875 (= -0.125).
    slices = aq.select_lon_indices(OISST_LON, 0.01, 10)
    assert [picked(OISST_LON, s)[[0, -1]].tolist() for s in slices] == [
        [359.875, 359.875],
        [0.125, 10.125],
    ]
    # and the same on a -180..180 grid at the antimeridian
    slices = aq.select_lon_indices(ERA5_LON, 150.1, 179.9)
    assert [picked(ERA5_LON, s)[[0, -1]].tolist() for s in slices] == [
        [150.0, 179.75],
        [-180.0, -180.0],
    ]
    # Re-review R1: grids that repeat the seam node (0..360, -180..180 inclusive)
    for grid in (np.arange(0, 360.01, 0.25), np.arange(-180, 180.01, 0.25)):
        (s,) = aq.select_lon_indices(grid, 10.1, 19.9)
        assert picked(grid, s)[[0, -1]].tolist() == [10.0, 20.0]
        slices = aq.select_lon_indices(grid, -0.1, 0.1)
        joined = np.concatenate([picked(grid, s) for s in slices]) % 360
        assert sorted(joined.tolist()) == [0.0, 0.25, 359.75]
    # a regional grid cannot wrap: it refuses instead
    with pytest.raises(aq.CoverageError):
        aq.select_lon_indices(np.arange(0.125, 20, 0.25), 0.01, 10)


def test_split_box_and_crossing():
    box = aq.Box(-10, 10, 150, -150)
    assert box.crosses_antimeridian and box.width == pytest.approx(60)
    assert aq.split_box(box) == [aq.Box(-10, 10, 150, 180), aq.Box(-10, 10, -180, -150)]
    assert aq.split_box(aq.Box(-10, 10, 150, 210)) == aq.split_box(box)  # same box in 0..360
    assert not aq.Box(-10, 10, -10, 10).crosses_antimeridian
    assert aq.split_box(UNALIGNED) == [UNALIGNED]
    parts = [aq.cds_area(p) for p in aq.split_box(aq.Box(-10.1, 10.1, 150.1, -150.1))]
    assert parts == [[10.25, 150.0, -10.25, 180.0], [10.25, -180.0, -10.25, -150.0]]


def test_check_coverage_records_extent_and_fails_loudly():
    lat, lon = np.arange(-12.0, -3.99, 0.25), np.arange(-82.0, -76.99, 0.25)
    # What the CDS returned for the unaligned box: only the nodes inside (revision 0, 3.1).
    with pytest.raises(aq.CoverageError, match="do not enclose") as err:
        aq.check_coverage(UNALIGNED, lat, lon)
    report = err.value.report
    assert report["latitude"]["nodes"] == [-12.0, -4.0]
    assert report["latitude"]["cells"] == [-12.125, -3.875]
    assert report["requested_box"] == UNALIGNED.as_dict()
    assert len(report["problems"]) == 2
    # The same nodes cover the cells of this box and lie inside it.
    assert aq.check_coverage(UNALIGNED, lat, lon, "cover-cells")["ok"]
    assert aq.check_coverage(UNALIGNED, lat, lon, "inside")["ok"]
    assert aq.check_coverage(ALIGNED, lat, lon)["ok"]
    # 0..360 data are compared in the frame of the box
    report = aq.check_coverage(ALIGNED, lat, lon + 360)
    assert report["longitude"]["nodes"] == [-82.0, -77.0]
    # a missing column inside the extent is a problem even if the edges are fine
    with pytest.raises(aq.CoverageError, match="gap"):
        aq.check_coverage(ALIGNED, lat, np.delete(lon, 5))
    # Review F2: with the declared step, a two-value axis missing its interior is a gap.
    with pytest.raises(aq.CoverageError, match="gap"):
        aq.check_coverage(aq.Box(-1, 1, 0, 1), [-1, 1], np.arange(0, 1.01, 0.25), lat_step=0.25)


# --- request identity, ledger, atomic writes --------------------------------------------


def test_request_identity():
    base = {"variable": ["t2m", "mx2t"], "area": [-4, -82, -12, -77], "year": ["2000"]}
    same = {"year": ["2000"], "area": [-4.0, -82.0, -12.0, -77.0], "variable": ["mx2t", "t2m"]}
    assert aq.request_id(base) == aq.request_id(same)
    assert aq.request_id(base, "sl").startswith("sl_")
    other_area = {**base, "area": [-3.75, -82.25, -12.25, -76.75]}
    flipped = {**base, "area": [-82, -4, -12, -77]}  # area order matters
    assert len({aq.request_id(r) for r in (base, other_area, flipped)}) == 3


def test_ledger_reuses_only_intact_validated_pieces(tmp_path):
    ledger = aq.Ledger(tmp_path / "ledger.json")
    request = {"variable": ["t2m"], "year": ["2000"]}
    pid = aq.request_id(request, "t2m")
    target = tmp_path / "pieces" / f"{pid}.bin"
    entry = ledger.plan(pid, request, target)
    assert entry["status"] == "planned" and entry["target"] == f"pieces/{pid}.bin"
    assert ledger.plan(pid, request, target) is entry
    with pytest.raises(aq.LedgerError):
        ledger.plan(pid, {"variable": ["t2m"], "year": ["2001"]}, target)
    assert not ledger.reusable(pid)
    target.parent.mkdir()
    target.write_bytes(b"one")
    ledger.record_piece(pid, target, {"n_times": 1})
    assert ledger.reusable(pid)
    # persisted atomically: a new Ledger reads the same state, no temporary file is left
    again = aq.Ledger(tmp_path / "ledger.json")
    assert again.get(pid)["status"] == "validated"
    assert again.get(pid)["piece"]["sha256"] == aq.file_sha256(target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.json", "pieces"]
    # Review F4: a path outside the ledger's folder is stored absolute and resolves back.
    outside = tmp_path.parent / f"{tmp_path.name}_elsewhere" / "x.bin"
    ledger.plan("ext", {"piece": "ext"}, outside)
    assert ledger.resolve(ledger.get("ext")["target"]) == outside.resolve()
    # same name, other bytes: not reused, back to planned with the reason
    target.write_bytes(b"two")
    assert not again.reusable(pid)
    assert again.get(pid)["status"] == "planned"
    assert again.get(pid)["reverted"][-1]["reason"] == f"sha256 changed: pieces/{pid}.bin"
    with pytest.raises(ValueError, match="unknown status"):
        again.update(pid, "downloading")


def test_product_status_tracks_its_input_pieces(tmp_path):
    # Revision 2, H1: a yearly file built from other pieces, or from a re-downloaded
    # piece with other content, must not be reused.
    ledger = aq.Ledger(tmp_path / "ledger.json")
    product = tmp_path / "2001.nc"
    inputs = {"2001_H1": "a" * 64, "2001_H2": "b" * 64}
    assert ledger.product_status("2001", product, inputs) == "missing"
    product.write_bytes(b"year")
    assert ledger.product_status("2001", product, inputs) == "unrecorded"
    ledger.record_product("2001", product, {"2001_boundary": "c" * 64})
    assert ledger.product_status("2001", product, inputs) == "stale"
    ledger.record_product("2001", product, inputs)
    assert ledger.product_status("2001", product, inputs) == "current"
    assert ledger.product_status("2001", product, {**inputs, "2001_H2": "d" * 64}) == "stale"
    product.write_bytes(b"partial")
    assert ledger.product_status("2001", product, inputs) == "changed"


def test_atomic_target_leaves_nothing_on_failure(tmp_path):
    target = tmp_path / "out.json"
    aq.write_json_atomic(target, {"a": 1})
    with pytest.raises(RuntimeError), aq.atomic_target(target) as tmp:
        tmp.write_text("half")
        raise RuntimeError("cut")
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_sidecar(tmp_path):
    path = tmp_path / "t2m_1991-01.nc"
    path.write_bytes(b"data")
    request = {"variable": ["t2m"], "area": [-4, -82, -12, -77]}
    side = aq.write_sidecar(path, request, dataset="reanalysis-era5-single-levels")
    assert side.name == "t2m_1991-01.provenance.json"
    assert json.loads(side.read_text())["request"] == request
    assert aq.sidecar_matches(path, request)
    assert not aq.sidecar_matches(path, {**request, "area": [-3.75, -82.25, -12.25, -76.75]})
    path.write_bytes(b"other")
    assert not aq.sidecar_matches(path, request)


# --- errors, retries, damaged files -----------------------------------------------------


class HTTPError(OSError):  # shaped like requests.exceptions.HTTPError
    def __init__(self, status, text=""):
        super().__init__(text)
        self.response = type("Response", (), {"status_code": status})()


class StorageError(Exception):  # shaped like icechunk.StorageError
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


class ErrorKind:  # an enum-like kind, as icechunk.ErrorKind.ObjectStore
    def __init__(self, name):
        self.name = name


CoordinatesOutOfDatasetBounds = type("CoordinatesOutOfDatasetBounds", (Exception,), {})
SnapshotNotFoundError = type("SnapshotNotFoundError", (KeyError,), {})
ReadTimeout = type("ReadTimeout", (OSError,), {})


@pytest.mark.parametrize(
    "exc, verdict",
    [
        (HTTPError(503), "transient"),
        (HTTPError(429), "transient"),
        (HTTPError(403, "403 cost limits exceeded. Your request is too large"), "permanent"),
        (HTTPError(404), "permanent"),
        (StorageError("object-store"), "transient"),
        (StorageError(ErrorKind("ObjectStore")), "transient"),
        (StorageError("virtual-chunk-fetch"), "once"),
        (StorageError("virtual-chunk-modified"), "permanent"),
        (SnapshotNotFoundError("x"), "permanent"),
        (CoordinatesOutOfDatasetBounds("1970"), "permanent"),
        (ReadTimeout(), "transient"),
        (ConnectionResetError(), "transient"),
        (TimeoutError(), "transient"),
        (aq.ValidationError("bad"), "permanent"),
        (ValueError("unknown"), "permanent"),
    ],
)
def test_classify_error(exc, verdict):
    assert aq.classify_error(exc) == verdict


def test_classify_error_is_extensible():
    def rule(exc):
        return "transient" if isinstance(exc, ValueError) and "busy" in str(exc) else None

    assert aq.classify_error(ValueError("busy"), extra=[rule]) == "transient"
    assert aq.classify_error(ValueError("other"), extra=[rule]) == "permanent"


def test_call_with_retries():
    waits = []

    def flaky(errors):
        def fn():
            if errors:
                raise errors.pop(0)
            return "ok"

        return fn

    assert aq.call_with_retries(flaky([HTTPError(503)] * 3), sleep=waits.append) == "ok"
    assert waits == [30.0, 60.0, 120.0]
    with pytest.raises(HTTPError):
        aq.call_with_retries(flaky([HTTPError(503)] * 4), sleep=waits.append)
    waits.clear()
    with pytest.raises(StorageError):  # retried once, then given up
        aq.call_with_retries(flaky([StorageError("virtual-chunk-fetch")] * 2), sleep=waits.append)
    assert len(waits) == 1
    waits.clear()
    with pytest.raises(HTTPError):  # permanent: no retry at all
        aq.call_with_retries(flaky([HTTPError(400)]), sleep=waits.append)
    assert waits == []
    assert aq.backoff(10) == 600


def fake_source(tmp_path, answers):
    """fetch writes the next answer; finish accepts only b'good' and publishes it."""
    calls = []

    def fetch(raw):
        answer = answers.pop(0)
        calls.append(answer)
        if isinstance(answer, BaseException):
            raw.write_bytes(b"partial")
            raise answer
        raw.write_bytes(answer)

    def finish(raw):
        if raw.read_bytes() != b"good":
            raise aq.ValidationError("not a NetCDF file")
        final = tmp_path / "pieces" / "p.bin"
        with aq.atomic_target(final) as tmp:
            tmp.write_bytes(raw.read_bytes())
        return final, {"n_times": 1}

    return fetch, finish, calls


def planned(tmp_path):
    ledger = aq.Ledger(tmp_path / "ledger.json")
    ledger.plan("p", {"piece": "p"}, tmp_path / "pieces" / "p.bin")
    return ledger


def test_acquire_damaged_once_is_recorded_deleted_and_retried(tmp_path):
    ledger = planned(tmp_path)
    fetch, finish, calls = fake_source(tmp_path, [b"garbage", b"good"])
    raw = tmp_path / "raw" / "p.download"
    entry = aq.acquire(ledger, "p", fetch=fetch, finish=finish, raw=raw, sleep=lambda s: None)
    assert entry["status"] == "validated" and len(calls) == 2 and entry["attempts"] == 2
    (damage,) = entry["damage"]
    assert damage["bytes"] == 7 and damage["sha256"] == aq.hashlib.sha256(b"garbage").hexdigest()
    assert "not a NetCDF file" in damage["error"]
    assert not raw.exists()
    # validated and intact: not fetched again
    assert aq.acquire(ledger, "p", fetch=fetch, finish=finish, raw=raw)["status"] == "validated"
    assert len(calls) == 2


def test_acquire_damaged_twice_fails_and_keeps_evidence(tmp_path):
    ledger = planned(tmp_path)
    fetch, finish, _ = fake_source(tmp_path, [b"garbage", b"bad"])
    raw = tmp_path / "raw" / "p.download"
    entry = aq.acquire(ledger, "p", fetch=fetch, finish=finish, raw=raw)
    assert entry["status"] == "failed" and len(entry["damage"]) == 2
    assert [d["bytes"] for d in entry["damage"]] == [7, 3]
    assert not raw.exists() and not (tmp_path / "pieces" / "p.bin").exists()


def test_acquire_keep_failed_quarantines(tmp_path):
    ledger = planned(tmp_path)
    fetch, finish, calls = fake_source(tmp_path, [b"garbage", b"good"])
    raw = tmp_path / "raw" / "p.download"
    entry = aq.acquire(ledger, "p", fetch=fetch, finish=finish, raw=raw, keep_failed=True)
    assert entry["status"] == "quarantined" and len(calls) == 1
    kept = ledger.resolve(entry["damage"][0]["quarantine"])
    assert kept.parent.name == "quarantine" and kept.read_bytes() == b"garbage"


def test_damage_evidence_is_saved_before_the_file_goes(tmp_path, monkeypatch):
    # Review F3: if the evidence cannot be saved, the damaged file must stay.
    ledger = planned(tmp_path)
    ledger.autosave = False
    raw = tmp_path / "p.raw"
    raw.write_bytes(b"garbage")

    def full_disk():
        raise OSError("no space left")

    monkeypatch.setattr(ledger, "save", full_disk)
    with pytest.raises(OSError, match="no space"):
        aq.discard_damaged(ledger, "p", raw, aq.ValidationError("bad"))
    assert raw.read_bytes() == b"garbage"
    monkeypatch.undo()
    aq.discard_damaged(ledger, "p", raw, aq.ValidationError("bad"))
    assert not raw.exists()
    assert aq.Ledger(tmp_path / "ledger.json").get("p")["damage"][-1]["bytes"] == 7


def test_acquire_no_retry_without_retry_damaged(tmp_path):
    ledger = planned(tmp_path)
    fetch, finish, calls = fake_source(tmp_path, [b"garbage", b"good"])
    entry = aq.acquire(
        ledger, "p", fetch=fetch, finish=finish, raw=tmp_path / "p.raw", retry_damaged=False
    )
    assert entry["status"] == "failed" and len(calls) == 1


def test_acquire_transient_then_permanent_fetch_errors(tmp_path):
    ledger = planned(tmp_path)
    fetch, finish, calls = fake_source(tmp_path, [HTTPError(503), b"good"])
    raw = tmp_path / "p.raw"
    entry = aq.acquire(ledger, "p", fetch=fetch, finish=finish, raw=raw, sleep=lambda s: None)
    assert entry["status"] == "validated" and len(calls) == 2
    ledger = aq.Ledger(tmp_path / "other.json")
    ledger.plan("p", {"piece": "p"}, tmp_path / "pieces" / "p.bin")
    fetch, finish, calls = fake_source(tmp_path, [HTTPError(403, "cost limits exceeded")])
    entry = aq.acquire(ledger, "p", fetch=fetch, finish=finish, raw=raw)
    assert entry["status"] == "failed" and "cost limits" in entry["error"]
    assert len(calls) == 1 and not raw.exists()


def test_run_parallel_reports_every_failure(tmp_path):
    # Revision 2, H2: pool.map surfaced only the first error and cancelled the rest.
    ledger = aq.Ledger(tmp_path / "ledger.json")
    ids = [f"p{i}" for i in range(6)]
    started = []
    lock = threading.Lock()

    def task(i):
        def run():
            with lock:
                started.append(i)
            ledger.plan(i, {"piece": i}, tmp_path / f"{i}.bin")
            if i in ("p0", "p4"):
                ledger.update(i, "failed", error=f"fake failure {i}")
                raise RuntimeError(f"fake failure {i}")
            (tmp_path / f"{i}.bin").write_bytes(i.encode())
            return ledger.record_piece(i, tmp_path / f"{i}.bin")

        return run

    outcomes = aq.run_parallel({i: task(i) for i in ids}, workers=3)
    assert list(outcomes) == ids and sorted(started) == ids
    assert [i for i, o in outcomes.items() if not o.ok] == ["p0", "p4"]
    summary = aq.RunSummary.from_ledger(ledger, ids, outcomes)
    assert summary.counts() == {"failed": 2, "validated": 4} and summary.exit_code == 1
    out = io.StringIO()
    summary.report(out)
    assert "fake failure p0" in out.getvalue() and "fake failure p4" in out.getvalue()
    assert aq.RunSummary.from_ledger(ledger, ["p1", "p2"]).exit_code == 0
    assert aq.RunSummary.from_ledger(ledger, ["p1", "never"]).statuses["never"] == "unplanned"


# --- manifest ---------------------------------------------------------------------------


def test_manifest_named_after_request_and_indexed(tmp_path):
    pytest.importorskip("h5netcdf")
    import xarray as xr

    path = tmp_path / "piece.nc"
    xr.Dataset({"t2m": (("y",), np.arange(3.0))}).to_netcdf(path, engine="h5netcdf")
    request = {"variable": ["t2m"], "area": [-3.75, -82.25, -12.25, -76.75]}
    lat, lon = np.arange(-12.25, -3.74, 0.25), np.arange(-82.25, -76.74, 0.25)
    coverage = aq.check_coverage(UNALIGNED, lat, lon)
    manifests = []
    for _ in range(2):
        manifest = aq.build_manifest(
            source="era5",
            request=request,
            files={"piece.nc": path},
            coverage=coverage,
            retrieval_window=["2026-11-03T00:00:00+00:00", "2026-11-03T01:00:00+00:00"],
            pieces=["sl_x"],
        )
        manifests.append(aq.write_manifest(tmp_path, "era5", manifest))
    first, second = manifests
    assert first != second and first.exists() and second.exists()
    assert first.name.startswith(f"era5_manifest_{aq.request_id(request)}_")
    stored = json.loads(first.read_text())
    assert stored["files"]["piece.nc"]["content_sha256"] == aq.content_sha256(path)
    assert stored["files"]["piece.nc"]["sha256"] == aq.file_sha256(path)
    assert stored["box"]["policy"] == "enclose" and stored["box"]["ok"]
    assert stored["request"] == request
    assert {"python", "xarray", "cdsapi"} <= set(stored["software"])
    index = json.loads((tmp_path / "manifests.json").read_text())
    assert [e["file"] for e in index] == [first.name, second.name]


def test_retrieval_window(tmp_path):
    ledger = aq.Ledger(tmp_path / "ledger.json")
    assert ledger.retrieval_window() is None
    for i, stamp in (("a", "2026-11-03T02:00:00+00:00"), ("b", "2026-11-03T01:00:00+00:00")):
        ledger.plan(i, {"piece": i}, tmp_path / i)
        ledger.update(i, downloaded_utc=stamp)
    assert ledger.retrieval_window() == ["2026-11-03T01:00:00+00:00", "2026-11-03T02:00:00+00:00"]


# --- CDS cost planner -------------------------------------------------------------------


def test_cds_cost_reproduces_measured_costs():
    # estimate_costs on 2026-11-03 (revisions 1 and 2)
    assert aq.cds_cost(3, 24, 181) == 78_192  # current plan: 3 variables, January-June
    assert aq.cds_cost(10, 24, 31) == 44_640
    assert aq.cds_cost(1, 24, 31, 1) == 4_464
    assert aq.cds_cost(1, 24, 31, 5) == 22_320
    assert aq.cds_cost(1, 24, 31, 37) == 165_168
    assert aq.cds_cost(1, 24, 31, 3) == 13_392


def test_pressure_default_is_one_monthly_request():
    chunks = aq.plan_cds_chunks(
        date(2023, 3, 1),
        date(2023, 3, 31),
        dataset="reanalysis-era5-pressure-levels",
        n_variables=1,
        levels=["850", "700", "500"],
    )
    assert chunks == [
        aq.CdsChunk(date(2023, 3, 1), date(2023, 3, 31), ("850", "700", "500"), 13_392)
    ]


@pytest.mark.parametrize("n_variables", [1, 2, 3])
def test_pressure_plans_never_exceed_the_margin(n_variables):
    levels = [str(p) for p in range(1000, 99, -25)][:37]
    budget = 60_000 * 0.9
    for n in range(1, 38):
        chunks = aq.plan_cds_chunks(
            date(2024, 1, 1),
            date(2024, 12, 31),
            dataset="reanalysis-era5-pressure-levels",
            n_variables=n_variables,
            levels=levels[:n],
        )
        assert all(c.cost <= budget for c in chunks)
        assert all(c.cost == aq.cds_cost(n_variables, 24, c.days, len(c.levels)) for c in chunks)
        # every (day, level) is requested exactly once
        pairs = sorted(
            (c.start.toordinal() + d, lev)
            for c in chunks
            for d in range(c.days)
            for lev in c.levels
        )
        assert len(pairs) == len(set(pairs)) == 366 * n
    # 37 levels of one variable: groups of at most 12 (13 would pass the limit, not the margin)
    chunks = aq.plan_cds_chunks(
        date(2023, 1, 1), date(2023, 1, 31), dataset="reanalysis-era5-pressure-levels",
        n_variables=1, levels=levels,
    )  # fmt: skip
    assert [len(c.levels) for c in chunks] == [10, 9, 9, 9]


def test_single_levels_plans():
    kw = dict(dataset="reanalysis-era5-single-levels")
    # historical half-year plan of three variables fits: two requests per year
    chunks = aq.plan_cds_chunks(
        date(2000, 1, 1), date(2000, 12, 31), n_variables=3, period="half-year", **kw
    )
    assert [(c.start, c.end) for c in chunks] == [
        (date(2000, 1, 1), date(2000, 6, 30)),
        (date(2000, 7, 1), date(2000, 12, 31)),
    ]
    assert chunks[0].date_fields()["month"] == ["01", "02", "03", "04", "05", "06"]
    # ten variables do not fit in half a year: back to months
    chunks = aq.plan_cds_chunks(
        date(2001, 1, 1), date(2001, 12, 31), n_variables=10, period="half-year", **kw
    )
    assert len(chunks) == 12 and all(c.cost <= 108_900 for c in chunks)
    # 15 variables of the first new case (revision 2, decision 3) in one monthly request
    (chunk,) = aq.plan_cds_chunks(date(2001, 1, 1), date(2001, 1, 31), n_variables=15, **kw)
    assert chunk.cost == 66_960
    # 40 variables: a month is too much, so blocks of days
    chunks = aq.plan_cds_chunks(date(2001, 1, 1), date(2001, 1, 31), n_variables=40, **kw)
    assert [c.days for c in chunks] == [18, 13]
    assert chunks[1].date_fields() == {
        "year": ["2001"], "month": ["01"], "day": [f"{d:02d}" for d in range(19, 32)],
    }  # fmt: skip


def test_plan_partial_months_and_limits():
    kw = dict(dataset="reanalysis-era5-single-levels", period="half-year")
    chunks = aq.plan_cds_chunks(date(2000, 3, 15), date(2001, 2, 10), n_variables=3, **kw)
    assert [(c.start, c.end) for c in chunks] == [
        (date(2000, 3, 15), date(2000, 3, 31)),
        (date(2000, 4, 1), date(2000, 6, 30)),
        (date(2000, 7, 1), date(2000, 12, 31)),
        (date(2001, 1, 1), date(2001, 1, 31)),
        (date(2001, 2, 1), date(2001, 2, 10)),
    ]
    for c in chunks:
        c.date_fields()  # every chunk is expressible as year/month/day lists
    with pytest.raises(aq.CostLimitError):
        aq.plan_cds_chunks(
            date(2023, 1, 1), date(2023, 1, 31), dataset="reanalysis-era5-pressure-levels",
            n_variables=1, levels=["1000", "925", "850"], level_group=13,
        )  # fmt: skip
    with pytest.raises(aq.CostLimitError):
        aq.plan_cds_chunks(date(2023, 1, 1), date(2023, 1, 1), n_variables=800, **kw)
    with pytest.raises(ValueError, match="positive"):  # review F5: no silent empty plan
        aq.plan_cds_chunks(
            date(2000, 1, 1), date(2000, 1, 31), dataset="reanalysis-era5-pressure-levels",
            n_variables=1, levels=["850"], level_group=-1,
        )  # fmt: skip
    with pytest.raises(ValueError, match="not one partial month"):
        aq.cds_date_fields(date(2000, 1, 15), date(2000, 2, 15))


# --- command line and registered inputs (F2.0) -------------------------------------------


def test_regions_and_box_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    aq.add_box_arguments(parser)
    aq.add_period_arguments(parser, "1991-01-01", "2025-12-31")
    args = parser.parse_args([])
    assert aq.box_from_args(args) == (ALIGNED, "peru") and args.box_policy == "enclose"
    box, name = aq.box_from_args(parser.parse_args(["--box", "-12.1", "-3.9", "-82.1", "-76.9"]))
    assert (box, name) == (UNALIGNED, None)
    with pytest.raises(SystemExit):
        parser.parse_args(["--region", "peru", "--box", "0", "1", "0", "1"])
    # the registered OISST queries, as boxes
    npacific = aq.REGIONS["npacific"]
    assert npacific.crosses_antimeridian and npacific.width == pytest.approx(70)
    (s,) = aq.select_lon_indices(OISST_LON, npacific.west, npacific.east, "inside")
    assert picked(OISST_LON, s)[[0, -1]].tolist() == [160.125, 229.875]
    args = parser.parse_args(["--start", "2000-02-01", "--end", "2000-01-01"])
    with pytest.raises(ValueError, match="after"):
        aq.period_from_args(args)
    args = parser.parse_args(["--start", "1981-01-01"])
    with pytest.raises(ValueError, match="first available"):
        aq.period_from_args(args, first=date(1981, 9, 1))


def test_matches_registered_decides_by_content():
    record = {"sha256": "a", "content_sha256": "c"}
    assert aq.matches_registered(record, {"sha256": "b", "content_sha256": "c"}) == {
        "content_sha256": True,
        "sha256": False,
    }
    assert aq.matches_registered(record, {"sha256": "a"}) == {
        "content_sha256": None,
        "sha256": True,
    }


def test_piece_with_several_files_needs_all_intact(tmp_path):
    ledger = aq.Ledger(tmp_path / "ledger.json")
    ledger.plan("req", {"piece": "req"}, tmp_path / "t2m.bin")
    files = [tmp_path / "t2m.bin", tmp_path / "mx2t.bin"]
    for f in files:
        f.write_bytes(f.name.encode())
    entry = ledger.record_piece("req", files)
    assert set(entry["piece"]["files"]) == {"t2m.bin", "mx2t.bin"}
    assert ledger.reusable("req")
    files[1].write_bytes(b"changed")
    assert not ledger.reusable("req")
    assert ledger.get("req")["reverted"][-1]["reason"] == "sha256 changed: mx2t.bin"
