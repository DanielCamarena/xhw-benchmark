# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of download_glorys.py with a fake copernicusmarine (revision 4, appendix A).

The fake ``subset`` writes a small NetCDF on the 1/12 degree grid, honouring the box, the
depth range, ``coordinates_selection_method`` and a coverage end (``last``); scripted
failures cut a download (truncated file + ConnectionError), return an unreadable file,
shift the grid or raise a given exception.
"""

from __future__ import annotations

import json
import math
import sys
import types
from datetime import date
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")

import _acquire as aq  # noqa: E402
import download_glorys as dg  # noqa: E402

BOX = ["--box", "-5", "-4.5", "-80", "-79.5"]  # aligned with the 1/12 degree grid
UNALIGNED = ["--box", "-5.01", "-4.49", "-80.01", "-79.49"]
LEVELS = [0.494, 1.541, 2.646, 3.819]


class DatasetVersionNotFound(Exception):
    """Same class name as copernicusmarine 2.5.0 (revision 5, 4.3)."""


def _nodes(low: float, high: float, method: str, bounds=(-2160, 2160)) -> np.ndarray:
    """Nodes k/12 selected like the client; ``bounds`` are the first and last k of the grid."""
    eps = 1e-9
    if method == "outside":
        k0, k1 = math.floor(low * 12 + eps), math.ceil(high * 12 - eps)
    else:
        k0, k1 = math.ceil(low * 12 - eps), math.floor(high * 12 + eps)
    k0, k1 = max(k0, bounds[0]), min(k1, bounds[1])
    return (np.arange(k0, k1 + 1) / 12).astype("f4")


class FakeClient(types.ModuleType):
    def __init__(self):
        super().__init__("copernicusmarine")
        self.calls: list[dict] = []
        self.last = pd.Timestamp("2026-09-22")
        self.fail: dict = {}  # year or (year, "west"/"east") -> list of modes
        self.before_return = None
        self.lon_last_k = 2159  # longitudes -180..179.917; 2160 also stores 180

    def subset(self, **cfg):
        self.calls.append(cfg)
        start = pd.Timestamp(cfg["start_datetime"][:10])
        end = min(pd.Timestamp(cfg["end_datetime"][:10]), self.last)
        method = cfg["coordinates_selection_method"]
        lat = _nodes(cfg["minimum_latitude"], cfg["maximum_latitude"], method)
        lon = _nodes(
            cfg["minimum_longitude"],
            cfg["maximum_longitude"],
            method,
            bounds=(-2160, self.lon_last_k),
        )
        depth = [d for d in LEVELS if d <= cfg["maximum_depth"]]
        if method == "outside" and depth[-1] < cfg["maximum_depth"]:
            depth.append(LEVELS[len(depth)])
        extent = [
            types.SimpleNamespace(coordinate_id="time", minimum=str(start), maximum=str(end)),
            types.SimpleNamespace(
                coordinate_id="latitude", minimum=float(lat[0]), maximum=float(lat[-1])
            ),
        ]
        response = types.SimpleNamespace(
            status="000",
            message="ok",
            file_size=0.1,
            data_transfer_size=1.2,
            service=cfg.get("service"),
            coordinates_extent=extent,
        )
        if cfg.get("dry_run"):
            return response
        out = Path(cfg["output_directory"]) / cfg["output_filename"]
        assert not out.exists(), "the script must clear stale raw files"
        side = "east" if cfg["minimum_longitude"] == -180.0 else "west"
        modes = self.fail.get((start.year, side)) or self.fail.get(start.year) or []
        mode = modes.pop(0) if modes else None
        if mode == "cut":
            out.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 1000)
            raise ConnectionError(f"simulated network cut during {start.year}")
        if mode == "garbage":
            out.write_bytes(b"not a netcdf file")
            return response
        if isinstance(mode, BaseException):
            raise mode
        if mode == "shift":
            lat = lat + np.float32(1 / 12)
        if mode == "extra":
            lat = np.append(lat, lat[-1] + np.float32(1 / 12))
        if mode == "no-depth":
            depth = []
        if mode == "contradict":
            extent[0].maximum = str(end - pd.Timedelta(days=100))
        t = pd.date_range(start, end, freq="D")
        if mode == "short-file":  # the response announces the whole period
            t = t[:-100]
        if mode == "no-seam-node":
            lon = lon[1:]
        shape = (len(t), len(depth), len(lat), len(lon))
        dims = ("time", "depth", "latitude", "longitude")
        # Values follow the longitude, so the order of a join can be checked.
        values = np.broadcast_to(15 + 0.01 * lon.astype("f8"), shape).astype("f4")
        data = {name: (dims, values) for name in cfg["variables"]}
        if mode == "scalar":
            data = {name: ((), np.float32(15.0)) for name in cfg["variables"]}
        packed = {
            "dtype": "int16",
            "scale_factor": 0.001,
            "add_offset": 15.0,
            "_FillValue": -32767,
        }  # as the service sends them: packed, uncompressed (F3 files, 2026-11-04)
        xr.Dataset(
            data, coords={"time": t, "depth": depth, "latitude": lat, "longitude": lon}
        ).to_netcdf(out, engine="h5netcdf", encoding={name: packed for name in cfg["variables"]})
        if self.before_return is not None:
            self.before_return(start.year)
        return response

    def years(self) -> list[int]:
        return [int(c["start_datetime"][:4]) for c in self.calls if not c.get("dry_run")]


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setitem(sys.modules, "copernicusmarine", fake)
    monkeypatch.setattr(dg, "_sleep", lambda seconds: None)
    return fake


def run(tmp_path: Path, *args: str, box=BOX) -> int:
    """A download (or a dry run) of the small test box, 0-2 m."""
    mode = [] if "--dry-run" in args else ["--download"]
    return dg.main(["--data-dir", str(tmp_path), *box, "--max-depth", "2", *mode, *args])


def folder(tmp_path: Path) -> Path:
    (path,) = tmp_path.glob("glorys12v1_*")
    return path


def finals(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in folder(tmp_path).glob("*.nc"))


def manifests(tmp_path: Path) -> list[dict]:
    index = json.loads((folder(tmp_path) / "manifests.json").read_text(encoding="utf-8"))
    return [
        json.loads((folder(tmp_path) / item["file"]).read_text(encoding="utf-8")) for item in index
    ]


def test_module_does_not_import_the_client():
    source = (EXAMPLES / "download_glorys.py").read_text(encoding="utf-8")
    assert "\nimport copernicusmarine" not in source


def test_plan_defaults_reproduce_the_registered_request(tmp_path, capsys, monkeypatch):
    monkeypatch.delitem(sys.modules, "copernicusmarine", raising=False)
    assert dg.main(["--data-dir", str(tmp_path)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [x["file"] for x in lines] == [f"thetao_{y}.nc" for y in range(1993, 2024)]
    first = lines[0]
    assert first["dataset_id"] == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    assert first["dataset_version"] == "202311"
    assert first["variables"] == ["thetao"]
    assert (first["minimum_latitude"], first["maximum_latitude"]) == (-12.0, -4.0)
    assert (first["minimum_longitude"], first["maximum_longitude"]) == (-82.0, -77.0)
    assert (first["minimum_depth"], first["maximum_depth"]) == (0.0, 500.0)
    assert first["start_datetime"] == "1993-01-01T00:00:00"
    assert lines[-1]["end_datetime"] == "2023-12-31T00:00:00"
    assert first["coordinates_selection_method"] == "outside"
    assert first["service"] == "arco-time-series"
    assert not tmp_path.joinpath("glorys12v1_peru").exists()
    assert "copernicusmarine" not in sys.modules


def test_policy_mapping_and_cover_cells_box():
    box = dg.aq.Box(-5.01, -4.49, -80.01, -79.49)
    kw = dict(variables=["thetao"], max_depth=500.0, dataset_version="202311", service="s")
    period = (date(1993, 1, 1), date(1993, 12, 31))
    assert dg.request(*period, box, "inside", **kw)["coordinates_selection_method"] == "inside"
    enclose = dg.request(*period, box, "enclose", **kw)
    assert enclose["coordinates_selection_method"] == "outside"
    assert enclose["minimum_latitude"] == -5.01
    cells = dg.request(*period, box, "cover-cells", **kw)
    assert cells["minimum_latitude"] == pytest.approx(-5.01 + 1 / 24)
    assert cells["maximum_longitude"] == pytest.approx(-79.49 - 1 / 24)


def test_cut_then_rerun_resumes_at_the_cut_year(tmp_path, client):
    # R1: a cut leaves no truncated file under a final name, and a rerun continues there.
    client.fail[1995] = ["cut"]
    assert run(tmp_path, "--years", "1993-1997", "--retries", "0") == 1
    kept = [f"thetao_{y}.nc" for y in (1993, 1994, 1996, 1997)]
    assert finals(tmp_path) == kept  # a failed year does not stop the others
    assert not list(folder(tmp_path).glob(".tmp/*"))
    (first,) = manifests(tmp_path)
    assert sorted(first["files"]) == kept
    assert sorted(first["statuses"].values()) == ["failed"] + ["validated"] * 4
    client.calls.clear()
    assert run(tmp_path, "--years", "1993-1997", "--retries", "0") == 0
    assert client.years() == [1995]
    assert finals(tmp_path) == [f"thetao_{y}.nc" for y in range(1993, 1998)]
    assert len(manifests(tmp_path)) == 2


def test_cut_is_retried_as_transient(tmp_path, client):
    client.fail[1993] = ["cut"]
    assert run(tmp_path, "--years", "1993") == 0
    assert client.years() == [1993, 1993]


def test_extending_the_period_reuses_valid_years(tmp_path, client):
    # R2: no FileExistsError; only the new years are requested; manifests are not overwritten.
    assert run(tmp_path, "--years", "1993-1994") == 0
    client.calls.clear()
    assert run(tmp_path, "--years", "1993-1996") == 0
    assert client.years() == [1995, 1996]
    first, second = manifests(tmp_path)
    assert sorted(first["files"]) == ["thetao_1993.nc", "thetao_1994.nc"]
    assert sorted(second["files"]) == [f"thetao_{y}.nc" for y in range(1993, 1997)]
    assert first["request_sha256"] != second["request_sha256"]


def test_changed_file_is_fetched_again(tmp_path, client):
    assert run(tmp_path, "--years", "1993") == 0
    (folder(tmp_path) / "thetao_1993.nc").write_bytes(b"edited")
    client.calls.clear()
    assert run(tmp_path, "--years", "1993") == 0
    assert client.years() == [1993]
    xr.open_dataset(folder(tmp_path) / "thetao_1993.nc").close()


def test_partial_year_is_rejected_by_default(tmp_path, client, capsys):
    # R3 / decision 5: fewer days than the year are not saved under the year's name.
    assert run(tmp_path, "--years", "2026", "--retries", "0") == 1
    assert finals(tmp_path) == []
    out = capsys.readouterr().out
    assert "2026-01-01..2026-09-22" in out and "--allow-partial" in out
    assert client.years() == [2026]  # permanent: not retried
    (manifest,) = manifests(tmp_path)  # a run that validates nothing still has its manifest
    assert manifest["files"] == {} and list(manifest["statuses"].values()) == ["failed"]


@pytest.mark.parametrize(
    "mode, message",
    [
        ("scalar", "has dimensions"),
        ("no-depth", "depth must be"),
        ("contradict", "subset response announces"),
        ("short-file", "subset response announces"),  # damaged, not a partial year
    ],
)
def test_file_not_matching_its_request_is_rejected(tmp_path, client, mode, message):
    client.fail[1993] = [mode, mode]
    assert run(tmp_path, "--years", "1993") == 1
    assert client.years() == [1993, 1993]  # damaged: fetched once more, then failed
    assert finals(tmp_path) == []
    assert not (folder(tmp_path) / "grid_reference.json").exists()
    ledger = json.loads((folder(tmp_path) / "ledger.json").read_text(encoding="utf-8"))
    (entry,) = ledger["pieces"].values()
    assert entry["status"] == "failed" and message in entry["damage"][-1]["error"]


def test_publication_cut_before_the_ledger_record_is_recovered(tmp_path, client, monkeypatch):
    # The file reached its final name but the run stopped before Ledger.record_piece.
    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(dg.aq.Ledger, "record_piece", crash)
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, "--years", "1993")
    assert finals(tmp_path) == ["thetao_1993.nc"]
    client.calls.clear()
    assert run(tmp_path, "--years", "1993") == 0
    assert client.calls == []  # validated again from disk, not downloaded
    ledger = json.loads((folder(tmp_path) / "ledger.json").read_text(encoding="utf-8"))
    (entry,) = ledger["pieces"].values()
    assert entry["status"] == "validated"
    assert entry["checks"]["response"] == {"recovered": "interrupted run"}


def test_manifest_is_updated_after_each_piece(tmp_path, client):
    seen = {}

    def look(year):
        if year == 1994:
            (manifest,) = manifests(tmp_path)
            seen.update(manifest["files"])

    client.before_return = look
    assert run(tmp_path, "--years", "1993-1994") == 0
    assert sorted(seen) == ["thetao_1993.nc"]
    (manifest,) = manifests(tmp_path)
    assert sorted(manifest["files"]) == ["thetao_1993.nc", "thetao_1994.nc"]


def test_partial_year_on_demand_and_refreshed_later(tmp_path, client):
    assert run(tmp_path, "--years", "2026", "--allow-partial") == 0
    assert finals(tmp_path) == ["thetao_2026_partial.nc"]
    (manifest,) = manifests(tmp_path)
    assert manifest["partial_files"] == ["thetao_2026_partial.nc"]
    checks = manifest["file_checks"]["thetao_2026_partial.nc"]
    assert (checks["days"], checks["expected_days"], checks["last_day"]) == (
        265,
        365,
        "2026-09-22",
    )
    assert checks["response"]["coordinates_extent"]["time"]["maximum"].startswith("2026-09-22")
    assert checks["response_days_agree"] is True
    # Once the service covers the whole year, the next run fetches it again.
    client.last = pd.Timestamp("2027-06-30")
    client.calls.clear()
    assert run(tmp_path, "--years", "2026") == 0
    assert client.years() == [2026]
    assert finals(tmp_path) == ["thetao_2026.nc"]


def test_unreadable_file_is_recorded_and_fetched_once_more(tmp_path, client):
    client.fail[1993] = ["garbage"]
    assert run(tmp_path, "--years", "1993") == 0
    assert client.years() == [1993, 1993]
    ledger = json.loads((folder(tmp_path) / "ledger.json").read_text(encoding="utf-8"))
    (entry,) = ledger["pieces"].values()
    assert entry["status"] == "validated"
    assert len(entry["damage"]) == 1 and entry["damage"][0]["bytes"] == len(b"not a netcdf file")


def test_keep_failed_quarantines_the_damaged_file(tmp_path, client):
    client.fail[1993] = ["garbage"]
    assert run(tmp_path, "--years", "1993", "--keep-failed") == 1
    assert len(list((folder(tmp_path) / "quarantine").glob("*.nc"))) == 1
    assert finals(tmp_path) == []


def test_grid_differing_from_the_reference_is_rejected(tmp_path, client):
    client.fail[1994] = ["extra", "extra"]
    assert run(tmp_path, "--years", "1993-1994") == 1
    assert finals(tmp_path) == ["thetao_1993.nc"]
    ledger = json.loads((folder(tmp_path) / "ledger.json").read_text(encoding="utf-8"))
    errors = [e["error"] for e in ledger["pieces"].values() if e["status"] == "failed"]
    assert len(errors) == 1 and "latitude" in errors[0]
    grids = json.loads((folder(tmp_path) / "grid_reference.json").read_text(encoding="utf-8"))
    (reference,) = grids.values()
    assert len(reference["latitude"]) == 7 and len(reference["depth"]) == 3


def test_grid_not_enclosing_the_box_fails_without_retry(tmp_path, client):
    client.fail[1993] = ["shift"]
    assert run(tmp_path, "--years", "1993") == 1
    assert client.years() == [1993]
    ledger = json.loads((folder(tmp_path) / "ledger.json").read_text(encoding="utf-8"))
    (entry,) = ledger["pieces"].values()
    assert "do not enclose" in entry["error"]


def test_permanent_client_error_is_not_retried(tmp_path, client):
    client.fail[1993] = [DatasetVersionNotFound("no version 199901")]
    assert run(tmp_path, "--years", "1993") == 1
    assert client.years() == [1993]


def test_unaligned_box_is_enclosed_and_recorded(tmp_path, client):
    assert run(tmp_path, "--years", "1993", box=UNALIGNED) == 0
    (manifest,) = manifests(tmp_path)
    box = manifest["box"]
    assert box["ok"] and box["policy"] == "enclose"
    assert box["latitude"]["nodes"][0] <= -5.01 and box["latitude"]["nodes"][1] >= -4.49
    assert box["longitude"]["nodes"][0] <= -80.01 and box["longitude"]["nodes"][1] >= -79.49
    assert manifest["request"]["requested_box"] == {
        "south": -5.01,
        "north": -4.49,
        "west": -80.01,
        "east": -79.49,
    }


@pytest.mark.parametrize("policy", ["inside", "cover-cells"])
def test_other_policies_pass_their_coverage_check(tmp_path, client, policy):
    assert run(tmp_path, "--years", "1993", "--box-policy", policy, box=UNALIGNED) == 0
    (manifest,) = manifests(tmp_path)
    assert manifest["box"]["policy"] == policy and manifest["box"]["ok"]


def test_manifest_records_version_service_extent_and_hashes(tmp_path, client):
    assert run(tmp_path, "--start", "1993-01-01", "--end", "1993-01-31") == 0
    assert finals(tmp_path) == ["thetao_19930101-19930131.nc"]
    (manifest,) = manifests(tmp_path)
    assert manifest["source"] == "glorys_box"
    assert manifest["dataset_version"] == "202311"
    assert manifest["service"] == "arco-time-series"
    record = manifest["files"]["thetao_19930101-19930131.nc"]
    assert len(record["sha256"]) == 64 and len(record["content_sha256"]) == 64
    assert "time" in manifest["coordinates_extent"]["thetao_19930101-19930131.nc"]
    assert manifest["file_checks"]["thetao_19930101-19930131.nc"]["days"] == 31
    assert manifest["retrieval_window_utc"] is not None
    name = next(folder(tmp_path).glob("glorys_box_manifest_*_*.json")).name
    assert name.startswith("glorys_box_manifest_")


def test_peru_manifest_compares_with_the_registered_record(tmp_path, client):
    args = ["--data-dir", str(tmp_path), "--max-depth", "1", "--years", "1993", "--download"]
    assert dg.main(args) == 0
    (manifest,) = manifests(tmp_path)
    assert manifest["source"] == "glorys"
    comparison = manifest["registered_comparison"]["thetao_1993.nc"]
    assert comparison == {"content_sha256": None, "sha256": False}


def test_existing_file_of_unknown_origin_is_not_replaced(tmp_path, client):
    target = tmp_path / "glorys12v1_peru" / "thetao_1993.nc"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"registered original")
    assert dg.main(["--data-dir", str(tmp_path), "--years", "1993", "--download"]) == 1
    assert target.read_bytes() == b"registered original"
    assert client.calls == []


def test_dry_run_writes_nothing(tmp_path, client, capsys):
    assert run(tmp_path, "--years", "2025-2026", "--dry-run") == 0
    assert [c["dry_run"] for c in client.calls] == [True, True]
    assert not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "overstated" in out and "partial: the service would return 2026-01-01..2026-09-22" in out


@pytest.mark.parametrize(
    "args",
    [
        ["--years", "1993", "--start", "1993-01-01"],
        ["--years", "1992"],
        ["--years", "x"],
        ["--start", "1994-01-01", "--end", "1993-01-01"],
        ["--max-depth", "0"],
    ],
)
def test_invalid_arguments_are_rejected(tmp_path, args):
    with pytest.raises(SystemExit) as err:
        dg.main(["--data-dir", str(tmp_path), *args])
    assert err.value.code == 2


def test_download_without_client_explains_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "copernicusmarine", None)
    with pytest.raises(SystemExit, match="Install copernicusmarine"):
        run(tmp_path, "--years", "1993", "--download")


def test_download_flag_is_needed_to_write(tmp_path, client):
    assert dg.main(["--data-dir", str(tmp_path), *BOX, "--years", "1993"]) == 0
    assert client.calls == [] and not list(tmp_path.iterdir())


# --- Boxes crossing the antimeridian (revision 0, B5) --------------------------------------

CROSS = ["--box", "25", "25.5", "150", "-150"]  # 150E-150W
JANUARY = ["--start", "1993-01-01", "--end", "1993-01-31"]


def unwrapped(lon) -> np.ndarray:
    return np.where(np.asarray(lon, dtype=float) < 0, np.asarray(lon, dtype=float) + 360, lon)


def joined(tmp_path: Path, name: str) -> xr.Dataset:
    return xr.open_dataset(folder(tmp_path) / name).load()


def test_npacific_plan_requests_two_parts_west_first(tmp_path, capsys, monkeypatch):
    monkeypatch.delitem(sys.modules, "copernicusmarine", raising=False)
    assert dg.main(["--data-dir", str(tmp_path), "--region", "npacific", "--years", "1993"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [x["file"] for x in lines] == ["thetao_1993_west.nc", "thetao_1993_east.nc"]
    assert {x["joins_into"] for x in lines} == {"thetao_1993.nc"}
    west, east = lines
    assert (west["minimum_longitude"], west["maximum_longitude"]) == (160.0, 180.0)
    assert (east["minimum_longitude"], east["maximum_longitude"]) == (-180.0, -130.0)
    assert all(x["minimum_longitude"] <= x["maximum_longitude"] for x in lines)
    assert west["id"] != east["id"]


def test_crossing_box_is_joined_west_to_east(tmp_path, client, monkeypatch):
    monkeypatch.setattr(dg, "JOIN_SLAB_BYTES", 1)  # one day per slab: exercises the slabs
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    assert finals(tmp_path) == ["thetao_19930101-19930131.nc"]
    assert sorted(p.name for p in (folder(tmp_path) / "parts").glob("*.nc")) == [
        "thetao_19930101-19930131_east.nc",
        "thetao_19930101-19930131_west.nc",
    ]
    ds = joined(tmp_path, "thetao_19930101-19930131.nc")
    lon = ds["longitude"].values
    assert lon[0] == 150 and lon[-1] == -150  # stored longitudes, not unwrapped
    assert lon.size == 30 * 12 + 30 * 12 + 1  # 150..179.917 and -180..-150
    step = np.diff(unwrapped(lon))
    assert np.allclose(step, 1 / 12, atol=1e-4)  # no repeat and no gap at the seam
    assert ds.sizes["time"] == 31 and ds.sizes["depth"] == 3
    expected = 15 + 0.01 * lon.astype(float)  # the fake writes values that follow longitude
    assert np.allclose(ds["thetao"].isel(time=-1, depth=0, latitude=0), expected, atol=2e-3)
    assert ds["thetao"].encoding["dtype"] == np.dtype("int16")  # copied packed
    parts = folder(tmp_path) / "parts"  # parts and join follow --compress (default zlib 4)
    joined_file = folder(tmp_path) / "thetao_19930101-19930131.nc"
    assert (
        aq.file_encoding(joined_file)
        == aq.file_encoding(parts / "thetao_19930101-19930131_west.nc")
        == "zlib-4"
    )
    (manifest,) = manifests(tmp_path)
    assert manifest["box"]["ok"] and manifest["box"]["policy"] == "enclose"
    assert manifest["request"]["longitude_parts"] == [[150.0, 180.0], [-180.0, -150.0]]
    assert [r["minimum_longitude"] for r in manifest["requests"]] == [150.0, -180.0]
    name = "thetao_19930101-19930131.nc"
    assert manifest["parts"][name] == [
        "thetao_19930101-19930131_west.nc",
        "thetao_19930101-19930131_east.nc",
    ]
    assert len(manifest["files"][name]["content_sha256"]) == 64
    assert set(manifest["coordinates_extent"][name]) == set(manifest["parts"][name])
    assert manifest["file_checks"][name]["longitude"]["seam_repeats_dropped"] == 0


def test_rerun_reuses_parts_and_join(tmp_path, client):
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    path = folder(tmp_path) / "thetao_19930101-19930131.nc"
    before = path.stat().st_mtime_ns
    client.calls.clear()
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    assert client.calls == [] and path.stat().st_mtime_ns == before


def test_seam_node_stored_on_both_sides_is_kept_once(tmp_path, client):
    client.lon_last_k = 2160  # the grid also has 180, the same meridian as -180
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    ds = joined(tmp_path, "thetao_19930101-19930131.nc")
    assert ds.sizes["longitude"] == 721
    assert np.allclose(np.diff(unwrapped(ds["longitude"].values)), 1 / 12, atol=1e-4)
    (manifest,) = manifests(tmp_path)
    checks = manifest["file_checks"]["thetao_19930101-19930131.nc"]
    assert checks["longitude"]["seam_repeats_dropped"] == 1


def test_gap_at_the_seam_fails_the_join(tmp_path, client, capsys):
    client.fail[(1993, "east")] = ["no-seam-node"]  # the east part starts at -179.917
    assert run(tmp_path, *JANUARY, box=CROSS) == 1
    assert finals(tmp_path) == []
    assert "gap of" in capsys.readouterr().out


def test_one_failing_part_is_reported_and_resumed(tmp_path, client, capsys):
    client.fail[(1993, "east")] = [DatasetVersionNotFound("no version")]
    assert run(tmp_path, *JANUARY, box=CROSS) == 1
    out = capsys.readouterr().out
    assert "thetao_19930101-19930131.nc (joined) [failed]" in out
    assert "parts not validated" in out and "thetao_19930101-19930131_east.nc failed" in out
    assert finals(tmp_path) == []
    (manifest,) = manifests(tmp_path)
    assert manifest["files"] == {}
    assert sorted(manifest["statuses"].values()) == ["failed", "validated"]
    client.calls.clear()
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    assert [c["minimum_longitude"] for c in client.calls] == [-180.0]  # only the east part
    assert finals(tmp_path) == ["thetao_19930101-19930131.nc"]


def test_npacific_region_downloads_its_whole_box(tmp_path, client):
    args = ["--data-dir", str(tmp_path), "--region", "npacific", "--max-depth", "1"]
    assert dg.main([*args, "--start", "1993-01-01", "--end", "1993-01-02", "--download"]) == 0
    assert folder(tmp_path).name == "glorys12v1_npacific"
    (manifest,) = manifests(tmp_path)
    assert manifest["source"] == "glorys_npacific"
    box = manifest["box"]
    assert box["ok"] and box["latitude"]["n"] == 361 and box["longitude"]["n"] == 841
    assert box["longitude"]["nodes"] == [160.0, 230.0]  # unwrapped frame of the report


def test_crossing_partial_year_on_demand_and_refreshed(tmp_path, client):
    box = ["--box", "25", "25.5", "179", "-179"]
    assert run(tmp_path, "--years", "2026", "--allow-partial", box=box) == 0
    assert finals(tmp_path) == ["thetao_2026_partial.nc"]
    (manifest,) = manifests(tmp_path)
    assert manifest["partial_files"] == ["thetao_2026_partial.nc"]
    assert manifest["file_checks"]["thetao_2026_partial.nc"]["last_day"] == "2026-09-22"
    client.last = pd.Timestamp("2027-06-30")
    client.calls.clear()
    assert run(tmp_path, "--years", "2026", box=box) == 0
    assert len(client.calls) == 2  # both parts fetched again
    assert finals(tmp_path) == ["thetao_2026.nc"]
    assert joined(tmp_path, "thetao_2026.nc").sizes["time"] == 365


def test_crossing_box_partial_year_rejected_by_default(tmp_path, client):
    box = ["--box", "25", "25.5", "179", "-179"]
    assert run(tmp_path, "--years", "2026", box=box) == 1
    assert finals(tmp_path) == []


@pytest.mark.parametrize("policy", ["inside", "cover-cells"])
def test_crossing_box_other_policies(tmp_path, client, policy):
    box = ["--box", "25.01", "25.49", "179.01", "-179.01"]
    assert run(tmp_path, *JANUARY, "--box-policy", policy, box=box) == 0
    (manifest,) = manifests(tmp_path)
    assert manifest["box"]["ok"] and manifest["box"]["policy"] == policy


@pytest.mark.parametrize("west, east", [(179.99, -179.8), (179.99, -179.95)])
def test_cover_cells_shrink_never_sends_inverted_longitudes(west, east):
    # Review H1: once shrunk, the box may stop crossing 180 deg, or collapse to one point.
    box = dg.aq.Box(25.0, 25.5, west, east)
    kw = dict(variables=["thetao"], max_depth=1.0, dataset_version="202311", service="s")
    for req in dg.requests(date(1993, 1, 1), date(1993, 1, 2), box, "cover-cells", **kw):
        assert -180 <= req["minimum_longitude"] <= req["maximum_longitude"] <= 180


def test_cover_cells_box_that_stops_crossing_is_covered(tmp_path, client):
    box = ["--box", "25.01", "25.49", "179.99", "-179.8"]
    assert run(tmp_path, *JANUARY, "--box-policy", "cover-cells", box=box) == 0
    assert len(client.calls) == 1
    (manifest,) = manifests(tmp_path)
    assert manifest["box"]["ok"]


def test_join_keeps_nan_fill_values(tmp_path):
    # Review H3: two parts with _FillValue = NaN are stored alike.
    paths = []
    for name, lon in (("w", [179.75, 179.8333, 179.9167]), ("e", [-180.0, -179.9167])):
        values = np.full((2, 1, 1, len(lon)), 15.0, "f4")
        values[0, 0, 0, 0] = np.nan
        ds = xr.Dataset(
            {"thetao": (("time", "depth", "latitude", "longitude"), values)},
            coords={
                "time": pd.date_range("1993-01-01", periods=2),
                "depth": [0.5],
                "latitude": np.array([25.0], "f4"),
                "longitude": np.array(lon, "f4"),
            },
        )
        paths.append(tmp_path / f"{name}.nc")
        ds.to_netcdf(paths[-1], engine="h5netcdf", encoding={"thetao": {"_FillValue": np.nan}})
    out = tmp_path / "joined.nc"
    box = dg.aq.Box(25.0, 25.0, 179.75, -179.9167)
    checks = dg.join_parts(paths, out, box=box, policy="inside", variables=["thetao"])
    assert checks["longitude"]["n"] == 5
    with xr.open_dataset(out) as ds:
        assert np.isnan(ds["thetao"].encoding["_FillValue"])
        assert int(ds["thetao"].isnull().sum()) == 2  # one per part, in place
        assert np.isnan(ds["thetao"].values[0, 0, 0, 3])


def test_join_cut_before_its_record_is_rebuilt(tmp_path, client, monkeypatch):
    # Review H4: a join published but not recorded is not refused as a foreign file.
    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(dg.aq.Ledger, "record_product", crash)
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, *JANUARY, box=CROSS)
    assert finals(tmp_path) == ["thetao_19930101-19930131.nc"]
    client.calls.clear()
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    assert client.calls == []
    pending = json.loads((folder(tmp_path) / "joins_pending.json").read_text(encoding="utf-8"))
    assert pending == {}
    (*_, manifest) = manifests(tmp_path)
    assert list(manifest["files"]) == ["thetao_19930101-19930131.nc"]


def test_manifest_omits_a_join_that_could_not_be_rebuilt(tmp_path, client, monkeypatch):
    # Review H5: a deleted join whose rebuild fails is not listed from its old record.
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    (folder(tmp_path) / "thetao_19930101-19930131.nc").unlink()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dg, "join_parts", fail)
    assert run(tmp_path, *JANUARY, box=CROSS) == 1
    first, second = manifests(tmp_path)
    assert list(first["files"]) == ["thetao_19930101-19930131.nc"]
    assert second["files"] == {}


@pytest.mark.parametrize(
    "box, policy",
    [
        (["--box", "25", "25.5", "179.9", "-179.99"], "cover-cells"),  # re-review H1
        (["--box", "25", "25.5", "179.97", "-179.99"], "cover-cells"),  # collapses to a point
        (["--box", "25", "25.5", "179.8", "179.95"], "enclose"),  # needs the -180 node
    ],
)
def test_box_ending_past_the_last_node_gets_the_minus_180_node(tmp_path, client, box, policy):
    assert run(tmp_path, *JANUARY, "--box-policy", policy, box=box) == 0
    assert [c["minimum_longitude"] for c in client.calls][-1] == -180.0
    (manifest,) = manifests(tmp_path)
    assert manifest["box"]["ok"]
    (name,) = finals(tmp_path)
    assert joined(tmp_path, name)["longitude"].values[-1] == -180.0


def test_another_encoding_rewrites_parts_and_join_without_downloading(tmp_path, client):
    assert run(tmp_path, *JANUARY, "--no-compress", box=CROSS) == 0
    files = sorted(folder(tmp_path).rglob("thetao_*.nc"))
    assert len(files) == 3 and {aq.file_encoding(p) for p in files} == {"none"}
    content = {p: aq.content_sha256(p) for p in files}
    client.calls.clear()
    assert run(tmp_path, *JANUARY, box=CROSS) == 0
    assert client.calls == []
    assert {aq.file_encoding(p) for p in files} == {"zlib-4"}
    assert {p: aq.content_sha256(p) for p in files} == content
    joined_file = folder(tmp_path) / "thetao_19930101-19930131.nc"
    manifest = [m for m in manifests(tmp_path) if joined_file.name in m["files"]][-1]
    assert manifest["files"][joined_file.name]["sha256"] == aq.file_sha256(joined_file)
