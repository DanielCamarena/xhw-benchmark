# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of download_sst_cci.py (revision 6): a fake CEDA archive serves local files.

The fake client answers listings, HEAD, the 8-byte probe and ranged reads from small HDF5
files with the real 0.05 degree longitudes (global, so the antimeridian is exercised) and a
latitude band. No network, no fsspec/aiohttp/requests.
"""

from __future__ import annotations

import functools
import json
import sys
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _acquire as aq  # noqa: E402
import download_sst_cci as d  # noqa: E402
from content_hash import content_sha256  # noqa: E402

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")

LAT = np.arange(-14.975, 7.0, 0.05).astype("float32")  # a band of the real cell centres
LON = np.arange(-179.975, 180.0, 0.05).astype("float32")  # the real 7200 centres
DAYS = {
    "2020-01-01": "CDR",
    "2020-01-02": "CDR",
    "2020-01-03": "CDR",
    "2021-12-31": "CDR",
    "2022-01-01": "ICDR",
}
EPOCH = date(1981, 1, 1)


class FakeHTTPError(Exception):
    """Like requests.HTTPError: carries ``response.status_code``."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class ClientPayloadError(Exception):
    """Same class name as aiohttp's error for a cut transfer."""


def file_name(day: str, series: str) -> str:
    stamp = day.replace("-", "")
    return f"{stamp}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_{series}3.0-v02.0-fv01.0.nc"


def write_day(path: Path, day: str, offset: int = 0, mask: bool = True) -> None:
    n = list(DAYS).index(day) if day in DAYS else 9
    lon_i = np.arange(LON.size) % 97
    lat_i = np.arange(LAT.size)[:, None] % 5
    sst = (n * 1000 + offset + lon_i[None, :] + 100 * lat_i).astype("int16")[None]
    seconds = (date.fromisoformat(day) - EPOCH).days * 86400 + 43200
    data = {
        "analysed_sst": (
            ("time", "lat", "lon"),
            sst,
            {
                "scale_factor": 0.01,
                "add_offset": 273.15,
                "_FillValue": np.int16(-32768),
                "units": "kelvin",
            },
        )
    }
    if mask:
        data["mask"] = (("time", "lat", "lon"), np.ones_like(sst, dtype="int8"))
    ds = xr.Dataset(
        data,
        coords={
            "time": ("time", np.array([seconds], "int32"), {"units": "seconds since 1981-01-01"}),
            "lat": LAT,
            "lon": LON,
        },
        attrs={"title": "fake ESA SST CCI", "date_created": day},
    )
    encoding = {v: {"zlib": True, "chunksizes": (1, 110, 1800)} for v in ds.data_vars}
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)


@pytest.fixture(scope="session")
def day_files(tmp_path_factory) -> dict:
    folder = tmp_path_factory.mktemp("ceda_files")
    files = {}
    for day, series in DAYS.items():
        files[day] = folder / file_name(day, series)
        write_day(files[day], day)
    files["changed"] = folder / "changed.nc"
    write_day(files["changed"], "2020-01-02", offset=7)
    files["no_mask"] = folder / "no_mask.nc"
    write_day(files["no_mask"], "2020-01-03", mask=False)
    return files


class FakeClient:
    """The CEDA archive as seen through ``download_sst_cci.CedaClient``'s methods.

    Its state is a JSON file, so it works in worker processes too. ``errors`` maps
    ``"<operation>:<day>"`` to a list of outcomes per call (an HTTP status, ``"payload"``,
    ``"timeout"`` or ``"ok"``); every call is an empty file in ``calls/`` named after it.

    One file per call, never a shared log: worker processes appending to and reading one
    file interleave on Windows (text mode writes ``\\r\\n`` in pieces), and a reader then
    parses a torn or empty line. Counting file names needs no reads.
    """

    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "archive.json").read_text())

    def _log(self, op: str, day: str) -> int:
        calls = self.folder / "calls"
        calls.mkdir(exist_ok=True)
        key = f"{op}@{day.replace('/', '_')}"
        previous = len(list(calls.glob(f"{key}~*")))
        (calls / f"{key}~{time.time_ns():020d}~{uuid.uuid4().hex[:8]}").open("x").close()
        outcomes = self.config.get("errors", {}).get(f"{op}:{day}", [])
        outcome = outcomes[previous] if previous < len(outcomes) else "ok"
        if outcome == "payload":
            raise ClientPayloadError("Response payload is not completed")
        if outcome == "timeout":
            raise TimeoutError("read timed out")
        if isinstance(outcome, int):
            raise FakeHTTPError(outcome)
        return previous

    def _day_of(self, url: str) -> str:
        name = url.rsplit("/", 1)[-1]
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"

    def listing(self, url: str) -> str:
        rel = url[len(d.BASE_URL) :].strip("/")
        parts = rel.split("/") if rel else []
        days = self.config["days"]
        if len(parts) == 3:
            day = "-".join(parts)
            self._log("listing", day)
            if day not in days:
                raise FakeHTTPError(404)
            names = [days[day]["name"], *days[day].get("extra", [])]
            links = ["../", *names]
        else:
            self._log("listing", rel)
            prefix = "-".join(parts)
            children = sorted(
                {k.split("-")[len(parts)] for k in days if k.startswith(prefix)} if prefix else ()
            )
            if not children:
                raise FakeHTTPError(404)
            # Absolute links, like CEDA's, plus a parent link that must be ignored.
            links = ["../", *(f"{urlpath(url)}{c}/" for c in children)]
        return "".join(f'<a href="{h}">{h}</a>' for h in links)

    def _entry(self, url: str) -> dict:
        return self.config["days"][self._day_of(url)]

    def head(self, url: str) -> dict:
        day = self._day_of(url)
        self._log("head", day)
        entry = self._entry(url)
        return {"etag": entry.get("head_etag", entry["etag"]), "last_modified": "Thu, 04 Apr"}

    def probe(self, url: str) -> tuple[int, bytes, dict]:
        day = self._day_of(url)
        self._log("probe", day)
        entry = self._entry(url)
        first = Path(entry["path"]).read_bytes()[:8] if not entry.get("not_hdf5") else b"<html>  "
        headers = {"etag": entry["etag"], "last_modified": "Thu, 04 Apr 2024 10:00:00 GMT"}
        return entry.get("probe_status", 206), first, headers

    def open(self, url: str):
        day = self._day_of(url)
        self._log("open", day)
        return open(self._entry(url)["path"], "rb")

    def close(self) -> None:
        pass


def urlpath(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).path


class Archive:
    """A fake archive folder and the helpers to run ``main`` against it."""

    def __init__(self, folder: Path, files: dict, days=("2020-01-01", "2020-01-02", "2020-01-03")):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.config = {
            "days": {
                day: {
                    "name": file_name(day, DAYS[day]),
                    "path": str(files[day]),
                    "etag": f'"etag-{day}"',
                }
                for day in days
            },
            "errors": {},
        }
        self.files = files
        self.save()

    def save(self) -> None:
        (self.folder / "archive.json").write_text(json.dumps(self.config))

    def day(self, day: str) -> dict:
        return self.config["days"][day]

    @property
    def factory(self):
        return functools.partial(FakeClient, str(self.folder))

    def _call_files(self) -> list[Path]:
        folder = self.folder / "calls"
        return list(folder.iterdir()) if folder.exists() else []

    def calls(self, op: str | None = None) -> list:
        """``[operation, day]`` of every call, in time order."""
        names = sorted(self._call_files(), key=lambda p: p.name.split("~")[1])
        out = []
        for path in names:
            name, day = path.name.split("~")[0].split("@")
            out.append([name, day.replace("_", "/")])
        return [c for c in out if op is None or c[0] == op]

    def reset_calls(self) -> None:
        for path in self._call_files():
            path.unlink()


@pytest.fixture
def archive(tmp_path, day_files) -> Archive:
    return Archive(tmp_path / "archive", day_files)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(d, "WAIT", lambda n: 0.0)


def run(archive: Archive, data: Path, *args: str, workers: int = 1) -> int:
    argv = [
        "--data-dir", str(data), "--start", "2020-01-01", "--end", "2020-01-03",
        "--workers", str(workers), "--download", *args,
    ]  # fmt: skip
    return d.main(argv, client_factory=archive.factory)


def outputs(data: Path) -> list[Path]:
    return sorted(data.glob("sst_cci_v3_*.nc"))


def manifests(data: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(data.glob("sst_cci_manifest_*.json"))]


def ledger_of(data: Path) -> dict:
    (path,) = data.glob("sst_cci_ledger_*.json")
    return json.loads(path.read_text())


def open_raw(path: Path):
    return xr.open_dataset(path, engine="h5netcdf", decode_cf=False, decode_times=False)


# --- Print only and argument checks ---------------------------------------------------------


def test_print_only_uses_no_client(tmp_path, capsys):
    def no_client():
        raise AssertionError("print-only must not contact the archive")

    assert d.main(["--data-dir", str(tmp_path)], client_factory=no_client) == 0
    plan = json.loads(capsys.readouterr().out.split("\nPrint only")[0])
    assert plan["box_policy"] == "enclose" and plan["region"] == "peru"
    assert plan["series"] == "CDR" and plan["days"] == 11323
    assert plan["request_id"] in plan["output"] and not list(tmp_path.iterdir())


def test_end_after_cdr_needs_include_icdr(tmp_path):
    argv = ["--data-dir", str(tmp_path), "--start", "2021-12-31", "--end", "2022-01-01"]
    with pytest.raises(SystemExit) as exc:
        d.main(argv)
    assert exc.value.code == 2
    assert d.main([*argv, "--include-icdr"]) == 0  # print only


def test_start_before_the_record_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        d.main(["--data-dir", str(tmp_path), "--start", "1981-08-31", "--end", "1981-09-02"])


# --- Download, combination and manifest -----------------------------------------------------


def test_download_encloses_box_and_keeps_stored_values(archive, tmp_path):
    data = tmp_path / "data"
    assert run(archive, data, "--box", "-12.1", "-3.9", "-82.1", "-76.9") == 0
    (out,) = outputs(data)
    with open_raw(out) as ds, open_raw(archive.files["2020-01-02"]) as src:
        sst = ds["analysed_sst"]
        assert sst.dtype == np.int16 and sst.attrs["scale_factor"] == 0.01
        assert sst.attrs["_FillValue"] == -32768 and list(ds.data_vars) == ["analysed_sst"]
        # enclose: nodes on or beyond every edge of the unaligned box
        assert ds.lat.min() <= -12.1 and ds.lat.max() >= -3.9
        assert ds.lon.min() <= -82.1 and ds.lon.max() >= -76.9
        assert ds.lat.min() > -12.15 and ds.lon.max() < -76.85  # and no more than needed
        expected = src["analysed_sst"].sel(lat=ds.lat, lon=ds.lon).values[0]
        np.testing.assert_array_equal(sst.values[1], expected)
        assert ds.sizes["time"] == 3 and "source_url" not in ds.attrs
    (manifest,) = manifests(data)
    assert manifest["request"]["box"] == {
        "south": -12.1,
        "north": -3.9,
        "west": -82.1,
        "east": -76.9,
    }
    assert manifest["request"]["variables"] == ["analysed_sst"]
    assert manifest["box"]["ok"] and manifest["box"]["policy"] == "enclose"
    assert manifest["series_days"] == {"CDR": 3}
    assert manifest["last_published_day_checked"] == "2020-01-03"
    assert [x["day"] for x in manifest["days"]] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    day = manifest["days"][1]
    assert day["etag"] == '"etag-2020-01-02"' and day["series"] == "CDR"
    assert day["url"].endswith(file_name("2020-01-02", "CDR")) and len(day["sha256"]) == 64
    assert out.name in manifest["files"]
    entries = ledger_of(data)["pieces"].values()
    assert {e["status"] for e in entries} == {"validated"}
    assert all(e["source"]["last_modified"] for e in entries)
    assert not list(data.rglob("*.unvalidated.nc")) and not list(data.rglob("*.tmp"))


def test_reuse_needs_same_box_and_variables(archive, tmp_path):
    """R1: parts of one request are never reused for another box or other variables."""
    data = tmp_path / "data"
    assert run(archive, data, "--region", "peru") == 0
    archive.reset_calls()
    assert run(archive, data, "--box", "-14", "0", "-90", "-70") == 0
    assert len(archive.calls("open")) == 3  # fetched again, not reused
    assert run(archive, data, "--region", "peru", "--variables", "mask") == 0
    assert len(archive.calls("open")) == 6
    assert len(list((data / "sst_cci_partial").iterdir())) == 3
    for manifest in manifests(data):
        (name,) = manifest["files"]
        with open_raw(data / name) as ds:
            assert list(ds.data_vars) == manifest["request"]["variables"]
            box = manifest["request"]["box"]
            assert ds.lat.min() <= box["south"] and ds.lat.max() >= box["north"]
            assert ds.lat.max() - ds.lat.min() < box["north"] - box["south"] + 0.11


def test_no_join_keeps_the_daily_parts_and_a_later_run_combines_them(archive, tmp_path):
    data = tmp_path / "data"
    assert run(archive, data, "--no-join") == 0
    assert len(list((data / "sst_cci_partial").rglob("*.nc"))) == 3
    assert outputs(data) == [] and manifests(data) == []
    assert not ledger_of(data).get("products")
    archive.reset_calls()
    assert run(archive, data) == 0
    assert archive.calls("open") == [] and len(outputs(data)) == 1 and len(manifests(data)) == 1


def test_resume_reuses_parts_and_checks_etag(archive, tmp_path):
    data = tmp_path / "data"
    assert run(archive, data) == 0
    first = outputs(data)[0].read_bytes()
    archive.reset_calls()
    assert run(archive, data) == 0
    assert archive.calls("open") == [] and len(archive.calls("head")) == 3
    assert outputs(data)[0].read_bytes() == first  # product current: not rewritten
    assert len(manifests(data)) == 2
    archive.reset_calls()
    assert run(archive, data, "--no-check-etag") == 0
    assert archive.calls("head") == [] and archive.calls("open") == []


def test_changed_etag_fetches_the_day_again(archive, tmp_path):
    """Decision 3: a changed ETag re-fetches the day, is logged and rebuilds the product."""
    data = tmp_path / "data"
    assert run(archive, data) == 0
    archive.day("2020-01-02").update(
        etag='"etag-new"', head_etag='"etag-new"', path=str(archive.files["changed"])
    )
    archive.save()
    archive.reset_calls()
    assert run(archive, data) == 0
    assert archive.calls("open") == [["open", "2020-01-02"]]
    entry = next(e for e in ledger_of(data)["pieces"].values() if e["id"].endswith("20200102"))
    assert entry["etag_changes"][0]["old_etag"] == '"etag-2020-01-02"'
    assert entry["etag_changes"][0]["new_etag"] == '"etag-new"'
    assert entry["source"]["etag"] == '"etag-new"'
    (out,) = outputs(data)
    with open_raw(out) as ds, open_raw(archive.files["changed"]) as src:
        expected = src["analysed_sst"].sel(lat=ds.lat, lon=ds.lon).values[0]
        np.testing.assert_array_equal(ds["analysed_sst"].values[1], expected)
    assert manifests(data)[-1]["days"][1]["etag"] == '"etag-new"'


def test_end_after_last_published_day_is_rejected(archive, tmp_path, capsys):
    """R4: the last published day is looked up before any day is fetched."""
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--start", "2020-01-01", "--end", "2020-01-05", "--download"]
    assert d.main(argv, client_factory=archive.factory) == 1
    assert "latest published day on or before it is 2020-01-03" in capsys.readouterr().err
    assert archive.calls("open") == [] and archive.calls("probe") == []


def test_last_published_day_walks_back_years(archive):
    client = archive.factory()
    assert d.last_published_day(client, date(2021, 3, 1)) == date(2020, 1, 3)
    assert d.last_published_day(client, date(2020, 1, 2)) == date(2020, 1, 2)
    with pytest.raises(d.NotPublished):
        d.last_published_day(client, date(2019, 12, 31))


def test_icdr_days_are_marked_in_the_manifest(tmp_path, day_files):
    archive = Archive(tmp_path / "archive", day_files, days=("2021-12-31", "2022-01-01"))
    data = tmp_path / "data"
    argv = [
        "--data-dir", str(data), "--start", "2021-12-31", "--end", "2022-01-01",
        "--include-icdr", "--workers", "1", "--download",
    ]  # fmt: skip
    assert d.main(argv, client_factory=archive.factory) == 0
    (manifest,) = manifests(data)
    assert manifest["series_days"] == {"CDR": 1, "ICDR": 1}
    assert [x["series"] for x in manifest["days"]] == ["CDR", "ICDR"]


def test_antimeridian_box_is_read_in_two_parts(archive, tmp_path):
    """A1/B5: 150E-150W crosses the antimeridian; both parts are joined west to east."""
    data = tmp_path / "data"
    assert run(archive, data, "--box", "-5", "5", "150", "-150") == 0
    (out,) = outputs(data)
    with open_raw(out) as ds:
        lon = ds.lon.values
        assert lon[0] <= 150 and lon[-1] >= -150 and lon[0] > 149.9 and lon[-1] < -149.9
        east = (lon + 360) % 360
        np.testing.assert_allclose(np.diff(east), 0.05, atol=1e-4)  # continuous across 180
        assert ds.sizes["lon"] == 1202 and ds["analysed_sst"].dtype == np.int16
    assert manifests(data)[0]["box"]["ok"]


# --- Errors ---------------------------------------------------------------------------------


def test_transient_errors_are_retried(archive, tmp_path):
    archive.config["errors"] = {
        "listing:2020-01-01": [503, 429],
        "probe:2020-01-02": ["timeout"],
        "open:2020-01-03": ["payload"],
    }
    archive.save()
    assert run(archive, tmp_path / "data") == 0
    assert len(archive.calls("listing")) >= 6  # 3 days + 2 retries + the published-day lookup


def test_every_permanent_error_is_reported_without_retry(archive, tmp_path, capsys, day_files):
    """R6/R7: 404, no Range, not HDF5, two files listed: each day fails once, all reported."""
    archive.config["days"]["2020-01-04"] = {
        "name": file_name("2020-01-04", "CDR"),
        "path": str(day_files["2020-01-01"]),
        "etag": "x",
        "extra": [file_name("2020-01-04", "CDR").replace("fv01", "fv02")],
    }
    archive.config["days"]["2020-01-05"] = {**archive.day("2020-01-01"), "name": "x.nc"}
    archive.config["days"]["2020-01-05"]["name"] = file_name("2020-01-05", "CDR")
    archive.day("2020-01-01")["probe_status"] = 200
    archive.day("2020-01-02")["not_hdf5"] = True
    archive.config["errors"] = {"probe:2020-01-03": [404], "open:2020-01-05": [410]}
    archive.save()
    data = tmp_path / "data"
    argv = ["--data-dir", str(data), "--start", "2020-01-01", "--end", "2020-01-05"]
    assert d.main([*argv, "--workers", "1", "--download"], client_factory=archive.factory) == 1
    out = capsys.readouterr().out
    assert "5 pieces: 5 failed" in out
    for text in ("RangeNotSupported", "NotHdf5", "FakeHTTPError: HTTP 404", "ListingError"):
        assert text in out
    assert len(archive.calls("probe")) == 4 and len(archive.calls("open")) == 1  # no retries
    assert outputs(data) == [] and manifests(data) == []


def test_missing_variable_is_permanent(archive, tmp_path, capsys):
    archive.day("2020-01-03")["path"] = str(archive.files["no_mask"])
    archive.save()
    assert run(archive, tmp_path / "data", "--variables", "analysed_sst", "mask") == 1
    out = capsys.readouterr().out
    assert "2 validated" in out and "MissingVariables" in out
    assert len(archive.calls("open")) == 3


def test_icdr_file_without_include_icdr_is_refused(tmp_path, day_files):
    archive = Archive(tmp_path / "archive", day_files)
    archive.day("2020-01-02")["name"] = file_name("2020-01-02", "ICDR")
    archive.save()
    assert run(archive, tmp_path / "data") == 1


def test_errors_are_collected_with_processes(archive, tmp_path, capsys):
    """R7 with a process pool: two failing days do not hide each other or the third."""
    archive.day("2020-01-01")["probe_status"] = 200
    archive.day("2020-01-03")["not_hdf5"] = True
    archive.save()
    assert run(archive, tmp_path / "data", workers=2) == 1
    out = capsys.readouterr().out
    assert "1 validated" in out and "RangeNotSupported" in out and "NotHdf5" in out


def test_damaged_part_goes_to_quarantine_with_keep_failed(archive, tmp_path, monkeypatch):
    def broken(path, *args):
        raise aq.ValidationError("broken part")

    monkeypatch.setattr(d, "validate_part", broken)
    data = tmp_path / "data"
    assert run(archive, data, "--keep-failed") == 1
    entries = ledger_of(data)["pieces"].values()
    assert {e["status"] for e in entries} == {"quarantined"}
    assert all(e["damage"][0]["sha256"] for e in entries)
    assert len(list(data.rglob("quarantine/*.nc"))) == 3


def test_classification():
    assert d.classify(FakeHTTPError(503)) == aq.TRANSIENT
    assert d.classify(FakeHTTPError(429)) == aq.TRANSIENT
    assert d.classify(FakeHTTPError(404)) == aq.PERMANENT
    assert d.classify(ClientPayloadError("cut")) == aq.TRANSIENT
    assert d.classify(TimeoutError()) == aq.TRANSIENT
    assert d.classify(FileNotFoundError("fsspec 404")) == aq.PERMANENT
    for exc in (d.RangeNotSupported(), d.NotHdf5(), d.MissingVariables(), d.ListingError()):
        assert d.classify(exc) == aq.PERMANENT


# --- Combination ----------------------------------------------------------------------------


def _part(path: Path, lat, value: int, day: int, dtype="int16") -> Path:
    ds = xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), np.full((1, len(lat), 3), value, dtype))},
        coords={"time": [day], "lat": lat, "lon": [-80.025, -79.975, -79.925]},
    )
    ds["analysed_sst"].attrs["scale_factor"] = 0.01
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def test_combine_refuses_different_grids_and_dtypes(tmp_path):
    """R2: parts of different boxes or dtypes are never merged with an outer join."""
    box = aq.Box(-12.0, -11.95, -80.0, -79.95)
    lat = [-12.025, -11.975, -11.925]
    a = _part(tmp_path / "a.nc", lat, 100, 1)
    b = _part(tmp_path / "b.nc", [x - 0.05 for x in lat], 200, 2)
    c = _part(tmp_path / "c.nc", lat, 200, 2, dtype="float32")
    out = tmp_path / "out.nc"
    with pytest.raises(aq.ValidationError, match="lat differs"):
        d.combine([a, b], out, box, "enclose", ["analysed_sst"])
    with pytest.raises(aq.ValidationError, match="dtype"):
        d.combine([a, c], out, box, "enclose", ["analysed_sst"])
    with pytest.raises(aq.ValidationError, match="strictly increasing"):
        d.combine([a, a], out, box, "enclose", ["analysed_sst"])
    good = _part(tmp_path / "g.nc", lat, 200, 2)
    report = d.combine([a, good], out, box, "enclose", ["analysed_sst"])
    assert report["ok"]
    with open_raw(out) as ds:
        assert ds["analysed_sst"].dtype == np.int16
        assert ds["analysed_sst"].values[:, 0, 0].tolist() == [100, 200]


def test_combined_file_not_written_by_the_ledger_is_not_overwritten(archive, tmp_path, capsys):
    data = tmp_path / "data"
    assert run(archive, data) == 0
    (out,) = outputs(data)
    out.write_bytes(b"edited by hand")
    assert run(archive, data) == 1
    assert "is not the one this ledger wrote (changed)" in capsys.readouterr().err
    assert out.read_bytes() == b"edited by hand"


# --- Review round 1 (GPT, 2026-11-03): F1-F4 ------------------------------------------------


def test_any_5xx_is_transient():
    """F4: revision 6 retries every 5xx, not only the shared 500/502/503/504."""
    assert d.classify(FakeHTTPError(507)) == aq.TRANSIENT
    assert d.classify(FakeHTTPError(501)) == aq.TRANSIENT


def test_part_of_another_day_is_rejected(archive, tmp_path, capsys):
    """F1: a file served under one day's name but holding another day does not validate."""
    archive.day("2020-01-02")["path"] = str(archive.files["2020-01-01"])
    archive.save()
    assert run(archive, tmp_path / "data") == 1
    out = capsys.readouterr().out
    assert "time decodes to 2020-01-01, not the requested 2020-01-02" in out
    assert outputs(tmp_path / "data") == []


def test_combine_refuses_different_time_units(tmp_path):
    """F1: raw times are only comparable with the same units and calendar."""
    box = aq.Box(-12.0, -11.95, -80.0, -79.95)
    lat = [-12.025, -11.975, -11.925]
    a = _part(tmp_path / "a.nc", lat, 100, 1)
    b = _part(tmp_path / "b.nc", lat, 200, 2)
    for path, units in ((a, "seconds since 2020-01-01"), (b, "days since 2020-01-01")):
        with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as ds:
            ds = ds.load()
        ds["time"].attrs["units"] = units
        ds.to_netcdf(path, engine="h5netcdf")
    with pytest.raises(aq.ValidationError, match="time units differs"):
        d.combine([a, b], tmp_path / "out.nc", box, "enclose", ["analysed_sst"])


def test_icdr_part_is_not_reused_without_include_icdr(tmp_path, day_files, capsys):
    """F2: a part fetched with --include-icdr is refused by a run without it."""
    archive = Archive(tmp_path / "archive", day_files)
    archive.day("2020-01-02")["name"] = file_name("2020-01-02", "ICDR")
    archive.save()
    data = tmp_path / "data"
    assert run(archive, data, "--include-icdr") == 0
    archive.reset_calls()
    assert run(archive, data, "--no-check-etag") == 1
    assert "pass --include-icdr" in capsys.readouterr().out
    assert [c[1] for c in archive.calls("listing") if c[1].count("-") == 2] == ["2020-01-02"]


def test_interrupted_combination_is_rebuilt_on_resume(archive, tmp_path, monkeypatch):
    """F3: a crash between publishing the combined file and recording it does not block."""
    data = tmp_path / "data"

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(aq.Ledger, "record_product", crash)
        with pytest.raises(KeyboardInterrupt):
            run(archive, data)
    (out,) = outputs(data)
    assert run(archive, data) == 0
    assert ledger_of(data)["products"][out.name]["sha256"] == aq.file_sha256(out)
    assert ledger_of(data)["pending_products"] == {}


# --- Re-review (GPT, 2026-11-03): F1-R2, F3-R2 ----------------------------------------------


def test_reused_part_of_another_day_is_refused_when_combining(archive, tmp_path, capsys):
    """F1-R2: a validated part from an older ledger is checked against its day too."""
    data = tmp_path / "data"
    assert run(archive, data) == 0
    ledger = ledger_of(data)
    entry = next(e for e in ledger["pieces"].values() if e["id"].endswith("20200102"))
    part = data / entry["piece"]["path"]
    with open_raw(part) as ds:
        ds = ds.load()
    ds["time"] = ds["time"] - 86400  # one day earlier, units unchanged
    ds.to_netcdf(part, engine="h5netcdf")
    entry["piece"]["sha256"] = aq.file_sha256(part)  # as if it had been validated so
    (path,) = data.glob("sst_cci_ledger_*.json")
    path.write_text(json.dumps(ledger))
    outputs(data)[0].unlink()
    assert run(archive, data, "--no-check-etag") == 1
    assert "time decodes to 2020-01-01, not 2020-01-02" in capsys.readouterr().err
    assert outputs(data) == []


def test_interrupted_combination_is_rebuilt_after_an_etag_change(archive, tmp_path, monkeypatch):
    """F3-R2: the announced rebuild still happens if a day changed before resuming."""
    data = tmp_path / "data"

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(aq.Ledger, "record_product", crash)
        with pytest.raises(KeyboardInterrupt):
            run(archive, data)
    archive.day("2020-01-02").update(
        etag='"etag-new"', head_etag='"etag-new"', path=str(archive.files["changed"])
    )
    archive.save()
    assert run(archive, data) == 0
    (out,) = outputs(data)
    with open_raw(out) as ds, open_raw(archive.files["changed"]) as src:
        expected = src["analysed_sst"].sel(lat=ds.lat, lon=ds.lon).values[0]
        np.testing.assert_array_equal(ds["analysed_sst"].values[1], expected)


# --- bounded memory (2026-11-04) -------------------------------------------------------------


def _packed_day(path: Path, day: int, *, comment: str) -> Path:
    rng = np.random.default_rng(day)
    values = rng.integers(-500, 500, (1, 3, 3)).astype("int16")
    values[0, 1, 1] = -32768  # the fill value
    ds = xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), values)},
        coords={
            "time": [day],
            "lat": [-12.025, -11.975, -11.925],
            "lon": [-80.025, -79.975, -79.925],
        },
        attrs={"title": "SST CCI", "date_created": f"2026-01-{day:02d}", "day": day},
    )
    ds["time"].attrs["units"] = "days since 2020-01-01"
    ds["analysed_sst"].attrs.update(
        scale_factor=0.01, add_offset=273.15, _FillValue=np.int16(-32768), units="kelvin",
        comment=comment,
    )  # fmt: skip
    ds.to_netcdf(path, engine="h5netcdf")
    return path


@pytest.mark.parametrize("block_days", [1, 2, 3, 31])
def test_combine_in_blocks_equals_one_concatenation(tmp_path, block_days):
    # The previous combine loaded every part and wrote one concatenation: same values,
    # same attributes (conflicting ones dropped), whatever the block.
    paths = [
        _packed_day(tmp_path / f"p{i}.nc", i, comment="b" if i == 4 else "a") for i in range(1, 8)
    ]
    parts = []
    for path in paths:
        with open_raw(path) as opened:
            parts.append(opened.load())
    whole = xr.concat(
        parts, dim="time", data_vars="minimal", coords="minimal", compat="override",
        join="exact", combine_attrs="drop_conflicts",
    )  # fmt: skip
    legacy = tmp_path / "legacy.nc"
    whole.to_netcdf(
        legacy, engine="h5netcdf",
        encoding={v: {"zlib": True, "complevel": 4, "shuffle": True} for v in whole.data_vars},
    )  # fmt: skip
    box = aq.Box(-12.0, -11.95, -80.0, -79.95)
    out = tmp_path / "out.nc"
    days = [date(2020, 1, 1) + timedelta(i) for i in range(1, 8)]
    d.combine(paths, out, box, "enclose", ["analysed_sst"], days, block_days=block_days)
    assert content_sha256(out) == content_sha256(legacy)
    assert aq.file_encoding(out) == "zlib-4"
    with open_raw(out) as got, open_raw(legacy) as ref:
        assert got.attrs == ref.attrs == {"title": "SST CCI"}
        for name in ref.variables:
            assert got[name].attrs.keys() == ref[name].attrs.keys(), name
        assert "comment" not in got["analysed_sst"].attrs
        assert got["analysed_sst"].dtype == np.int16


def test_combine_keeps_variables_without_time_from_the_first_part(tmp_path):
    # Review 2026-11-04, H1: one concat keeps the first part's static variables.
    paths = []
    for i in range(1, 5):
        path = _packed_day(tmp_path / f"p{i}.nc", i, comment="a")
        with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as opened:
            ds = opened.load()
        ds["mask"] = (("lat", "lon"), np.full((3, 3), 1 if i < 3 else 2, "int8"))
        ds.to_netcdf(path, engine="h5netcdf")
        paths.append(path)
    parts = []
    for path in paths:
        with open_raw(path) as opened:
            parts.append(opened.load())
    whole = xr.concat(
        parts, dim="time", data_vars="minimal", coords="minimal", compat="override",
        join="exact", combine_attrs="drop_conflicts",
    )  # fmt: skip
    legacy = tmp_path / "legacy.nc"
    whole.to_netcdf(legacy, engine="h5netcdf")
    box = aq.Box(-12.0, -11.95, -80.0, -79.95)
    out = tmp_path / "out.nc"
    d.combine(paths, out, box, "enclose", ["analysed_sst", "mask"], block_days=2)
    assert content_sha256(out) == content_sha256(legacy)
    with open_raw(out) as got:
        assert int(got["mask"].max()) == 1


def test_another_encoding_rewrites_days_and_output_without_fetching(archive, tmp_path):
    data = tmp_path / "data"
    assert run(archive, data) == 0
    (output,) = outputs(data)
    days = sorted(data.rglob("sst_cci_2020*.nc"))
    assert len(days) == 3
    assert {aq.file_encoding(p) for p in [output, *days]} == {"zlib-4"}
    content = {p: content_sha256(p) for p in [output, *days]}
    archive.reset_calls()
    assert run(archive, data, "--no-compress") == 0
    assert archive.calls("open") == []
    assert {aq.file_encoding(p) for p in [output, *days]} == {"none"}
    assert {p: content_sha256(p) for p in [output, *days]} == content
