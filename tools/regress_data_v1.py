# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline regression of the registered inputs (data-v1) from the retained originals.

    python tools/regress_data_v1.py --originals ../xhw/data --work-dir <scratch folder>

Revision 0, 5.1 (phase F2.9): the new code must rebuild every registered input from the
historical originals kept on disk with the same ``content_sha256``. Nothing is downloaded.

* Integrity: every retained original has the file ``sha256`` of its registration.
* ERA5 daily: the CDS is replayed from the registered hourly year files
  (``era5_peru_hourly/*.nc``). ``download_era5.py`` runs unchanged (plan, validation,
  ledger, split per variable and month, then ``--daily``) and its daily file must have
  the ``content_sha256`` of ``era5_peru_daily_1991_2025.nc``. The year files kept no
  ``expver``, so the replay runs with ``--accept-unknown-expver-before`` (the
  registration says final data).
* ``make_test_datasets.py`` (recompression, hourly sample, GLORYS decimation and the
  OSTIA and GLORYS joins in time): every dataset it builds must have the registered
  ``content_sha256``.
* OISST North Pacific: the retained file has its registered ``content_sha256``.

Not covered, and reported as pending: the OISST, GLORYS and OSTIA downloaders themselves,
whose originals are the answers of the services (their regional files are what those
downloaders write, without a transformation that could be replayed here). They are
checked by the short real downloads of phase F3.

The work folder receives about 3 GB (ERA5 monthly files and the rebuilt datasets);
put it outside Dropbox. The report is ``<work-dir>/data_v1_regression.json``; the exit
code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import download_era5 as era5  # noqa: E402
import make_test_datasets as mtd  # noqa: E402
from content_hash import content_sha256, file_sha256, registered  # noqa: E402

ERA5_FOLDER = "era5_peru_hourly"
ERA5_START, ERA5_END = "1991-01-01", "2025-12-31"
EXPVER_BEFORE = "2026-01-02"  # the boundary hours of 2026-01-01 00-06 UTC are registered too
EXPVER_REASON = (
    "replay of the registered hourly files (accessed 2026-10-05, registered as final data); "
    "the consolidated year files did not keep expver"
)


class ReplayCDS:
    """Answers CDS request bodies from the registered hourly year files (no network).

    The answer holds exactly the hours, variables and box of the body, with the stored
    values and attributes of the year files, laid out as the CDS does (``valid_time``).
    A body asking for an hour or a node the year files lack is an error, never a guess.
    """

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._lock = threading.Lock()  # HDF5 reads are not thread-safe
        self._open: dict[str, xr.Dataset] = {}
        self.calls = 0

    def _dataset(self, name: str) -> xr.Dataset:
        if name not in self._open:
            path = self.folder / name
            if not path.is_file():
                raise FileNotFoundError(f"replay: {path} is not among the originals")
            self._open[name] = xr.open_dataset(path, engine="h5netcdf")
        return self._open[name]

    def close(self) -> None:
        for ds in self._open.values():
            ds.close()
        self._open.clear()

    @staticmethod
    def stamps(body: dict) -> pd.DatetimeIndex:
        """Every hour a body asks for (invalid dates such as 31 April are skipped, as the CDS)."""
        out = []
        for y in body["year"]:
            for m in body["month"]:
                days = pd.Timestamp(f"{y}-{m}-01").days_in_month
                for d in body["day"]:
                    if int(d) <= days:
                        out += [pd.Timestamp(f"{y}-{m}-{d} {t}") for t in body["time"]]
        return pd.DatetimeIndex(sorted(out))

    def answer(self, body: dict) -> xr.Dataset:
        names = {spec.cds: name for name, spec in era5.CATALOGUE.items()}
        variables = [names[c] for c in body["variable"]]
        north, west, south, east = body["area"]
        invariant = all(era5.CATALOGUE[v].kind == "invariant" for v in variables)
        with self._lock:
            if invariant:
                ds = self._dataset(f"{variables[0]}.nc")[variables].load()
            else:
                times = self.stamps(body)
                pieces = []
                for year in sorted(set(times.year)):
                    source = self._dataset(f"{year}.nc")
                    wanted = times[times.year == year]
                    missing = wanted.difference(pd.DatetimeIndex(source["valid_time"].values))
                    if len(missing):
                        raise RuntimeError(
                            f"replay: {len(missing)} hours missing from {year}.nc "
                            f"(first {missing[0]})"
                        )
                    pieces.append(source[variables].sel(valid_time=wanted).load())
                ds = xr.concat(pieces, dim="valid_time") if len(pieces) > 1 else pieces[0]
        ds = ds.sel(latitude=slice(north, south), longitude=slice(west, east))
        lat, lon = ds["latitude"].values, ds["longitude"].values
        if lat.size == 0 or lon.size == 0 or not (
            np.isclose(lat.max(), north) and np.isclose(lat.min(), south)
            and np.isclose(lon.min(), west) and np.isclose(lon.max(), east)
        ):  # fmt: skip
            raise RuntimeError(f"replay: the originals do not hold the area {body['area']}")
        for var in ds.variables.values():
            var.encoding = {}  # write the decoded values as they are, unpacked
        return ds

    def retrieve(self, dataset, body, target):
        if dataset != era5.DATASET:
            raise RuntimeError(f"replay: only {era5.DATASET}, not {dataset}")
        self.calls += 1
        self.answer(body).to_netcdf(target, engine="h5netcdf")


def era5_argv(work: Path, start: str, end: str, extra=()) -> list[str]:
    return [
        "--data-dir", str(work), "--region", "peru", "--start", start, "--end", end,
        "--accept-unknown-expver-before", EXPVER_BEFORE,
        "--expver-justification", EXPVER_REASON, *extra,
    ]  # fmt: skip


def replay_era5(
    originals: Path,
    work: Path,
    start: str = ERA5_START,
    end: str = ERA5_END,
    workers: int = 4,
    extra=(),
) -> Path:
    """Run ``download_era5.py --download`` then ``--daily`` against the replayed CDS.

    ``work`` must be new or empty: the ledger reuses validated pieces and an unchanged
    product, so a previous run would be compared instead of the current code.
    """
    if work.exists() and any(work.iterdir()):
        raise SystemExit(
            f"{work} holds a previous replay; the regression needs a new folder (a reused "
            "ledger would compare the old product, not the current code)"
        )
    client = ReplayCDS(originals / ERA5_FOLDER)
    try:
        argv = era5_argv(work, start, end, extra)
        code = era5.main([*argv, "--download", "--workers", str(workers)], lambda: client)
        if code != 0:
            raise RuntimeError(f"download_era5.py --download ended with {code}")
        if era5.main([*argv, "--daily"]) != 0:
            raise RuntimeError("download_era5.py --daily failed")
    finally:
        client.close()
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    stamp = f"{a:%Y%m%d}_{b:%Y%m%d}"
    if (a.month, a.day, b.month, b.day) == (1, 1, 12, 31):
        stamp = f"{a.year}_{b.year}"
    return work / f"era5_peru_daily_{stamp}.nc"


def check(name: str, expected: str | None, got: str | None, **info) -> dict:
    status = "pass" if expected is not None and expected == got else "fail"
    return {"check": name, "status": status, "expected": expected, "got": got, **info}


def integrity(originals: Path) -> list[dict]:
    """File ``sha256`` of every retained original against its registration."""
    out = []
    era = registered("era5")
    for name, digest in era["hourly_files_sha256"].items():
        path = originals / ERA5_FOLDER / name
        got = file_sha256(path) if path.is_file() else None
        out.append(check(f"original {ERA5_FOLDER}/{name}", digest, got))
    sets = registered("test_datasets")["datasets"]
    for key, folder in (("glorys_decimated", "glorys12v1_peru"), ("ostia", "ostia_peru")):
        for name, rec in sets[key]["derived_from"]["files"].items():
            path = originals / folder / name
            got = file_sha256(path) if path.is_file() else None
            out.append(check(f"original {folder}/{name}", rec["sha256"], got))
    for key in ("oisst", "oisst_equatorial", "era5_daily"):
        rec = sets[key]["derived_from"]
        path = originals / rec["file"]
        got = file_sha256(path) if path.is_file() else None
        out.append(check(f"original {rec['file']}", rec["sha256"], got))
    npac = registered("oisst_npacific")
    path = originals / npac["file"]
    got = file_sha256(path) if path.is_file() else None
    out.append(check(f"original {npac['file']}", npac["sha256"], got))
    return out


def run(originals: Path, work: Path, only=("integrity", "era5", "datasets", "npacific"), workers=4):
    work.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    sets = registered("test_datasets")["datasets"]

    def timed(label, fn):
        t0 = time.monotonic()
        print(f"--- {label}", flush=True)
        out = fn()
        print(f"--- {label}: {time.monotonic() - t0:.0f} s", flush=True)
        return out

    if "integrity" in only:
        results += timed("integrity of the originals", lambda: integrity(originals))
    if "era5" in only:
        product = timed(
            "ERA5 replay", lambda: replay_era5(originals, work / "era5", workers=workers)
        )
        results.append(
            check(
                "era5_daily rebuilt by download_era5.py from the hourly originals",
                registered("era5")["content_sha256"],
                content_sha256(product),
                product=str(product),
            )  # fmt: skip
        )
    if "datasets" in only:
        built = timed(
            "make_test_datasets.py",
            lambda: mtd.build(originals, work / "test-data", only=mtd.NAMES),
        )
        for key in mtd.NAMES:
            if key not in built:
                results.append({"check": f"dataset {key}", "status": "fail", "error": "not built"})
                continue
            results.append(
                check(
                    f"dataset {key} ({sets[key]['operation']})",
                    sets[key]["content_sha256"],
                    built[key]["content_sha256"],
                    file=built[key]["file"],
                    same_file_sha256=built[key]["sha256"] == sets[key]["sha256"],
                )  # fmt: skip
            )
    if "npacific" in only:
        npac = registered("oisst_npacific")
        path = originals / npac["file"]
        results.append(
            check(
                f"content of {npac['file']}",
                npac["content_sha256"],
                timed("OISST npacific content", lambda: content_sha256(path))
                if path.is_file()
                else None,
            )  # fmt: skip
        )
    pending = [
        {
            "check": f"download_{name}.py against its service",
            "status": "pending",
            "reason": "the originals are the service's answers; phase F3 (real downloads)",
        }
        for name in ("oisst", "glorys", "ostia", "sst_cci")
    ]
    failed = [r for r in results if r["status"] == "fail"]
    report = {
        "format": "xhw-benchmark-data-v1-regression-1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "originals": str(originals),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "results": results + pending,
    }
    path = work / "data_v1_regression.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in results + pending:
        print(f"{r['status']:8s} {r['check']}")
    print(f"{report['passed']} passed, {report['failed']} failed; report {path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--originals", type=Path, required=True, help="e.g. ../xhw/data")
    parser.add_argument("--work-dir", type=Path, required=True, help="scratch, outside Dropbox")
    parser.add_argument(
        "--only", nargs="+", default=["integrity", "era5", "datasets", "npacific"],
        choices=["integrity", "era5", "datasets", "npacific"],
    )  # fmt: skip
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = run(args.originals, args.work_dir, tuple(args.only), args.workers)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
