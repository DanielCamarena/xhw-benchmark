# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional OSTIA acquisition; default prints requests without networking.

Retrieves the UK Met Office OSTIA daily sea surface temperature analysis (0.05 degree,
reprocessed series from 1981-10-01 to 2026-03-31, which covers 1991-2025) for the same
12S-4S x 82W-77W box as the OISST tutorial, so that the effect of resolution (0.05 versus
0.25 degree) can be studied. Same tool, account and behaviour as download_glorys.py:
install copernicusmarine separately and run ``copernicusmarine login`` once.

    python examples/download_ostia.py --data-dir data                  # print the requests
    python examples/download_ostia.py --data-dir data --dry-run        # ask the service to check
    python examples/download_ostia.py --data-dir data --download       # one NetCDF per year

``analysed_sst`` is in kelvin: xHW does not convert units, so declare it (or subtract
273.15 yourself) before building specs. Files are never overwritten and a
``download_manifest.json`` records the request, client version and hashes. A new retrieval
is not automatically the same bytes as another one: compare ``content_sha256`` with
``verify_data.py``. Product page: https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from content_hash import content_sha256, file_sha256

DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
DATASET_VERSION = "202003"  # version label in the catalogue on 2026-10-11
FIRST_YEAR = 1981  # reprocessed series starts on 1981-10-01


def request(year: int, output_dir: Path, dataset_version: str) -> dict:
    return dict(
        dataset_id=DATASET_ID,
        dataset_version=dataset_version,
        variables=["analysed_sst"],
        minimum_latitude=-12.0,
        maximum_latitude=-4.0,
        minimum_longitude=-82.0,
        maximum_longitude=-77.0,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T00:00:00",
        output_directory=str(output_dir),
        output_filename=f"analysed_sst_{year}.nc",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--years", default="1991-2025")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        first, last = map(int, args.years.split("-"))
        if not FIRST_YEAR <= first <= last <= 2026:
            raise ValueError
    except ValueError:
        parser.error("--years must be a valid inclusive range, for example 1991-2025")
    target = args.data_dir / "ostia_peru"
    client = None
    if args.download or args.dry_run:
        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "Install copernicusmarine in an acquisition environment; "
                "see the README. It is not an xHW runtime dependency."
            ) from exc
        client = copernicusmarine
    entries = {}
    for year in range(first, last + 1):
        config = request(year, target, args.dataset_version)
        path = target / config["output_filename"]
        if path.exists() and args.download:
            raise FileExistsError(f"{path.name} already exists; use a separate data directory.")
        print(json.dumps(config, indent=2))
        if client is not None:
            if args.download:
                target.mkdir(parents=True, exist_ok=True)
            client.subset(**config, dry_run=args.dry_run)
        if args.download:
            entries[path.name] = {
                "sha256": file_sha256(path),
                "content_sha256": content_sha256(path),
                "bytes": path.stat().st_size,
                "request": config,
            }
    if not (args.download or args.dry_run):
        print("Print only. Add --dry-run to check the requests or --download to retrieve them.")
    if entries:
        (target / "download_manifest.json").write_text(
            json.dumps(
                {
                    "accessed": datetime.now(timezone.utc).isoformat(),
                    "client_version": version("copernicusmarine"),
                    "units": "kelvin",
                    "files": entries,
                    "review_status": "new retrieval; not a registered reviewed input",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
