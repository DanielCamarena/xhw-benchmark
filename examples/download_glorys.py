# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Explicit, optional GLORYS acquisition; default prints requests without networking.

Install copernicusmarine separately and authenticate in your home directory.
New files are not automatically equivalent to the registered tutorial inputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from content_hash import content_sha256, file_sha256


def request(year: int, output_dir: Path) -> dict:
    return dict(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_latitude=-12.0,
        maximum_latitude=-4.0,
        minimum_longitude=-82.0,
        maximum_longitude=-77.0,
        minimum_depth=0.0,
        maximum_depth=500.0,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T00:00:00",
        output_directory=str(output_dir),
        output_filename=f"thetao_{year}.nc",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--years", default="1993-2023")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        first, last = map(int, args.years.split("-"))
        if not 1993 <= first <= last:
            raise ValueError
    except ValueError:
        parser.error("--years must be a valid inclusive range, for example 1993-2023")
    target = args.data_dir / "glorys12v1_peru"
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
        config = request(year, target)
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
    if entries:
        (target / "download_manifest.json").write_text(
            json.dumps(
                {
                    "accessed": datetime.now(timezone.utc).isoformat(),
                    "client_version": version("copernicusmarine"),
                    "files": entries,
                    "review_status": "new retrieval; not the registered reviewed input",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
