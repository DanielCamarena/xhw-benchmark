# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Fetch the compressed test datasets from a GitHub release or a local folder, with verification.

    python examples/fetch_data.py --release-url https://github.com/<owner>/xhw-benchmark/releases/download/<tag>
    python examples/fetch_data.py --from-dir /path/to/folder/with/the/files
    python examples/fetch_data.py --only oisst ostia --from-dir ...

Each dataset listed in ``examples/sources/test_datasets.json`` is downloaded (or copied) to
``--data-dir`` (default ``data``) under its registered file name, and kept only if its bytes
match the registered ``sha256`` or its stored values match the registered ``content_sha256``.
An existing file is verified and left alone, never overwritten; a failed download is deleted.
The release URL can also come from the XHW_BENCHMARKS_RELEASE_URL environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from content_hash import content_sha256, file_sha256

MANIFEST = Path(__file__).parent / "sources" / "test_datasets.json"


def verified(path: Path, entry: dict) -> bool:
    if file_sha256(path) == entry["sha256"]:
        return True
    try:
        return content_sha256(path) == entry["content_sha256"]
    except (OSError, ValueError):  # not a readable NetCDF file at all
        return False


def fetch(entry: dict, target: Path, release_url: str | None, from_dir: Path | None) -> str:
    if target.exists():
        return "present, verified" if verified(target, entry) else "present but DIFFERENT"
    partial = target.with_suffix(".part")
    try:
        if from_dir is not None:
            shutil.copyfile(from_dir / entry["file"], partial)
        else:
            url = f"{release_url.rstrip('/')}/{entry['file']}"
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, length=1 << 20)
        if not verified(partial, entry):
            return "REJECTED: neither the bytes nor the content match the manifest"
        partial.replace(target)
        return "downloaded, verified" if release_url else "copied, verified"
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--release-url", default=os.environ.get("XHW_BENCHMARKS_RELEASE_URL"))
    parser.add_argument("--from-dir", type=Path)
    parser.add_argument("--only", nargs="+", help="dataset keys, e.g. oisst ostia (default: all)")
    args = parser.parse_args()
    if args.release_url is None and args.from_dir is None:
        parser.error("give --release-url (or XHW_BENCHMARKS_RELEASE_URL) or --from-dir")
    datasets = json.loads(MANIFEST.read_text(encoding="utf-8"))["datasets"]
    unknown = set(args.only or ()) - set(datasets)
    if unknown:
        parser.error(f"unknown dataset keys {sorted(unknown)}; known: {sorted(datasets)}")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    bad = 0
    for key, entry in datasets.items():
        if args.only and key not in args.only:
            continue
        try:
            status = fetch(entry, args.data_dir / entry["file"], args.release_url, args.from_dir)
        except OSError as exc:
            status = f"FAILED: {exc}"
        bad += status.startswith(("REJECTED", "FAILED")) or status.endswith("DIFFERENT")
        print(f"{status:34s} {entry['file']} ({entry['bytes'] / 1e6:.1f} MB)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
