# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Check data files against the registered manifests, by bytes and by content.

    python examples/verify_data.py --data-dir data          # every registered file found there
    python examples/verify_data.py some_file.nc ...         # print both hashes of given files

A file can match in two ways. ``identical bytes``: its SHA-256 is the registered one.
``same content``: the bytes differ (for example the file was recompressed) but the stored
values and decoding attributes hash to the registered ``content_sha256``, so the
information is the same. See content_hash.py for what the content hash covers. Anything
else is a ``MISMATCH`` and the exit status is 1. Missing files are only reported.

Manifests read: examples/sources/oisst.json, era5.json, glorys.json (registered inputs,
with optional ``content_sha256``) and test_datasets.json (derived compressed test sets).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from content_hash import content_sha256, file_sha256

MANIFESTS = Path(__file__).parent / "sources"


def registered_files() -> list[dict]:
    """Every file named in the manifests: relative path, file hash, optional content hash."""
    entries = []
    for name in ("oisst", "oisst_equatorial", "era5"):
        path = MANIFESTS / f"{name}.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "manifest": name,
                "file": rec["file"],
                "sha256": rec["sha256"],
                "content_sha256": rec.get("content_sha256"),
            }
        )
        for fname, digest in rec.get("hourly_files_sha256", {}).items():
            entries.append(
                {
                    "manifest": name,
                    "file": f"era5_peru_hourly/{fname}",
                    "sha256": digest,
                    "content_sha256": None,
                }
            )
    glorys = MANIFESTS / "glorys.json"
    if glorys.is_file():
        for fname, entry in json.loads(glorys.read_text(encoding="utf-8"))["files"].items():
            entries.append(
                {
                    "manifest": "glorys",
                    "file": f"glorys12v1_peru/{fname}",
                    "sha256": entry["sha256"],
                    "content_sha256": entry.get("content_sha256"),
                }
            )
    tests = MANIFESTS / "test_datasets.json"
    if tests.is_file():
        for key, entry in json.loads(tests.read_text(encoding="utf-8"))["datasets"].items():
            entries.append(
                {
                    "manifest": f"test_datasets:{key}",
                    "file": entry["file"],
                    "sha256": entry["sha256"],
                    "content_sha256": entry["content_sha256"],
                }
            )
    return entries


def judge(path: Path, expected_file: str | None, expected_content: str | None) -> str:
    if expected_file is not None and file_sha256(path) == expected_file:
        return "identical bytes"
    if expected_content is not None and content_sha256(path) == expected_content:
        return "same content"
    return "MISMATCH"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path, help="print the hashes of these files")
    parser.add_argument("--data-dir", type=Path, help="verify registered files found here")
    parser.add_argument("--verbose", action="store_true", help="also list absent files")
    args = parser.parse_args()
    if not args.files and args.data_dir is None:
        parser.error("give --data-dir or one or more files")
    failed = 0
    registered = registered_files()
    for path in args.files:
        file_hash, content = file_sha256(path), content_sha256(path)
        match = [
            e["manifest"]
            for e in registered
            if e["sha256"] == file_hash or e.get("content_sha256") == content
        ]
        print(f"{path}\n  sha256          {file_hash}\n  content_sha256  {content}")
        print(f"  registered as   {', '.join(sorted(set(match))) or '(no manifest entry)'}")
    if args.data_dir is not None:
        absent = []
        for entry in registered:
            path = args.data_dir / entry["file"]
            if not path.is_file():
                absent.append(entry["file"])
                continue
            verdict = judge(path, entry["sha256"], entry["content_sha256"])
            failed += verdict == "MISMATCH"
            print(f"{verdict:16s}  {entry['file']}  [{entry['manifest']}]")
        if args.verbose:
            for name in absent:
                print(f"skipped (absent)  {name}")
        elif absent:
            print(
                f"{len(absent)} registered files are not in {args.data_dir} (use --verbose to list)"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
