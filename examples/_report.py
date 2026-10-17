# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Portable run reports for the public tutorials; figures live in plotting.py.

Copied from the examples folder of the xHW repository: keep both copies in step by hand.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from content_hash import content_sha256, file_sha256

# Re-exported so tutorial code can keep importing figures from one place.
from plotting import (  # noqa: F401
    connectivity_figure,
    frequency_figure,
    genealogy_figure,
    largest_event,
    map_panel,
    mask_figure,
    metrics_figure,
    pyplot,
    save_figure,
    sequence_figure,
    series_figure,
    vertical_figure,
)
from provenance import code_fingerprint

import xhw


def arguments(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=None,
        help="also write the processed products (compressed NetCDF) and their manifest here",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs") / parser.prog.removesuffix(".py")
    )
    return parser.parse_args()


def temporal_summary(mask, catalog) -> dict:
    ds = mask.dataset
    return {
        "events": int(catalog.dataset.sizes["event"]),
        "exceedance_cell_days": int(ds["exceedance"].sum()),
        "event_cell_days": int(ds["event_membership"].sum()),
        "joined_cell_days": int(ds["joined_gap"].sum()),
        "valid_cell_days": int(ds["valid"].sum()),
        "filled_cell_days": int(ds["filled"].sum()) if "filled" in ds else 0,
        "max_duration_days": int(catalog.dataset["duration_days"].max())
        if catalog.dataset.sizes["event"]
        else 0,
    }


def mask_summary(mask) -> dict:
    """The cell-day counts of `temporal_summary`, from the mask alone (no catalog needed)."""
    ds = mask.dataset
    return {
        "exceedance_cell_days": int(ds["exceedance"].sum()),
        "event_cell_days": int(ds["event_membership"].sum()),
        "joined_cell_days": int(ds["joined_gap"].sum()),
        "valid_cell_days": int(ds["valid"].sum()),
        "filled_cell_days": int(ds["filled"].sum()) if "filled" in ds else 0,
    }


def spatial_summary(objects, graph) -> dict:
    ds, g = objects.dataset, graph.dataset
    return {
        "patches": int(ds.sizes["object"]),
        "tracks": int(ds.sizes.get("track", 0)),
        "branches": int(g.sizes["branch"]),
        "spacetime_events": int(g.sizes["spacetime_event"]),
        "splits": int(g["is_split"].sum()),
        "merges": int(g["is_merge"].sum()),
        "max_size": float(ds["object_area"].max()),
        "size_units": ds["object_area"].attrs["units"],
        "max_size_time": float(g["event_area_days"].max()),
        "max_duration_days": int(g["event_duration_days"].max()),
    }


def lineage_catalog_status(graph, mask) -> dict:
    """Inspect catalog qualification; never suppress other validation errors.

    Tracking's duration filter suppresses track IDs, not the daily patch labels.
    A graph over those labels can contain short components. Its EventCatalog export
    declares all rows qualified, and therefore rejects such components.
    """
    short = int((graph.dataset.event_duration_days < mask.definition.min_duration_days).sum())
    if short:
        try:
            graph.to_event_catalog(mask)
        except xhw.InvariantError as exc:
            if str(exc) != "qualified events shorter than min_duration_days":
                raise
            print(
                f"Catalog export constraint: {short} graph components are shorter than "
                f"{mask.definition.min_duration_days} days. Use graph.dataset for these components."
            )
            return {
                "status": "not exportable as qualified EventCatalog",
                "short_components": short,
                "reason": str(exc),
            }
        raise AssertionError("Expected the documented catalog qualification constraint")
    catalog = graph.to_event_catalog(mask)
    graph.check_catalog(catalog)
    return {"status": "exported and checked", "short_components": 0}


def write_report(
    output_dir: Path,
    case: str,
    metrics: dict,
    *,
    grid,
    baseline,
    definition,
    recipe=None,
    graph=None,
    review: str,
    extra: dict | None = None,
) -> dict:
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "xhw",
            "numpy",
            "xarray",
            "scipy",
            "matplotlib",
            "h5netcdf",
            "nbformat",
            "nbclient",
        )
    }
    report = {
        "case": case,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "review_status": review,
        "platform": platform.system(),
        "python": platform.python_version(),
        "versions": versions,
        "code": code_fingerprint(),
        "execution_options": {
            "requested_memory_mapping": os.environ.get("XHW_MAPPED_ARRAYS") == "1"
        },
        "source": baseline.source.to_dict(),
        "grid": grid.to_dict(),
        "baseline": baseline.to_dict(),
        "definition": definition.to_dict(),
        "recipe": recipe.to_dict() if recipe else None,
        "link_policy": graph.policy.to_dict() if graph else None,
        "extra": extra or {},
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Only portable scientific results are printed; no workstation paths or credentials.
    print(json.dumps(metrics, indent=2))
    return report


MASK_LAYERS = ("exceedance", "event_membership", "joined_gap", "valid", "invalid_reason", "filled")


def _write_compressed(ds, path: Path) -> dict:
    encoding = {
        v: {"zlib": True, "complevel": 4, "shuffle": True}
        for v in ds.data_vars
        if ds[v].size >= 1000
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    return {
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "content_sha256": content_sha256(path),
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "variables": list(ds.data_vars),
    }


def export_products(
    directory: Path, label: str, *, baseline=None, mask=None, catalog=None, objects=None, graph=None
) -> dict:
    """Write the processed products of one analysis as compressed NetCDF files.

    ``<label>_baseline.nc``, ``_mask.nc``, ``_catalog.nc``, ``_objects.nc`` and ``_lineage.nc``,
    for those products that are given. The mask file keeps only its boolean and flag layers:
    temperature, reference and threshold are float64 arrays that recompute from the input and
    the baseline, and they would make the file larger than every other product together. The
    others are the full ``to_dataset()`` of the product and can be reloaded with its
    ``from_dataset``. Returns the entries to merge into the case's products manifest.
    """
    entries = {}
    for kind, product in (
        ("baseline", baseline),
        ("mask", mask),
        ("catalog", catalog),
        ("objects", objects),
        ("lineage", graph),
    ):
        if product is None:
            continue
        ds = product.to_dataset()
        if kind == "mask":
            static = [v for v in ds.data_vars if "time" not in ds[v].dims]  # domain mask, areas
            ds = ds[[v for v in MASK_LAYERS if v in ds] + static]
        entries[f"{label}_{kind}.nc"] = _write_compressed(ds, directory / f"{label}_{kind}.nc")
    return entries


def write_products_manifest(directory: Path, case: str, entries: dict) -> None:
    """Record the products of a case: sizes, hashes and what each file holds."""
    manifest = {
        "case": case,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xhw_version": importlib.metadata.version("xhw"),
        "distribution": "NetCDF files are release assets (ignored by git); this manifest is in git",
        "mask_files": "boolean and flag layers only; temperature, reference and threshold omitted",
        "files": entries,
        "total_bytes": sum(e["bytes"] for e in entries.values()),
    }
    (directory / "products_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"products: {len(entries)} files, {manifest['total_bytes'] / 1e6:.1f} MB in {directory}")


def frequency(mask, valid=None) -> xr.DataArray:
    """Percentage of valid days inside an event, per cell (optionally on shared valid days)."""
    valid = mask.dataset["valid"] if valid is None else valid
    days = valid.sum("time")
    inside = (mask.dataset["event_membership"] & valid).sum("time")
    return 100 * inside / days.where(days > 0)
