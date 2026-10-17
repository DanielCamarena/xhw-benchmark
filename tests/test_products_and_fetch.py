# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Products round-trip through compressed NetCDF; fetch_data keeps only verified files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import xhw

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


@pytest.fixture(scope="module")
def products():
    from _data import synthetic_inputs

    raw, grid, baseline, definition = synthetic_inputs()
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )
    fitted = xhw.fit_baseline(
        prepared, variable="sst", grid=grid, baseline=baseline, definition=definition
    )
    mask = xhw.detect(
        prepared, variable="sst", grid=grid, definition=definition, baseline_result=fitted
    )
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(min_area=xhw.MinArea.none()),
            xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe)
    graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
    return fitted, mask, catalog, objects, graph


def test_exported_products_reload_with_their_own_constructors(products, tmp_path):
    import xarray as xr
    from _report import export_products, write_products_manifest

    fitted, mask, catalog, objects, graph = products
    entries = export_products(
        tmp_path, "case", baseline=fitted, mask=mask, catalog=catalog, objects=objects, graph=graph
    )
    write_products_manifest(tmp_path, "case", entries)
    assert sorted(entries) == [
        f"case_{k}.nc" for k in ("baseline", "catalog", "lineage", "mask", "objects")
    ]
    manifest = json.loads((tmp_path / "products_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_bytes"] == sum(e["bytes"] for e in entries.values())

    def read(name):
        with xr.open_dataset(tmp_path / name, engine="h5netcdf") as opened:
            return opened.load()

    assert xhw.EventCatalog.from_dataset(read("case_catalog.nc")).dataset.sizes["event"] == 20
    reloaded = xhw.LineageGraph.from_dataset(read("case_lineage.nc"))
    assert int(reloaded.dataset.is_split.sum()) == 1 and int(reloaded.dataset.is_merge.sum()) == 1
    assert reloaded.to_dataset().attrs["xhw_schema_version"] == "0.1"
    for name in ("object_intensity_valid_cells", "object_intensity_valid_measure"):
        xr.testing.assert_identical(reloaded.dataset[name], graph.dataset[name])
    reloaded_objects = xhw.ObjectField.from_dataset(read("case_objects.nc"))
    assert int(reloaded_objects.dataset.label.max()) == int(objects.dataset.label.max())
    layers = read("case_mask.nc")
    # only the layers, never the three large float arrays; the counts of the live mask survive
    assert not {"temperature", "reference", "threshold"} & set(layers.variables)
    assert int(layers.event_membership.sum()) == int(mask.dataset.event_membership.sum()) == 216


def run_fetch(data_dir: Path, source: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "fetch_data.py"),
            "--data-dir",
            str(data_dir),
            "--from-dir",
            str(source),
            "--only",
            "oisst",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_fetch_data_keeps_only_files_that_match_the_manifest(tmp_path):
    entry = json.loads((EXAMPLES / "sources" / "test_datasets.json").read_text("utf-8"))[
        "datasets"
    ]["oisst"]
    source, target = tmp_path / "source", tmp_path / "data"
    source.mkdir()
    (source / entry["file"]).write_bytes(b"not the registered dataset")
    result = run_fetch(target, source)
    assert result.returncode == 1 and "REJECTED" in result.stdout
    assert not (target / entry["file"]).exists()  # a rejected file is not left behind
    assert not list(target.glob("*.part"))
