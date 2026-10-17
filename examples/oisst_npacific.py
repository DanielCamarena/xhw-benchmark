# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""OISST over the North Pacific: 35 years of MHW and MCS, and the 2012-2018 Blob as objects.

The box is 25N-55N, 160E-130W (120 x 280 cells, 0.25 degree, longitudes kept in 0-360 because it
crosses the dateline; 33 540 ocean cells). Detection covers all of 1991-2025 with the fixed
1991-2020 climatology, summarised from the masks. The per-cell event catalog, objects, tracking
and genealogy are built for 2012-2018, the window of the 2013-2016 "Blob", with the same
baseline. The 2012-2018 catalog/object window is retained as the recorded scientific
selection and to bound memory. Earlier timing estimates described an older build;
they are not performance guarantees for the corrected implementation. A full dynamic
baseline is not run: its four time-by-cell float64 fields alone need about 14 GB.
Summaries are pending review.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from _data import (
    add_area,
    checked_file,
    fixed_baseline,
    marine_definition,
    record,
    surface_grid,
)
from _report import (
    arguments,
    export_products,
    lineage_catalog_status,
    spatial_summary,
    temporal_summary,
    write_products_manifest,
    write_report,
)
from npacific_blocks import BLOCK_ROWS, ocean_domain, temporal_blocks
from plotting import (
    genealogy_figure,
    largest_event,
    map_panel,
    metrics_figure,
    pyplot,
    save_figure,
    sequence_figure,
    series_figure,
)

import xhw


def load_oisst_npacific(data_dir: Path):
    """OISST over the North Pacific, 25N-55N and 160E-130W (0-360 longitudes), 0.25 degree.

    Personal case, not versioned: the file is either the original download or its lossless
    recompression (`compressed` in sources/oisst_npacific.json), accepted by stored content.
    """
    rec = record("oisst_npacific")
    original, packed = rec["file"], rec["compressed"]
    if (data_dir / original).is_file():
        path = checked_file(data_dir, original, rec["sha256"], rec["content_sha256"])
    else:
        path = checked_file(data_dir, packed["file"], packed["sha256"], packed["content_sha256"])
    # Keep temperature on the backend until each bounded latitude block is selected.
    ds = add_area(xr.open_dataset(path, engine="h5netcdf"), "lat", "lon", 0.25)
    ds["domain_mask"] = ocean_domain(ds)
    source = xhw.SourceRef(
        dataset_id=rec["dataset_id"],
        variable="sst",
        accessed=rec["accessed"],
        product_status="final",
        version=rec["version"],
        checksum="sha256:" + rec["sha256"],
    )
    grid = surface_grid(product="NOAA OISST v2.1 AVHRR-only", resolution="0.25 degree")
    return ds, grid, fixed_baseline(source), marine_definition()


WINDOW = ("2012-01-01", "2018-12-31")
# Absolute, because the default 5% of this 33 000-cell domain would discard 99% of the patches.
SPATIAL = xhw.SpatialRecipe(
    input_layer="event_membership",
    steps=(
        xhw.RecipeStep.segment(
            footprint=xhw.Footprint.connected(1),
            min_area=xhw.MinArea(kind="absolute_km2", value=10_000.0),
        ),
        xhw.RecipeStep.track(min_overlap=0.0, max_gap_days=0, min_duration_days=5),
    ),
)


def frequency_figure(frequencies: dict, output_dir: Path):
    plt = pyplot()
    fig, axes = plt.subplots(
        1, len(frequencies), figsize=(7 * len(frequencies), 3.8), layout="constrained"
    )
    for ax, (name, freq) in zip(np.atleast_1d(axes), frequencies.items(), strict=True):
        map_panel(ax, freq, title=name, cbar_kwargs={"label": "% of valid days"})
        ax.set_xlabel("Longitude (degrees east, 0-360)")
    return save_figure(fig, output_dir, "frequency")


def run(data_dir: Path, output_dir: Path, products_dir: Path | None = None) -> dict:
    # %% Open the registered 1991-2025 subset (or its compressed copy) and declare its grid
    raw, grid, baseline, definition = load_oisst_npacific(data_dir)
    grid.check(raw)
    ocean_cells = int(raw.domain_mask.sum())
    print(f"OISST North Pacific; {ocean_cells} ocean cells; reference 1991–2020.", flush=True)
    # Temporal rules use each cell's complete 35-year series in bounded latitude blocks.
    window, metrics, frequencies, fits, entries = temporal_blocks(
        raw, grid, baseline, definition, window=WINDOW, products_dir=products_dir
    )
    del raw
    del fits["MCS"]
    figure = frequency_figure(frequencies, output_dir)
    del figure

    # Assemble every latitude before the global spatial stages; never track per block.
    mask = xhw.detect(
        window, variable="sst", grid=grid, definition=definition, baseline_result=fits["MHW"]
    )
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    metrics["window_MHW"] = temporal_summary(mask, catalog)
    series_figure(
        window,
        mask,
        variable="sst",
        selection={"lat": 79, "lon": 220},
        period=("2014-06-01", "2015-03-01"),
        output_dir=output_dir,
    )
    objects = xhw.build_objects(mask, recipe=SPATIAL, analysis_mode="retrospective")
    objects.check_invariants()
    objects.check_against(mask)
    sequence_figure(objects, ["2014-01-15", "2014-09-15", "2015-01-15"], output_dir)
    graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
    graph.check_invariants()
    graph.check_against(objects)
    metrics["lineage_catalog_export"] = lineage_catalog_status(graph, mask)
    event = largest_event(graph)
    genealogy_figure(graph, event, output_dir)
    metrics_figure(graph, objects, event, output_dir)
    metrics["spatial"] = spatial_summary(objects, graph)
    metrics["largest_size_time_event"] = int(event)
    if products_dir is not None:
        entries |= export_products(
            products_dir,
            "blob_2012_2018",
            mask=mask,
            catalog=catalog,
            objects=objects,
            graph=graph,
        )
        write_products_manifest(products_dir, "oisst_npacific", entries)

    # %% Record the run; the review state stays explicit
    return write_report(
        output_dir,
        "oisst_npacific",
        metrics,
        grid=grid,
        baseline=baseline,
        definition=definition,
        recipe=SPATIAL,
        graph=graph,
        review="executed; summaries pending scientific review",
        extra={
            "selection": "North Pacific: lat 25 to 55; lon 160 to 230 (0-360); 1991–2025",
            "object_window": list(WINDOW),
            "min_area": "absolute 10 000 km2 (the default 5% removes 99% of patches)",
            "ocean_cells": ocean_cells,
            "temporal_block_rows": BLOCK_ROWS,
            "temporal_time_selection": "complete 1991–2025 series per cell; spatial stages global",
        },
    )


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir, args.products_dir)
