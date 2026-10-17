# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""GLORYS12V1: decimated volume, vertical connectivity, tracking and weighted genealogy."""

from pathlib import Path

from _data import load_glorys
from _report import (
    arguments,
    export_products,
    lineage_catalog_status,
    spatial_summary,
    temporal_summary,
    write_products_manifest,
    write_report,
)
from plotting import (
    frequency_figure,
    genealogy_figure,
    largest_event,
    metrics_figure,
    sequence_figure,
    vertical_figure,
)

import xhw


def run(data_dir: Path, output_dir: Path, products_dir: Path | None = None) -> dict:
    # %% Load only the registered retained cells and levels, after verifying every file
    ds, grid, baseline, definition = load_glorys(data_dir)
    grid.check(ds)
    print("GLORYS12V1 reanalysis 1993–2023; reference 1993–2022; stride 8; 12 retained levels.")

    # %% Marine heatwaves at every retained ocean voxel; no interpolation in this case
    fitted = xhw.fit_baseline(
        ds, variable="thetao", grid=grid, baseline=baseline, definition=definition
    )
    fitted.check_invariants()
    mask = xhw.detect(
        ds, variable="thetao", grid=grid, definition=definition, baseline_result=fitted
    )
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    figure = frequency_figure({"MHW: first retained depth": mask}, output_dir)

    # %% Compare index and physical vertical neighbourhoods without changing the evidence
    connectivity = {}
    for gap in (0.0, 5.0, 20.0):
        patches = xhw.build_objects(
            mask,
            recipe=xhw.SpatialRecipe(
                input_layer="event_membership",
                steps=(
                    xhw.RecipeStep.segment(
                        footprint=xhw.Footprint.connected(1, vertical_gap_m=gap),
                        min_area=xhw.MinArea.none(),
                    ),
                ),
            ),
        )
        patches.check_invariants()
        patches.check_against(mask)
        connectivity[f"{gap:g} m"] = int(patches.dataset.sizes["object"])

    # %% Track 6-neighbour volume patches with positive shared volume and overlap >= 0.1
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(
                footprint=xhw.Footprint.connected(1), min_area=xhw.MinArea.none()
            ),
            xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe, analysis_mode="retrospective")
    objects.check_invariants()
    objects.check_against(mask)
    figure = sequence_figure(objects, ["2023-05-01", "2023-05-15", "2023-06-01"], output_dir)
    figure = vertical_figure(ds, mask, objects, output_dir, date="2023-05-15")

    # %% Weight sizes and intensities by approximate area times retained-level thickness
    graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
    graph.check_invariants()
    graph.check_against(objects)
    export_status = lineage_catalog_status(graph, mask)
    event = largest_event(graph)
    figure = genealogy_figure(graph, event, output_dir)
    metrics_figure(graph, objects, event, output_dir)
    del figure
    weights = ds.cell_area * ds.dz
    total = float(weights.where(ds.domain_mask, 0).sum())
    occupied = mask.dataset.event_membership.sel(time="2023-05-15")
    metrics = {
        "temporal": temporal_summary(mask, catalog),
        "spatial": spatial_summary(objects, graph),
        "patches_by_vertical_gap": connectivity,
        "domain_volume_km3": total / 1e9,
        "volume_fraction_in_event_2023_05_15": float(weights.where(occupied, 0).sum()) / total,
        "lineage_catalog_export": export_status,
    }

    from exploration import vertical_gif, volume_html, volume_preview

    volume_preview(mask, objects, graph, output_dir)
    volume_html(mask, objects, graph, output_dir)
    vertical_gif(mask, objects, graph, output_dir)

    # %% Record approximation choices and the outstanding scientific review
    if products_dir is not None:
        entries = export_products(
            products_dir,
            "mhw",
            baseline=fitted,
            mask=mask,
            catalog=catalog,
            objects=objects,
            graph=graph,
        )
        write_products_manifest(products_dir, "glorys_volume", entries)
    report = write_report(
        output_dir,
        "glorys",
        metrics,
        grid=grid,
        baseline=baseline,
        definition=definition,
        recipe=recipe,
        graph=graph,
        review="executed; real volumetric and lineage summaries pending scientific review",
        extra={
            "selection": "Peru 1993–2023; horizontal stride 8; original depth indices 0,2,…,22",
            "layer_thickness_m": ds.dz.values.tolist(),
            "volume_geometry": "spherical 2/3-degree cell areas; midpoint depth interfaces",
            "decimation_limit": "coarse retained-cell connectivity cannot resolve omitted features",
            "domain_volume_units": "km3",
        },
    )
    return report


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir, args.products_dir)
