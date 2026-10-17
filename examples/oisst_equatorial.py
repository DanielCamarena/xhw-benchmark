# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""OISST over the equatorial Pacific (Nino 3.4 box): MHW/MCS, fixed versus detrended, El Nino as
objects and genealogy.

The box is 5S-5N, 170W-120W (40 x 200 cells, 0.25 degree, all ocean) over 1991-2025 with the fixed
1991-2020 climatology. The strong El Ninos of 1997-98 and 2015-16 and the 2023 event stand out as
large, long-lived marine heatwave objects. Summaries are pending scientific review.
"""

from dataclasses import replace
from pathlib import Path

from _data import load_oisst_equatorial, marine_definition
from _report import (
    arguments,
    export_products,
    lineage_catalog_status,
    spatial_summary,
    temporal_summary,
    write_products_manifest,
    write_report,
)
from array_storage import (
    enabled,
    mapped_array,
    mapped_intensity,
    mapped_product,
    reuse_temperature,
)
from plotting import (
    frequency_figure,
    genealogy_figure,
    largest_event,
    metrics_figure,
    sequence_figure,
    series_figure,
)

import xhw


def run(data_dir: Path, output_dir: Path, products_dir: Path | None = None) -> dict:
    # %% Open the checked daily SST subset
    raw, grid, baseline, definition = load_oisst_equatorial(data_dir)
    grid.check(raw)
    storage = output_dir / "_arrays" if enabled() else None
    if storage is not None:
        raw["sst"] = mapped_array(raw.sst, storage, "raw-sst")
    print("NOAA OISST v2.1; equatorial Pacific 1991–2025; reference 1991–2020; checked input.")
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )

    # %% Marine heatwaves and cold spells use distinct percentile tails
    masks, catalogs, metrics, fits = {}, {}, {}, {}
    for name, detector in (("MHW", definition), ("MCS", marine_definition(cold=True))):
        fitted = xhw.fit_baseline(
            prepared, variable="sst", grid=grid, baseline=baseline, definition=detector
        )
        fitted.check_invariants()
        fits[name] = fitted
        masks[name] = xhw.detect(
            prepared, variable="sst", grid=grid, definition=detector, baseline_result=fitted
        )
        if storage is not None:
            masks[name] = mapped_product(masks[name], storage, name)
            if name == "MHW":
                reuse_temperature(prepared, masks[name])
        masks[name].check_invariants()
        catalogs[name] = xhw.build_catalog(masks[name], analysis_mode="retrospective")
        catalogs[name].check_against(masks[name])
        metrics[name] = temporal_summary(masks[name], catalogs[name])
    mask = masks["MHW"]
    figure = frequency_figure(masks, output_dir)
    figure = series_figure(
        raw,
        mask,
        variable="sst",
        selection={"lat": 20, "lon": 100},
        period=("2015-03-01", "2016-06-01"),
        output_dir=output_dir,
    )

    # %% Compare the same definition with a retrospective detrended baseline
    dynamic = replace(baseline, method="detrended")
    fitted_dynamic = xhw.fit_baseline(
        prepared, variable="sst", grid=grid, baseline=dynamic, definition=definition
    )
    if storage is not None:
        fitted_dynamic = mapped_product(fitted_dynamic, storage, "dynamic-baseline")
    fitted_dynamic.check_invariants()
    dynamic_mask = xhw.detect(
        prepared, variable="sst", grid=grid, definition=definition, baseline_result=fitted_dynamic
    )
    if storage is not None:
        dynamic_mask = mapped_product(dynamic_mask, storage, "dynamic-mask")
    dynamic_mask.check_invariants()
    dynamic_catalog = xhw.build_catalog(dynamic_mask, analysis_mode="retrospective")
    dynamic_catalog.check_against(dynamic_mask)
    metrics["detrended"] = temporal_summary(dynamic_mask, dynamic_catalog)
    common_valid = mask.dataset.valid & dynamic_mask.dataset.valid
    a, b = mask.dataset.event_membership, dynamic_mask.dataset.event_membership
    union = int(((a | b) & common_valid).sum())
    metrics["fixed_detrended_jaccard"] = (
        float(((a & b) & common_valid).sum()) / union if union else 1.0
    )
    figure = frequency_figure(
        {"Fixed MHW": mask, "Detrended MHW": dynamic_mask}, output_dir, name="baseline_comparison"
    )

    # %% Track event-membership patches; the default recipe would not track
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(
                footprint=xhw.Footprint.connected(1), min_area=xhw.MinArea.default()
            ),
            xhw.RecipeStep.track(min_overlap=0.0, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe, analysis_mode="retrospective")
    objects.check_invariants()
    objects.check_against(mask)
    figure = sequence_figure(objects, ["2015-09-15", "2015-12-15", "2016-03-15"], output_dir)

    # %% Derive branches and spacetime events with area-weighted intensity
    intensity = mapped_intensity(mask, storage, "intensity") if storage else mask.intensity()
    graph = xhw.build_lineage(objects, field=intensity, weighted=True)
    graph.check_invariants()
    graph.check_against(objects)
    metrics["lineage_catalog_export"] = lineage_catalog_status(graph, mask)
    event = largest_event(graph)
    figure = genealogy_figure(graph, event, output_dir)
    metrics_figure(graph, objects, event, output_dir)
    del figure
    metrics["spatial"] = spatial_summary(objects, graph)
    metrics["largest_size_time_event"] = int(event)

    # %% Preserve the distinction between reproduced output and scientific approval
    if products_dir is not None:
        entries = export_products(
            products_dir,
            "mhw",
            baseline=fits["MHW"],
            mask=mask,
            catalog=catalogs["MHW"],
            objects=objects,
            graph=graph,
        )
        entries |= export_products(
            products_dir, "mcs", baseline=fits["MCS"], mask=masks["MCS"], catalog=catalogs["MCS"]
        )
        entries |= export_products(
            products_dir, "detrended_mhw", mask=dynamic_mask, catalog=dynamic_catalog
        )
        write_products_manifest(products_dir, "oisst_equatorial", entries)
    report = write_report(
        output_dir,
        "oisst_equatorial",
        metrics,
        grid=grid,
        baseline=baseline,
        definition=definition,
        recipe=recipe,
        graph=graph,
        review="executed; summaries pending scientific review",
        extra={
            "selection": (
                "Equatorial Pacific (Nino 3.4): latitude -5 to 5; longitude -170 to -120; 1991–2025"
            ),
            "dynamic_baseline": dynamic.to_dict(),
            "comparison_denominator": "union of membership on jointly valid cell-days",
            "common_valid_cell_days": int(common_valid.sum()),
            "source_resolution_degrees": 0.25,
        },
    )
    return report


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir, args.products_dir)
