# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Resolution study off Peru: OISST (0.25 degree) against OSTIA (0.05 degree) in three arms.

Separates two effects that a plain OISST versus OSTIA comparison mixes:

* product effect  = OISST at 0.25 degree  versus  OSTIA averaged to the same 0.25 degree grid;
* resolution effect = OSTIA averaged to 0.25 degree  versus  OSTIA at its native 0.05 degree.

Every arm uses the same definition (Hobday MHW, P90, five days, two-day gaps), the same
fixed 1991-2020 climatology and the same spatial recipe, so only the data change. Needs the
compressed test datasets (see the README): oisst_peru_1991_2025.nc and
ostia_peru_1991_2025.nc. Summaries are reproduced numbers pending scientific review.
"""

import time
from pathlib import Path

import numpy as np
import xarray as xr
from _data import fixed_baseline, load_resolution_study, marine_definition, surface_grid
from _report import (
    arguments,
    export_products,
    frequency,
    spatial_summary,
    temporal_summary,
    write_products_manifest,
    write_report,
)
from array_storage import enabled, mapped_array, mapped_intensity, mapped_product, reuse_temperature
from content_hash import content_sha256
from plotting import map_panel, pyplot, save_figure

import xhw

LABELS = {
    "oisst": "OISST 0.25°",
    "ostia_coarse": "OSTIA averaged to 0.25°",
    "ostia_fine": "OSTIA 0.05°",
}
COARSEN = 5  # OSTIA sub-cells per OISST cell along each axis
SPATIAL = xhw.SpatialRecipe(
    input_layer="event_membership",
    steps=(
        xhw.RecipeStep.segment(
            footprint=xhw.Footprint.connected(1), min_area=xhw.MinArea.default()
        ),
        xhw.RecipeStep.track(min_overlap=0.0, max_gap_days=0, min_duration_days=5),
    ),
)
UNFILTERED = xhw.SpatialRecipe(
    input_layer="event_membership",
    steps=(
        xhw.RecipeStep.segment(footprint=xhw.Footprint.connected(1), min_area=xhw.MinArea.none()),
    ),
)


def analyse(name: str, ds: xr.Dataset, resolution: str, content_sha: str,
            *, output_dir: Path | None = None) -> dict:
    """Run every xHW stage on one arm and keep the products."""
    grid = surface_grid(product=LABELS[name], resolution=resolution)
    grid.check(ds)
    storage = output_dir / "_arrays" / name if enabled() and output_dir is not None else None
    if storage is not None:
        ds["sst"] = mapped_array(ds.sst, storage, "input-sst")
    assert bool((ds["sst"].notnull() | ~ds["domain_mask"]).all()), "missing ocean days"
    source = xhw.SourceRef(
        dataset_id=f"xhw-test-datasets:{name}",
        variable="sst",
        accessed="2026-10-11",
        product_status="final",
        checksum=f"sha256:{content_sha}",
    )
    baseline = fixed_baseline(source)
    definition = marine_definition()
    fitted = xhw.fit_baseline(
        ds, variable="sst", grid=grid, baseline=baseline, definition=definition
    )
    fitted.check_invariants()
    mask = xhw.detect(ds, variable="sst", grid=grid, definition=definition, baseline_result=fitted)
    if storage is not None:
        mask = mapped_product(mask, storage, "mask")
        reuse_temperature(ds, mask)
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    # Only the patch count is needed from this comparison; release the dense labels
    # before constructing the tracked field and genealogy.
    loose = xhw.build_objects(mask, recipe=UNFILTERED)
    loose.check_invariants()
    loose_count = int(loose.dataset.sizes["object"])
    del loose
    objects = xhw.build_objects(mask, recipe=SPATIAL, analysis_mode="retrospective")
    objects.check_invariants()
    objects.check_against(mask)
    intensity = mapped_intensity(mask, storage, "intensity") if storage else mask.intensity()
    graph = xhw.build_lineage(objects, field=intensity, weighted=True)
    graph.check_invariants()
    graph.check_against(objects)
    return dict(
        grid=grid,
        baseline=baseline,
        definition=definition,
        fitted=fitted,
        mask=mask,
        catalog=catalog,
        objects=objects,
        graph=graph,
        loose_count=loose_count,
    )


def aggregate_fine_frequency(mask, reference: xr.DataArray) -> xr.DataArray:
    """Frequency of the 0.05° arm, pooled over the 5 x 5 sub-cells of each 0.25° cell."""
    ds = mask.dataset
    inside = xr.zeros_like(reference, dtype="int64")
    total = xr.zeros_like(reference, dtype="int64")
    for year in range(1991, 2026):  # by year, to bound the integer working copies
        part = ds.sel(time=str(year))
        member = (part["event_membership"] & part["valid"]).astype("int32")
        for target, field in ((inside, member), (total, part["valid"].astype("int32"))):
            pooled = field.coarsen(lat=COARSEN, lon=COARSEN).sum().sum("time")
            target += pooled.assign_coords(lat=reference["lat"], lon=reference["lon"]).astype(
                "int64"
            )
    return 100 * inside / total.where(total > 0)


def agreement(a, b) -> dict:
    """Event agreement of two masks on the same grid, on jointly valid cell-days."""
    valid = a.dataset["valid"] & b.dataset["valid"]
    ma, mb = a.dataset["event_membership"] & valid, b.dataset["event_membership"] & valid
    union = int((ma | mb).sum())
    fa, fb = frequency(a, valid), frequency(b, valid)
    both = fa.notnull() & fb.notnull()
    diff = (fb - fa).where(both)
    return {
        "jointly_valid_cell_days": int(valid.sum()),
        "membership_jaccard": float((ma & mb).sum()) / union if union else 1.0,
        "frequency_correlation": float(xr.corr(fa.where(both), fb.where(both))),
        "frequency_mean_difference_pct_points": float(diff.mean()),
        "frequency_rmse_pct_points": float(np.sqrt((diff**2).mean())),
    }


def temperature_agreement(oisst: xr.Dataset, ostia: xr.Dataset) -> dict:
    """Daily SST of two products on cells that are ocean in both (degC)."""
    both = oisst["domain_mask"] & ostia["domain_mask"]
    a, b = oisst["sst"].where(both), ostia["sst"].where(both)
    diff = (b - a).stack(s=("time", "lat", "lon")).dropna("s")
    return {
        "common_ocean_cells": int(both.sum()),
        "mean_difference_degC": float(diff.mean()),
        "rmse_degC": float(np.sqrt((diff**2).mean())),
        "correlation": float(xr.corr(a, b)),
    }


def resolution_frequency_figure(arms: dict, pooled: xr.DataArray, output_dir: Path):
    plt = pyplot()
    f = {k: frequency(arms[k]["mask"]) for k in ("oisst", "ostia_coarse", "ostia_fine")}
    low = float(np.floor(min(x.min() for x in (*f.values(), pooled))))
    high = float(np.ceil(max(x.max() for x in (*f.values(), pooled))))
    shared = f["oisst"].notnull() & f["ostia_coarse"].notnull()
    both = f["ostia_coarse"].notnull() & pooled.notnull()
    product = (f["ostia_coarse"] - f["oisst"]).where(shared)
    resolution = (pooled - f["ostia_coarse"]).where(both)
    limit = float(np.ceil(2 * max(abs(product).max(), abs(resolution).max())) / 2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    for ax, key in zip(axes[0], f, strict=True):
        map_panel(
            ax,
            f[key],
            title=LABELS[key],
            vmin=low,
            vmax=high,
            cbar_kwargs={"label": f"% of valid days in an MHW (scale starts at {low:g})"},
        )
    map_panel(
        axes[1, 0],
        product,
        title="Product effect: OSTIA 0.25° − OISST",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        cbar_kwargs={"label": "percentage points"},
    )
    map_panel(
        axes[1, 1],
        resolution,
        title="Resolution effect: OSTIA 0.05° pooled − 0.25°",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        cbar_kwargs={"label": "percentage points"},
    )
    ax = axes[1, 2]
    ax.scatter(f["oisst"].where(shared), f["ostia_coarse"].where(shared), s=14, label="OSTIA 0.25°")
    ax.scatter(
        f["oisst"].where(both), pooled.where(both), s=14, marker="x", label="OSTIA 0.05° pooled"
    )
    ax.plot([low, high], [low, high], color="k", linewidth=1)
    ax.set(
        xlabel="OISST frequency (%)",
        ylabel="OSTIA frequency (%)",
        title="Cell by cell",
        xlim=(low, high),
        ylim=(low, high),
    )
    ax.legend(fontsize=8)
    fig.suptitle("MHW frequency off Peru, 1991–2025: product and resolution effects", fontsize=12)
    return save_figure(fig, output_dir, "resolution_frequency")


def resolution_events_figure(arms: dict, output_dir: Path):
    plt = pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for key, color in zip(arms, ("#28364a", "#167c80", "#c58220"), strict=True):
        ds = arms[key]["mask"].dataset
        valid = ds["valid"].groupby("time.year").sum(["time", "lat", "lon"])
        inside = (
            (ds["event_membership"] & ds["valid"]).groupby("time.year").sum(["time", "lat", "lon"])
        )
        (100 * inside / valid).plot(ax=left, label=LABELS[key], color=color)
    left.set(
        xlabel="Year",
        ylabel="% of valid cell-days in an MHW",
        title="Domain-mean frequency by year",
    )
    left.legend(fontsize=8)
    names = list(arms)
    width = 0.38
    for i, label in enumerate(("5% area filter", "no filter")):
        counts = [int(arms[k]["objects"].dataset.sizes["object"]) if i == 0
                  else arms[k]["loose_count"] for k in names]
        bars = right.bar(np.arange(len(names)) + (i - 0.5) * width, counts, width, label=label)
        right.bar_label(bars, fontsize=8)
    right.set_xticks(np.arange(len(names)), [LABELS[k] for k in names], fontsize=8)
    right.set(
        ylabel="Daily patches, 1991–2025", title="Connected patches before and after the filter"
    )
    right.legend(fontsize=8)
    return save_figure(fig, output_dir, "resolution_events")


def run(data_dir: Path, output_dir: Path, products_dir: Path | None = None) -> dict:
    # %% Open the compressed test datasets (their content hash is verified) and build the arms
    datasets, paths = load_resolution_study(data_dir)
    shas = {"oisst": content_sha256(paths["oisst"]), "ostia": content_sha256(paths["ostia"])}
    print("Arms:", {k: dict(v.sizes) for k, v in datasets.items()})

    # %% Same specs, same recipe: detect and follow events in each arm
    resolutions = {
        "oisst": "0.25 degree",
        "ostia_coarse": "0.25 degree (mean of 5 x 5)",
        "ostia_fine": "0.05 degree",
    }
    arms = {}
    for name, ds in datasets.items():
        start = time.perf_counter()
        arms[name] = analyse(
            name, ds, resolutions[name], shas["oisst" if name == "oisst" else "ostia"],
            output_dir=output_dir,
        )
        events = arms[name]["catalog"].dataset.sizes["event"]
        print(f"{name}: {events} per-cell events ({time.perf_counter() - start:.0f} s)", flush=True)

    # %% Compare: temperature, product effect and resolution effect
    pooled = aggregate_fine_frequency(arms["ostia_fine"]["mask"], frequency(arms["oisst"]["mask"]))
    base = frequency(arms["ostia_coarse"]["mask"])
    both = base.notnull() & pooled.notnull()
    metrics = {
        "arms": {
            name: {
                "ocean_cells": int(datasets[name]["domain_mask"].sum()),
                "temporal": temporal_summary(arm["mask"], arm["catalog"]),
                "spatial": spatial_summary(arm["objects"], arm["graph"]),
                "daily_patches_without_area_filter": arm["loose_count"],
                "event_cell_days_pct_of_valid": 100
                * float(arm["mask"].dataset["event_membership"].sum())
                / float(arm["mask"].dataset["valid"].sum()),
                "events_per_ocean_cell": arm["catalog"].dataset.sizes["event"]
                / int(datasets[name]["domain_mask"].sum()),
            }
            for name, arm in arms.items()
        },
        "temperature_oisst_vs_ostia_025": temperature_agreement(
            datasets["oisst"], datasets["ostia_coarse"]
        ),
        "product_effect_oisst_vs_ostia_025": agreement(
            arms["oisst"]["mask"], arms["ostia_coarse"]["mask"]
        ),
        "resolution_effect_ostia_025_vs_005_pooled": {
            "cells": int(both.sum()),
            "frequency_correlation": float(xr.corr(base.where(both), pooled.where(both))),
            "frequency_mean_difference_pct_points": float((pooled - base).where(both).mean()),
            "frequency_rmse_pct_points": float(np.sqrt(((pooled - base) ** 2).where(both).mean())),
        },
    }
    yearly = {
        name: (arm["mask"].dataset["event_membership"] & arm["mask"].dataset["valid"])
        .groupby("time.year")
        .sum(["time", "lat", "lon"])
        / arm["mask"].dataset["valid"].groupby("time.year").sum(["time", "lat", "lon"])
        for name, arm in arms.items()
    }
    metrics["yearly_domain_frequency_correlation"] = {
        "oisst_vs_ostia_025": float(xr.corr(yearly["oisst"], yearly["ostia_coarse"])),
        "ostia_025_vs_005": float(xr.corr(yearly["ostia_coarse"], yearly["ostia_fine"])),
    }
    if products_dir is not None:
        entries = {}
        for name, arm in arms.items():
            entries |= export_products(
                products_dir,
                name,
                baseline=arm["fitted"],
                mask=arm["mask"],
                catalog=arm["catalog"],
                objects=arm["objects"],
                graph=arm["graph"],
            )
        write_products_manifest(products_dir, "ostia_resolution", entries)
    figure = resolution_frequency_figure(arms, pooled, output_dir)
    figure = resolution_events_figure(arms, output_dir)
    del figure

    # %% Record the run; the review state stays explicit
    fine = arms["ostia_fine"]
    report = write_report(
        output_dir,
        "ostia",
        metrics,
        grid=fine["grid"],
        baseline=fine["baseline"],
        definition=fine["definition"],
        recipe=SPATIAL,
        graph=fine["graph"],
        review="executed; temporal and spatial summaries pending scientific review",
        extra={
            "selection": "Peru: latitude -12 to -4; longitude -82 to -77; 1991–2025",
            "arms": LABELS,
            "ostia_to_oisst_grid": "mean of 5 x 5 sub-cells; ocean needs at least 13 of 25",
            "calendar": "OISST stamps (12:00) and OSTIA stamps (00:00) compared by calendar day",
            "source_content_sha256": shas,
        },
    )
    return report


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir, args.products_dir)
