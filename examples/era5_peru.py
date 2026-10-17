# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""ERA5: declared local-day aggregation, TX90/TN90 and joint profiles over land and ocean."""

from pathlib import Path

import numpy as np
from _data import era5_profiles, hourly_era5, load_era5, record, surface_grid
from _report import (
    arguments,
    export_products,
    temporal_summary,
    write_products_manifest,
    write_report,
)
from plotting import frequency_figure, pyplot, save_figure

import xhw


def run(data_dir: Path, output_dir: Path, products_dir: Path | None = None) -> dict:
    # %% Load the registered daily analysis and a short hourly demonstration
    daily, baseline = load_era5(data_dir)
    hourly = hourly_era5(data_dir)
    print("ERA5 Peru 1991–2025; reference 1991–2020; offset:-05:00; SHA-256 verified.")

    # %% Hourly period-ending extrema belong to the day of their interval start
    tx = xhw.aggregate_daily(
        hourly,
        variable="mx2t",
        statistic="max",
        sampling="period_ending",
        input_clock="UTC",
        day_definition="offset:-05:00",
        name="tx",
    )
    tn = xhw.aggregate_daily(
        hourly,
        variable="mn2t",
        statistic="min",
        sampling="period_ending",
        input_clock="UTC",
        day_definition="offset:-05:00",
        name="tn",
    )
    contrast = xhw.aggregate_daily(
        hourly,
        variable="t2m",
        statistic="max",
        sampling="instantaneous",
        input_clock="UTC",
        day_definition="offset:-05:00",
        name="tx_t2m",
    )
    interior = slice("2023-03-01", "2023-03-03")
    errors = {}
    for name, aggregated in (("tx", tx), ("tn", tn)):
        a, b = aggregated[name].sel(time=interior), daily[name].sel(time=interior)
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
        errors[name] = 0.0
    plt = pyplot()
    figure, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for values, label in (
        (tx.tx, "TX: max mx2t"),
        (tn.tn, "TN: min mn2t"),
        (contrast.tx_t2m, "Contrast only: max t2m"),
    ):
        ax.plot(values.time, values.isel(latitude=15, longitude=10), "o-", label=label)
    ax.set(
        ylabel="Temperature (K)",
        xlabel="Local date (offset:-05:00)",
        title="Complete interior days have 24 hourly samples; partial edges are NaN",
    )
    ax.legend(fontsize=8)
    figure = save_figure(figure, output_dir, "aggregation")

    # %% Define complementary air domains; neither is volumetric ocean temperature
    grids = {
        domain: surface_grid(
            lat="latitude",
            lon="longitude",
            domain=domain,
            domain_mask=mask,
            product="ERA5 single levels",
            resolution="0.25 degree",
        )
        for domain, mask in (("air_over_land", "land_mask"), ("air_over_ocean", "ocean_mask"))
    }
    profiles = era5_profiles()
    metrics, masks, catalogs = {}, {}, {}

    # %% Fit and detect each profile; combine daily TX and TN before testing persistence
    for domain, grid in grids.items():
        grid.check(daily)
        metrics[domain], masks[domain] = {}, {}
        for profile, (definition, variable) in profiles.items():
            fitted = xhw.fit_baseline(
                daily, variable=variable, grid=grid, baseline=baseline, definition=definition
            )
            fitted.check_invariants()
            mask = xhw.detect(
                daily, variable=variable, grid=grid, definition=definition, baseline_result=fitted
            )
            mask.check_invariants()
            catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
            catalog.check_against(mask)
            masks[domain][profile] = mask
            catalogs[(domain, profile)] = catalog
            metrics[domain][profile] = temporal_summary(mask, catalog)
        figure = frequency_figure(masks[domain], output_dir, name=domain)

    # %% Report profile definitions and the verified hourly-to-daily agreement
    grid = grids["air_over_land"]
    definition = profiles["TX90"][0]
    metrics["aggregation_max_absolute_difference_K"] = errors
    if products_dir is not None:
        entries = {}
        for (domain, profile), catalog in catalogs.items():
            label = f"{domain}_{profile}".lower().replace("+", "").replace(" ", "_")
            entries |= export_products(
                products_dir, label, mask=masks[domain][profile], catalog=catalog
            )
        write_products_manifest(products_dir, "era5_peru", entries)
    report = write_report(
        output_dir,
        "era5",
        metrics,
        grid=grid,
        baseline=baseline,
        definition=definition,
        review="executed; real AHW summaries pending scientific review",
        extra={
            "selection": "Peru -12 to -4 degrees north, -82 to -77 east; 1991–2025",
            "domains": {k: v.to_dict() for k, v in grids.items()},
            "profiles": {k: v[0].to_dict() for k, v in profiles.items()},
            "hourly_demonstration": "2023-03-01 to 2023-03-04 UTC; 24 samples/day",
            "hourly_sha256": record("era5")["hourly_files_sha256"]["2023.nc"],
            "aggregation_counts": tx.tx_n_samples.isel(latitude=15, longitude=10).values.tolist(),
        },
    )
    return report


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir, args.products_dir)
