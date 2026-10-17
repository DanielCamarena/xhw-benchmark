# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Bound memory in the North Pacific example, without splitting any cell's time series.

Only interpolation, fixed baselines and temporal detection run on latitude blocks.
The complete prepared object window is assembled before segmentation or tracking.
No regridding, dtype conversion or change to scientific selections occurs here.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from _data import marine_definition
from _report import export_products, frequency, mask_summary

import xhw

BLOCK_ROWS = 4


def ocean_domain(raw, *, rows=BLOCK_ROWS):
    """Ocean = any finite observation over the entire original time interval."""
    lat, lon = "lat", "lon"
    values = np.zeros((raw.sizes[lat], raw.sizes[lon]), bool)
    for start in range(0, raw.sizes[lat], rows):
        stop = min(start + rows, raw.sizes[lat])
        tile = raw.sst.isel({lat: slice(start, stop)}).values
        values[start:stop] = np.isfinite(tile).any(axis=raw.sst.get_axis_num("time"))
    return xr.DataArray(values, dims=(lat, lon), coords={lat: raw[lat], lon: raw[lon]})


def latitude_blocks(raw, *, rows=BLOCK_ROWS):
    """Load bounded spatial pieces, each retaining every original timestamp."""
    if not isinstance(rows, int) or rows < 1:
        raise ValueError("rows must be a positive integer")
    for start in range(0, raw.sizes["lat"], rows):
        stop = min(start + rows, raw.sizes["lat"])
        yield f"lat{start:04d}_{stop:04d}", raw.isel(lat=slice(start, stop)).load()


def combine_baselines(parts, grid, baseline, definition):
    """Join independent per-cell fixed fits and record their parent runs."""
    if baseline.method != "fixed_climatology":
        raise ValueError("This example combines fixed climatology products only")
    joined = xr.concat(
        [p.dataset for p in parts],
        dim="lat",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        combine_attrs="drop_conflicts",
    )
    provenance = xhw.Provenance.new(
        inputs=(baseline.source,),
        parents=tuple(p.provenance.run_id for p in parts),
        steps=(
            xhw.ProcessingStep(
                name="assemble_latitude_blocks", params=(("whole_time_series_per_cell", True),)
            ),
        ),
    )
    result = xhw.BaselineResult.from_dataset(
        joined, grid=grid, baseline=baseline, definition=definition, provenance=provenance
    )
    result.check_invariants()
    return result


def temporal_blocks(
    raw, grid, baseline, definition, *, window, rows=BLOCK_ROWS, products_dir: Path | None = None
):
    """Summarize full-period masks; retain a prepared global window for spatial stages.

    Full-period flag products, when requested, are saved separately per block.
    Their physical latitude coordinates determine how to concatenate them.
    """
    detectors = {"MHW": definition, "MCS": marine_definition(cold=True)}
    metrics = {name: {} for name in detectors}
    frequency_parts = {name: [] for name in detectors}
    fit_parts = {name: [] for name in detectors}
    windows, entries = [], {}
    for label, tile in latitude_blocks(raw, rows=rows):
        prepared = xhw.interpolate_gaps(
            tile, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
        )
        # Preserve full-period filled fractions and edge context in the saved window.
        windows.append(prepared.sel(time=slice(*window)).copy(deep=True))
        for name, detector in detectors.items():
            fitted = xhw.fit_baseline(
                prepared, variable="sst", grid=grid, baseline=baseline, definition=detector
            )
            fitted.check_invariants()
            mask = xhw.detect(
                prepared, variable="sst", grid=grid, definition=detector, baseline_result=fitted
            )
            mask.check_invariants()
            for key, value in mask_summary(mask).items():
                metrics[name][key] = metrics[name].get(key, 0) + value
            frequency_parts[name].append(frequency(mask))
            fit_parts[name].append(fitted)
            if products_dir is not None:
                entries |= export_products(
                    products_dir, f"{name.lower()}_1991_2025_{label}", mask=mask
                )
            del mask, fitted
        del prepared, tile
        print(f"{label}: both full-period temporal masks checked", flush=True)
    raw.close()
    prepared_window = xr.concat(
        windows,
        dim="lat",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        combine_attrs="drop_conflicts",
    )
    del windows
    fits, frequencies = {}, {}
    for name, detector in detectors.items():
        fits[name] = combine_baselines(fit_parts.pop(name), grid, baseline, detector)
        frequencies[name] = xr.concat(frequency_parts[name], dim="lat")
        if products_dir is not None:
            entries |= export_products(
                products_dir, f"{name.lower()}_1991_2025", baseline=fits[name]
            )
    return prepared_window, metrics, frequencies, fits, entries
