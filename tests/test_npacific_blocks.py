# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Spatial partitioning preserves complete temporal rules before global segmentation."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xhw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from _data import marine_definition, synthetic_inputs  # noqa: E402
from _report import frequency, mask_summary  # noqa: E402
from npacific_blocks import ocean_domain, temporal_blocks  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


@pytest.mark.parametrize("rows", [2, 4])
def test_blocks_equal_whole_grid_including_missing_and_window_edge_context(rows):
    raw, grid, baseline, definition = synthetic_inputs()
    raw = raw.sel(time=slice("2000-01-01", "2003-01-31")).copy(deep=True)
    baseline = replace(baseline, reference_period=("2000-01-01", "2002-12-31"), min_samples=20)
    # A gap straddles the object window edge. Interpolation and its quality fraction
    # must use all original times, rather than treating the window as a new series.
    raw.sst.loc[dict(time=slice("2002-12-31", "2003-01-01"), lat=raw.lat[2], lon=raw.lon[2])] = (
        np.nan
    )
    raw["domain_mask"] = ocean_domain(raw, rows=rows)
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )
    bounds = ("2003-01-01", "2003-01-20")
    window, summaries, frequencies, fits, _ = temporal_blocks(
        raw, grid, baseline, definition, window=bounds, rows=rows
    )
    xr.testing.assert_identical(window, prepared.sel(time=slice(*bounds)))
    for name, detector in [("MHW", definition), ("MCS", marine_definition(cold=True))]:
        full_fit = xhw.fit_baseline(
            prepared, variable="sst", grid=grid, baseline=baseline, definition=detector
        )
        xr.testing.assert_identical(fits[name].dataset, full_fit.dataset)
        full = xhw.detect(
            prepared, variable="sst", grid=grid, definition=detector, baseline_result=full_fit
        )
        assert summaries[name] == mask_summary(full)
        xr.testing.assert_identical(frequencies[name], frequency(full))
    reference = xhw.detect(
        prepared.sel(time=slice(*bounds)),
        variable="sst",
        grid=grid,
        definition=definition,
        baseline_result=fits["MHW"],
    )
    assembled = xhw.detect(
        window, variable="sst", grid=grid, definition=definition, baseline_result=fits["MHW"]
    )
    xr.testing.assert_identical(assembled.dataset, reference.dataset)
    recipe = xhw.SpatialRecipe(input_layer="event_membership",
                               steps=(xhw.RecipeStep.segment(min_area=xhw.MinArea.none()),))
    a = xhw.build_objects(assembled, recipe=recipe)
    b = xhw.build_objects(reference, recipe=recipe)
    xr.testing.assert_identical(a.dataset.label, b.dataset.label)
