# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Disk backing preserves the labelled products, missing values and physical intensity."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xhw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from _data import synthetic_inputs  # noqa: E402
from array_storage import (  # noqa: E402
    mapped_array,
    mapped_intensity,
    mapped_product,
    reuse_temperature,
)


@pytest.fixture
def products():
    raw, grid, baseline, definition = synthetic_inputs()
    prepared = xhw.interpolate_gaps(raw, variable="sst", grid=grid, max_days=2,
                                     max_filled_fraction=0.05)
    fit = xhw.fit_baseline(prepared, variable="sst", grid=grid, baseline=baseline,
                           definition=definition)
    mask = xhw.detect(prepared, variable="sst", grid=grid, definition=definition,
                      baseline_result=fit)
    return raw, prepared, fit, mask


def test_mapped_mask_and_intensity_are_identical(products, tmp_path):
    _, _, _, mask = products
    cached = mapped_product(mask, tmp_path, "mask")
    xr.testing.assert_identical(cached.to_dataset(), mask.to_dataset())
    assert cached.provenance == mask.provenance
    cached.check_invariants()
    xr.testing.assert_identical(mapped_intensity(cached, tmp_path, "intensity"), mask.intensity())
    backing = cached.dataset.temperature.data
    while backing is not None and not isinstance(backing, np.memmap):
        backing = getattr(backing, "base", None)
    assert isinstance(backing, np.memmap)


def test_reusing_temperature_keeps_raw_observations_separate(products, tmp_path):
    raw, prepared, _, mask = products
    original, original_raw = prepared.copy(deep=True), raw.copy(deep=True)
    raw["sst"] = mapped_array(raw.sst, tmp_path, "raw", block_days=7)
    cached = mapped_product(mask, tmp_path, "mask")
    reuse_temperature(prepared, cached)
    xr.testing.assert_identical(prepared, original)
    xr.testing.assert_identical(raw, original_raw)
    assert np.shares_memory(prepared.sst.data, cached.dataset.temperature.data)


def test_dynamic_baseline_survives_mapping(products, tmp_path):
    _, prepared, fit, mask = products
    dynamic = xhw.fit_baseline(prepared, variable="sst", grid=mask.grid,
                               baseline=replace(fit.baseline, method="detrended"),
                               definition=mask.definition)
    cached = mapped_product(dynamic, tmp_path, "dynamic")
    xr.testing.assert_identical(cached.to_dataset(), dynamic.to_dataset())
    cached.check_invariants()


def test_read_only_backing_is_reused_without_another_disk_copy(products, tmp_path):
    _, _, _, mask = products
    first = mapped_array(mask.dataset.temperature, tmp_path, "first")
    before = set(tmp_path.iterdir())
    second = mapped_array(first, tmp_path, "second")
    assert set(tmp_path.iterdir()) == before
    xr.testing.assert_identical(first, second)
    assert np.shares_memory(first.data, second.data)
