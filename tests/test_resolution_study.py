# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""The comparison helpers of the OSTIA tutorial, checked on the hand-counted synthetic field."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import xhw

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


@pytest.fixture(scope="module")
def masks():
    from _data import synthetic_inputs

    raw, grid, baseline, definition = synthetic_inputs()
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )

    def detect(spec):
        fitted = xhw.fit_baseline(
            prepared, variable="sst", grid=grid, baseline=baseline, definition=spec
        )
        return xhw.detect(
            prepared, variable="sst", grid=grid, definition=spec, baseline_result=fitted
        )

    return detect(definition), detect(replace(definition, min_duration_days=14))


def test_frequency_recovers_the_hand_counted_membership(masks):
    from ostia_peru import frequency

    mask, _ = masks
    days = mask.dataset["valid"].sum("time")
    # 216 member cell-days were counted by hand in the tutorial (see test_tutorial_contracts).
    assert round(float((frequency(mask) / 100 * days).sum())) == 216


def test_identical_masks_agree_perfectly(masks):
    from ostia_peru import agreement

    mask, _ = masks
    result = agreement(mask, mask)
    assert result["membership_jaccard"] == 1.0
    assert result["frequency_rmse_pct_points"] == 0.0
    assert result["frequency_correlation"] == pytest.approx(1.0)


def test_a_stricter_definition_lowers_the_agreement(masks):
    from ostia_peru import agreement

    loose, strict = masks
    result = agreement(loose, strict)
    # The 14-day minimum removes the 13-day cells and keeps the 15-day bars, so membership
    # shrinks but never grows: intersection / union = |strict| / |loose|.
    expected = float(strict.dataset["event_membership"].sum()) / float(
        loose.dataset["event_membership"].sum()
    )
    assert 0.0 < result["membership_jaccard"] < 1.0
    assert result["membership_jaccard"] == pytest.approx(expected)
    assert np.isfinite(result["frequency_mean_difference_pct_points"])
