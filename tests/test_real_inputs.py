# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Input contracts of the real-data cases: missing files, wrong checksums, optional readers."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


def test_missing_registered_input_has_reproduction_instructions(tmp_path):
    from _data import load_oisst

    with pytest.raises(FileNotFoundError, match="--data-dir.*README"):
        load_oisst(tmp_path)


def test_missing_equatorial_input_has_reproduction_instructions(tmp_path):
    from _data import load_oisst_equatorial

    with pytest.raises(FileNotFoundError, match="--data-dir.*README"):
        load_oisst_equatorial(tmp_path)


def test_equatorial_dataset_matches_its_registration():
    """Skipped when `data/` has no compressed equatorial file (it is a release asset)."""
    from _data import load_oisst_equatorial, record

    data = EXAMPLES.parent / "data"
    entry = record("test_datasets")["datasets"]["oisst_equatorial"]
    if not (data / entry["file"]).is_file():
        pytest.skip("compressed equatorial dataset not downloaded")
    ds, grid, _, _ = load_oisst_equatorial(data)
    grid.check(ds)
    assert dict(ds.sizes) == {"time": 12784, "lat": 40, "lon": 200}
    assert bool(ds["domain_mask"].all())  # a box of open ocean


def test_wrong_checksum_is_not_silently_registered(tmp_path):
    from _data import checked_file

    (tmp_path / "input.nc").write_bytes(b"a different input")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checked_file(tmp_path, "input.nc", "0" * 64)


def test_netcdf_reader_missing_has_an_actionable_error(monkeypatch):
    original = builtins.__import__

    def without_h5netcdf(name, *args, **kwargs):
        if name.split(".")[0] == "h5netcdf":
            raise ImportError("simulated absent optional requirement")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_h5netcdf)
    from _data import open_loaded

    with pytest.raises(ImportError, match="examples/requirements.txt"):
        open_loaded(Path("unused.nc"))


def test_retained_level_thickness_has_explicit_interfaces():
    from _data import level_thickness

    # Depths .5, 2, 7 imply interfaces 0, 1.25, 4.5, 9.5 (metres).
    np.testing.assert_allclose(level_thickness(np.array([0.5, 2.0, 7.0])), [1.25, 3.25, 5.0])
