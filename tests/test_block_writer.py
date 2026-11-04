# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""BlockWriter and re-encoding keep every stored value (``content_sha256``)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5netcdf")
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import _acquire as aq  # noqa: E402
from content_hash import content_sha256  # noqa: E402


def sample(times, dim="time") -> xr.Dataset:
    """Float with NaN, an integer count, a mask without ``dim``, attributes."""
    rng = np.random.default_rng(0)
    n = len(times)
    value = rng.normal(290, 5, (n, 3, 4)).astype("float32")
    value[1 % n, 1, 2] = np.nan
    return xr.Dataset(
        {
            "tx": ((dim, "latitude", "longitude"), value, {"units": "K"}),
            "tx_n_samples": ((dim, "latitude", "longitude"), np.full((n, 3, 4), 24, "int16")),
            "lsm": (("latitude", "longitude"), rng.random((3, 4))),
        },
        coords={dim: times, "latitude": [0.0, -0.25, -0.5], "longitude": [1.0, 1.25, 1.5, 1.75]},
        attrs={"source": "test"},
    )


def blocks_of(ds, sizes, dim="time"):
    out, start = [], 0
    for size in sizes:
        part = ds.isel({dim: slice(start, start + size)})
        out.append(part if start == 0 else part.drop_vars("lsm"))
        start += size
    return out


@pytest.mark.parametrize("sizes", [[31, 28, 31], [1, 28, 2], [1, 1, 1, 30], [3]])
@pytest.mark.parametrize("compress", [None, 4])
def test_blocks_give_the_content_of_one_write(tmp_path, sizes, compress):
    # A first block of one day is held: its time units would otherwise be other ones.
    ds = sample(pd.date_range("2023-01-01", periods=sum(sizes)))
    whole = tmp_path / "whole.nc"
    ds.to_netcdf(whole, engine="h5netcdf", encoding=aq.zlib_encoding(ds.data_vars, compress))
    path = tmp_path / "blocks.nc"
    with aq.BlockWriter(path, compress=compress) as writer:
        for block in blocks_of(ds, sizes):
            writer.append(block)
        writer.commit()
    assert content_sha256(path) == content_sha256(whole)
    assert aq.file_encoding(path) == aq.encoding_label(compress)
    with xr.open_dataset(path, engine="h5netcdf") as got:
        xr.testing.assert_identical(got.load(), ds)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blocks.nc", "whole.nc"]


def test_hourly_blocks_along_valid_time(tmp_path):
    ds = sample(pd.date_range("2023-01-31 19:00", "2023-02-01 04:00", freq="h"), "valid_time")
    whole = tmp_path / "whole.nc"
    ds.to_netcdf(whole, engine="h5netcdf")
    with aq.BlockWriter(tmp_path / "blocks.nc", dim="valid_time") as writer:
        for block in blocks_of(ds, [5, 5], "valid_time"):
            writer.append(block)
        writer.commit()
    assert content_sha256(tmp_path / "blocks.nc") == content_sha256(whole)


def test_blocks_out_of_order_or_unlike_leave_no_file(tmp_path):
    ds = sample(pd.date_range("2023-01-01", periods=6))
    first, second = blocks_of(ds, [3, 3])
    path = tmp_path / "out.nc"
    with pytest.raises(ValueError, match="does not follow"):
        with aq.BlockWriter(path) as writer:
            writer.append(second)
            writer.append(first)
    with pytest.raises(ValueError, match="a block holds"):
        with aq.BlockWriter(path) as writer:
            writer.append(first)
            writer.append(second.drop_vars("tx_n_samples"))
    with aq.BlockWriter(path) as writer:  # left without commit: nothing appears
        writer.append(first)
    assert list(tmp_path.iterdir()) == []


def test_attributes_known_at_the_end(tmp_path):
    ds = sample(pd.date_range("2023-01-01", periods=4))
    with aq.BlockWriter(tmp_path / "out.nc") as writer:
        for block in blocks_of(ds, [2, 2]):
            writer.append(block)
        writer.update_attrs({"era5_expver_status": "final+preliminary"})
        writer.commit()
    with xr.open_dataset(tmp_path / "out.nc", engine="h5netcdf") as got:
        assert got.attrs == {"source": "test", "era5_expver_status": "final+preliminary"}


PACKED = {"scale_factor": 0.01, "add_offset": 290.0, "dtype": "int16", "_FillValue": -1}


def packed_dims(path) -> tuple:
    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as raw:
        tx = raw["tx"]
        return (
            str(tx.dtype),
            tx.attrs["scale_factor"],
            tx.attrs["add_offset"],
            tx.attrs["_FillValue"],
        )


@pytest.mark.parametrize(("before", "after"), [(None, 4), (4, None), (1, 9)])
def test_reencoding_changes_only_the_layout(tmp_path, before, after):
    # Review H3: packing and compression are combined in the file written first.
    ds = sample(pd.date_range("2023-01-01", periods=5))
    encoding = aq.zlib_encoding(ds.data_vars, before) or {}
    encoding["tx"] = {**encoding.get("tx", {}), **PACKED}
    path = tmp_path / "piece.nc"
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    assert packed_dims(path) == ("int16", 0.01, 290.0, -1)
    content = content_sha256(path)
    aq.reencode_netcdf_atomic(path, after)
    assert aq.file_encoding(path) == aq.encoding_label(after)
    assert content_sha256(path) == content  # packed int16, fill value and units kept
    assert packed_dims(path) == ("int16", 0.01, 290.0, -1)


@pytest.mark.parametrize("compress", [None, 4])
def test_packed_blocks_keep_their_packing(tmp_path, compress):
    ds = sample(pd.date_range("2023-01-01", periods=6))
    ds["tx"].encoding.update(PACKED)
    whole = tmp_path / "whole.nc"
    encoding = aq.zlib_encoding(ds.data_vars, compress) or {}
    encoding["tx"] = {**encoding.get("tx", {}), **PACKED}
    ds.to_netcdf(whole, engine="h5netcdf", encoding=encoding)
    path = tmp_path / "blocks.nc"
    with aq.BlockWriter(path, compress=compress) as writer:
        for block in blocks_of(ds, [2, 4]):
            writer.append(block)
        writer.commit()
    assert packed_dims(path) == ("int16", 0.01, 290.0, -1)
    assert content_sha256(path) == content_sha256(whole)


@pytest.mark.parametrize("held", [False, True])
def test_blocks_on_another_grid_are_refused(tmp_path, held):
    # Review H1: blocks are written by position, so their other coordinates must match.
    ds = sample(pd.date_range("2023-01-01", periods=6))
    first, second = blocks_of(ds, [1, 5] if held else [3, 3])
    flipped = second.isel(longitude=slice(None, None, -1))
    for other in (flipped, second.assign(lsm=ds["lsm"] + 1)):
        with pytest.raises(ValueError):
            with aq.BlockWriter(tmp_path / "out.nc") as writer:
                writer.append(first)
                writer.append(other)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("held", [False, True])
def test_blocks_with_other_variables_or_sizes_are_refused(tmp_path, held):
    # Re-review N1 (before the first write) and N2 (a dimension shrunk, coordinate omitted).
    ds = sample(pd.date_range("2023-01-01", periods=6))
    first, second = blocks_of(ds, [1, 5] if held else [3, 3])
    renamed = second.rename({"tx": "tn"})
    shrunk = second.isel(longitude=[0]).drop_vars("longitude")
    for other in (renamed, shrunk):
        with pytest.raises(ValueError, match="a block (holds|has sizes)"):
            with aq.BlockWriter(tmp_path / "out.nc") as writer:
                writer.append(first)
                writer.append(other)
    assert list(tmp_path.iterdir()) == []


def test_times_the_stored_units_cannot_hold_are_refused(tmp_path):
    # Review H2: daily units fixed by the first block cannot store a time at noon.
    ds = sample(pd.date_range("2023-01-01", periods=2))
    later = sample(pd.DatetimeIndex(["2023-01-02T12:00"])).drop_vars("lsm")
    with pytest.raises(ValueError, match="cannot be stored"):
        with aq.BlockWriter(tmp_path / "out.nc") as writer:
            writer.append(ds)
            writer.append(later)
    assert list(tmp_path.iterdir()) == []


# --- compression shared by every downloader (2026-11-04) -----------------------------------


def _raw_series(path: Path, days: int = 40) -> None:
    """Packed values over time, a static mask, coordinates with and without a fill value."""
    t = pd.date_range("2020-01-01", periods=days, freq="D")
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), rng.integers(-500, 3000, (days, 6, 5)) * 0.01),
            "mask": (("lat", "lon"), np.arange(30, dtype="f4").reshape(6, 5)),
        },
        coords={"time": t, "lat": np.arange(6, dtype="f4"), "lon": np.arange(5, dtype="f8")},
        attrs={"title": "t", "history": "h"},
    )
    ds["sst"].attrs = {"units": "degC", "long_name": "x"}
    encoding = {
        "sst": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": np.int16(-999)},
        "time": {"units": "days since 1978-01-01", "dtype": "float32"},
        "lon": {"_FillValue": None},
    }
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def _described(path: Path) -> dict:
    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as ds:
        return {
            "attrs": repr(ds.attrs),
            "vars": {k: (str(v.dtype), repr(v.attrs)) for k, v in ds.variables.items()},
        }


@pytest.mark.parametrize("block_bytes", [1, 10**9])  # block by block, and whole
@pytest.mark.parametrize("compress", [None, 4])
def test_reencode_keeps_content_and_attributes(tmp_path, compress, block_bytes):
    path = tmp_path / "a.nc"
    _raw_series(path)
    content, described = content_sha256(path), _described(path)
    aq.reencode_netcdf_atomic(path, 9, block_bytes=block_bytes)
    aq.reencode_netcdf_atomic(path, compress, block_bytes=block_bytes)
    assert aq.file_encoding(path) == aq.encoding_label(compress)
    assert content_sha256(path) == content and _described(path) == described
    assert not list(tmp_path.glob("*.reencode.*"))


def test_reencode_refuses_a_copy_with_other_content(tmp_path):
    path = tmp_path / "a.nc"
    _raw_series(path)
    before = aq.file_sha256(path)
    with pytest.raises(ValueError, match="change its content"):
        aq.reencode_netcdf_atomic(path, 4, expect="0" * 64)
    assert aq.file_sha256(path) == before and not list(tmp_path.glob("*.reencode.*"))


def test_file_encoding_names_zlib_without_shuffle(tmp_path):
    path = tmp_path / "a.nc"
    _raw_series(path)
    zlib = {v: {"zlib": True, "complevel": 4} for v in ("sst", "mask")}  # no shuffle
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        ds.load().to_netcdf(tmp_path / "b.nc", engine="h5netcdf", encoding=zlib)
    assert aq.file_encoding(tmp_path / "b.nc") == "zlib-4-noshuffle"


def test_compression_arguments_default_to_zlib_4():
    import argparse

    parser = argparse.ArgumentParser()
    aq.compression_arguments(parser)
    assert aq.compression(parser.parse_args([])) == 4
    assert aq.compression(parser.parse_args(["--compression-level", "9"])) == 9
    assert aq.compression(parser.parse_args(["--no-compress"])) is None


def test_ledger_reencodes_pieces_and_products_in_place(tmp_path):
    ledger = aq.Ledger(tmp_path / "ledger.json")
    piece, product = tmp_path / "piece.nc", tmp_path / "product.nc"
    _raw_series(piece)
    ledger.plan("p1", {"day": 1}, piece)
    ledger.record_piece("p1", piece, {"ok": True})
    old = ledger.get("p1")["piece"]["sha256"]
    _raw_series(product)
    # a product recorded before encodings were part of the identity, with a manifest
    ledger.record_product("prod", product, {"p1": old}, manifest="m.json")
    content = content_sha256(product)

    assert ledger.reencode_piece("p1", 4)
    assert not ledger.reencode_piece("p1", 4)  # already as asked
    entry = ledger.get("p1")
    assert entry["checks"] == {"ok": True} and entry["piece"]["sha256"] != old
    assert entry["piece"]["content_sha256"] == content_sha256(piece)
    new = {"p1": entry["piece"]["sha256"], "encoding": "zlib-4"}
    assert ledger.product_status("prod", product, new) == "reencode"
    assert ledger.refresh_product("prod", product, new, 4) == "reencoded"
    record = ledger.data["products"]["prod"]
    assert aq.file_encoding(product) == "zlib-4" and content_sha256(product) == content
    assert record["sha256"] == aq.file_sha256(product) and "manifest" not in record
    assert ledger.refresh_product("prod", product, new, 4) == "current"
    assert ledger.product_status("prod", product, {**new, "p1": "other"}) == "stale"
    assert aq.Ledger(tmp_path / "ledger.json").product_status("prod", product, new) == "current"


def test_a_run_cut_between_rewrite_and_record_keeps_the_piece(tmp_path, monkeypatch):
    ledger = aq.Ledger(tmp_path / "ledger.json")
    piece = tmp_path / "piece.nc"
    _raw_series(piece)
    ledger.plan("p1", {"day": 1}, piece)
    ledger.record_piece("p1", piece, {})

    def cut(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(aq.Ledger, "record_piece", cut)
    with pytest.raises(KeyboardInterrupt):
        ledger.reencode_piece("p1", 4)
    monkeypatch.undo()
    again = aq.Ledger(tmp_path / "ledger.json")  # the next run
    assert aq.file_encoding(piece) == "zlib-4"
    assert again.reusable("p1")  # not sent back to the service
    assert again.get("p1")["piece"]["sha256"] == aq.file_sha256(piece)


def test_a_rewrite_announced_but_not_done_leaves_products_current(tmp_path, monkeypatch):
    # Review H1: a cut after the new SHA-256 was saved but before the file was replaced.
    ledger = aq.Ledger(tmp_path / "ledger.json")
    piece, product = tmp_path / "piece.nc", tmp_path / "product.nc"
    _raw_series(piece)
    _raw_series(product)
    ledger.plan("p1", {"day": 1}, piece)
    ledger.record_piece("p1", piece, {})
    inputs = {"p1": ledger.get("p1")["piece"]["sha256"], "encoding": "none"}
    ledger.record_product("prod", product, inputs)

    def cut(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(aq.os, "replace", cut)
    with pytest.raises(KeyboardInterrupt):
        ledger.reencode_piece("p1", 4)
    monkeypatch.undo()
    again = aq.Ledger(tmp_path / "ledger.json")
    assert again.reusable("p1") and aq.file_encoding(piece) == "none"
    assert again.product_status("prod", product, inputs) == "current"


def test_reencodings_back_and_forth_keep_products_current(tmp_path):
    # Review H3: none -> zlib 4 -> none may give the first bytes again (a cycle of hashes).
    ledger = aq.Ledger(tmp_path / "ledger.json")
    piece, product = tmp_path / "piece.nc", tmp_path / "product.nc"
    _raw_series(piece)
    _raw_series(product)
    ledger.plan("p1", {"day": 1}, piece)
    ledger.record_piece("p1", piece, {})
    first = ledger.get("p1")["piece"]["sha256"]
    ledger.record_product("prod", product, {"p1": first, "encoding": "none"})
    seen = [first]
    for level in (4, None, 4, None):
        assert ledger.reencode_piece("p1", level)
        seen.append(ledger.get("p1")["piece"]["sha256"])
        for recorded in seen:  # a product built at any stage counts as built from this piece
            ledger.data["products"]["prod"]["inputs"]["p1"] = recorded
            now = {"p1": seen[-1], "encoding": "none"}
            assert ledger.product_status("prod", product, now) == "current"


def test_a_multi_file_piece_cut_after_the_rewrite_keeps_its_main_hash(tmp_path, monkeypatch):
    # Review H2: the piece's own SHA-256 follows the record of its first file.
    ledger = aq.Ledger(tmp_path / "ledger.json")
    paths = [tmp_path / "a.nc", tmp_path / "b.nc"]
    for path in paths:
        _raw_series(path)
    ledger.plan("p1", {"day": 1}, paths[0])
    ledger.record_piece("p1", paths, {})

    def cut(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(aq.Ledger, "record_piece", cut)
    with pytest.raises(KeyboardInterrupt):
        ledger.reencode_piece("p1", 4)
    monkeypatch.undo()
    again = aq.Ledger(tmp_path / "ledger.json")
    assert again.reusable("p1")
    piece = again.get("p1")["piece"]
    assert piece["sha256"] == aq.file_sha256(paths[0]) == piece["files"][piece["path"]]["sha256"]
    assert again.reencode_piece("p1", 4)  # the cut record is finished, nothing rewritten
    assert again.reencode_piece("p1", 4) is False


def test_two_rewrites_of_one_piece_at_once_do_not_collide(tmp_path):
    # Review H4: the ledger's methods are thread-safe; rewrites are serialised.
    from concurrent.futures import ThreadPoolExecutor

    ledger = aq.Ledger(tmp_path / "ledger.json")
    piece = tmp_path / "piece.nc"
    _raw_series(piece, days=400)
    ledger.plan("p1", {"day": 1}, piece)
    ledger.record_piece("p1", piece, {})
    content = content_sha256(piece)
    with ThreadPoolExecutor(4) as pool:
        done = list(pool.map(lambda _: ledger.reencode_piece("p1", 4), range(4)))
    assert done.count(True) == 1 and content_sha256(piece) == content
    assert ledger.reusable("p1") and not list(tmp_path.glob("*.reencode.*"))


def test_identical_files_keep_their_own_pending_rewrite(tmp_path, monkeypatch):
    # Re-review H5: two pieces with the same bytes, the first cut after its replacement.
    ledger = aq.Ledger(tmp_path / "ledger.json")
    paths = [tmp_path / "a.nc", tmp_path / "b.nc"]
    _raw_series(paths[0])
    paths[1].write_bytes(paths[0].read_bytes())
    for i, path in enumerate(paths):
        ledger.plan(f"p{i}", {"day": i}, path)
        ledger.record_piece(f"p{i}", path, {})

    def cut(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(aq.Ledger, "_confirm", cut)
    with pytest.raises(KeyboardInterrupt):
        ledger.reencode_piece("p0", 4)
    monkeypatch.undo()
    again = aq.Ledger(tmp_path / "ledger.json")
    assert again.reencode_piece("p1", 9)  # the other file, to another level, first
    assert again.reusable("p0") and aq.file_encoding(paths[0]) == "zlib-4"


def test_a_cut_while_updating_a_sidecar_runs_it_again(tmp_path):
    # Re-review H6: the callback (an ERA5 sidecar) must not be lost by a cut.
    ledger = aq.Ledger(tmp_path / "ledger.json")
    piece = tmp_path / "piece.nc"
    _raw_series(piece)
    ledger.plan("p1", {"day": 1}, piece)
    ledger.record_piece("p1", piece, {})
    calls = []

    def cut(path):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ledger.reencode_piece("p1", 4, cut)
    again = aq.Ledger(tmp_path / "ledger.json")
    assert again.reencode_piece("p1", 4, calls.append)  # nothing to rewrite, sidecar redone
    assert calls == [piece] and "rewriting" not in again.get("p1")
    assert again.reencode_piece("p1", 4, calls.append) is False and calls == [piece]
