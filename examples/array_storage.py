# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Opt-in disk backing for large examples; values, dtypes and scientific Specs are unchanged.

Set XHW_MAPPED_ARRAYS=1 for the OSTIA/equatorial examples. Files stay under the
configured output directory and must remain while their live notebook uses them.
No cache file is overwritten or removed by these helpers.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import xhw


def enabled():
    return os.environ.get('XHW_MAPPED_ARRAYS') == '1'


def mapped_array(data, directory: Path, name: str, *, time_dim='time', block_days=64):
    """Copy one labelled array in bounded time blocks to a unique, read-only NPY file."""
    backing = data.data
    while backing is not None:
        if isinstance(backing, np.memmap) and backing.mode == 'r':
            return data.copy(deep=False)
        backing = getattr(backing, 'base', None)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=name+'-', suffix='.npy', dir=directory)
    os.close(descriptor)
    target = np.lib.format.open_memmap(filename, mode='w+', dtype=data.dtype, shape=data.shape)
    axis = data.get_axis_num(time_dim)
    for start in range(0, data.sizes[time_dim], block_days):
        index = [slice(None)]*data.ndim
        index[axis] = slice(start, start+block_days)
        target[tuple(index)] = data.isel({time_dim: slice(start, start+block_days)}).values
    target.flush()
    del target
    result = xr.DataArray(np.load(filename, mmap_mode='r', allow_pickle=False),
                          dims=data.dims, coords=data.coords,
                          attrs=dict(data.attrs), name=data.name)
    result.encoding = dict(data.encoding)
    return result


def mapped_product(product, directory: Path, name: str):
    """Retain product identity while backing all its time-dependent variables with files."""
    ds = product.to_dataset()
    updates = {key: mapped_array(value, directory, name+'-'+key, time_dim=product.grid.time_dim)
               for key,value in ds.data_vars.items()
               if product.grid.time_dim in value.dims}
    if not updates:
        return product
    return type(product).from_dataset(ds.assign(updates))


def reuse_temperature(dataset, mask, variable='sst'):
    """Use the same backing for the single native input series and mask temperature.

    Call only for the dataset supplied to detect, after any interpolation. Raw
    pre-interpolation observations must keep their own values and missing flags.
    """
    if mask.definition.combination != 'single' or mask.definition.daily_statistic != 'native':
        raise ValueError('Backing reuse requires a single native daily temperature')
    source = dataset[variable]
    result = xr.DataArray(mask.dataset.temperature.data, dims=source.dims, coords=source.coords,
                          attrs=dict(source.attrs), name=source.name)
    result.encoding = dict(source.encoding)
    dataset[variable] = result


def mapped_intensity(mask, directory: Path, name: str):
    """Compute the same signed difference in bounded blocks, without a full extra array."""
    if mask.definition.combination != 'single':
        raise ValueError('This teaching helper expects a single temperature component')
    ds, g = mask.dataset, mask.grid
    directory.mkdir(parents=True, exist_ok=True)
    dimensions = (g.time_dim, *g.spatial_dims)
    template = xhw.XMask.from_dataset(mask.to_dataset().isel({g.time_dim: slice(0,1)})).intensity()
    descriptor, filename = tempfile.mkstemp(prefix=name+'-', suffix='.npy', dir=directory)
    os.close(descriptor)
    shape = tuple(ds.sizes[d] for d in dimensions)
    dtype = np.result_type(ds.temperature.dtype, ds.reference.dtype)
    target = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
    for start in range(0, ds.sizes[g.time_dim], 64):
        part = ds.isel({g.time_dim: slice(start,start+64)})
        target[start:start+64] = (part.temperature-part.reference).transpose(*dimensions).values
    target.flush()
    del target
    coords = {key:value for key,value in template.coords.items() if g.time_dim not in value.dims}
    coords[g.time_dim] = ds[g.time_dim]
    return xr.DataArray(np.load(filename, mmap_mode='r', allow_pickle=False), dims=dimensions,
                        coords=coords, attrs=dict(template.attrs), name=template.name)
