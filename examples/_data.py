# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Small, explicit data and specification helpers shared by scripts and notebooks.

The helpers (specs, areas, synthetic inputs) are copied from the examples folder of the xHW
repository; the real-data loaders and the resolution study are specific to this project.

No downloader runs on import. All arrays used by these tutorials are loaded eagerly.
The registered files remain outside the distribution; source manifests ship beside us.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr
from content_hash import content_sha256

import xhw

MANIFESTS = Path(__file__).parent / "sources"
RADIUS_M = 6_371_000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(name: str) -> dict:
    return json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))


def checked_file(data_dir: Path, name: str, expected: str, content: str | None = None) -> Path:
    """A registered input: identical bytes, or the same stored content (e.g. recompressed)."""
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input {name!r} is absent. Supply --data-dir or XHW_DATA_DIR; "
            "see the README. Analysis never downloads data automatically."
        )
    if sha256(path) != expected and (content is None or content_sha256(path) != content):
        raise ValueError(
            f"SHA-256 mismatch for {name!r}; use the registered input or register "
            "a new source explicitly. Do not relabel it as this reproduced case."
        )
    return path


def open_loaded(path: Path) -> xr.Dataset:
    try:
        import h5netcdf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NetCDF tutorials need examples/requirements.txt (h5netcdf[h5py])."
        ) from exc
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        return opened.load()


def add_area(ds: xr.Dataset, lat: str, lon: str, step: float) -> xr.Dataset:
    radians = np.deg2rad(ds[lat].values)
    half = np.deg2rad(step / 2)
    area = RADIUS_M**2 * np.deg2rad(step) * np.abs(np.sin(radians + half) - np.sin(radians - half))
    ds["cell_area"] = (
        (lat, lon),
        np.repeat(area[:, None], ds.sizes[lon], axis=1),
        {"units": "m2"},
    )
    return ds


def surface_grid(
    *,
    lat: str = "lat",
    lon: str = "lon",
    domain: str = "ocean",
    domain_mask: str = "domain_mask",
    product: str = "synthetic",
    resolution: str = "1 degree",
) -> xhw.GridSpec:
    return xhw.GridSpec(
        support="surface",
        topology="structured",
        time_dim="time",
        spatial_dims=(lat, lon),
        coords=xhw.CoordRoles(lat=lat, lon=lon),
        calendar="standard",
        frequency="1D",
        time_axes="single",
        periodic_x=False,
        cell_measure=xhw.CellMeasure(kind="area", var="cell_area", units="m2"),
        vertical=None,
        domain=domain,
        domain_mask=domain_mask,
        domain_fraction=None,
        native=xhw.NativeGrid(product=product, resolution=resolution, is_native=True),
    )


def marine_definition(
    *, cold: bool = False, variable: str = "sea_surface_temperature"
) -> xhw.DefinitionSpec:
    return xhw.DefinitionSpec(
        profile="schlegel2021" if cold else "hobday2016",
        phenomenon="mcs" if cold else "mhw",
        variable=variable,
        combination="single",
        daily_statistic="native",
        day_definition="native",
        tail="lower" if cold else "upper",
        comparator="<" if cold else ">",
        threshold_kind="percentile",
        percentile=10.0 if cold else 90.0,
        absolute_threshold=None,
        min_duration_days=5,
        max_gap_days=2,
        join_rule="qualify_then_join",
        missing_policy="as_non_exceedance",
        season_months=None,
        category_scheme="schlegel2021" if cold else "hobday2018",
    )


def fixed_baseline(
    source: xhw.SourceRef,
    start: str = "1991-01-01",
    end: str = "2020-12-31",
    *,
    atmospheric: bool = False,
) -> xhw.BaselineSpec:
    return xhw.BaselineSpec(
        method="fixed_climatology",
        computed_on="temperature",
        reference_period=(start, end),
        source=source,
        doy_window_days=15 if atmospheric else 11,
        smoothing_window_days=None if atmospheric else 31,
        percentile_estimator="hyndman_fan_8" if atmospheric else "hyndman_fan_7",
        leap_day_policy="hobday_366_interp",
        min_samples=360 if atmospheric else 264,
        fit_mode="retrospective",
        available_until=None,
        in_base_correction="none",
    )


def synthetic_inputs():
    """A hand-constructed 15-day event: connected -> split for 5 days -> merge.

    Reference is a constant 20 degC in 1973-2002, so P90 and mean are both 20.
    Two five-cell bars remain hot throughout. Their three-cell connector is hot
    on days 1-5 and 11-15 only. Consequently its long gap cannot be temporally joined.
    A separate cell illustrates a joined cool day and another a filled missing day.
    """
    time = np.arange(np.datetime64("1973-01-01"), np.datetime64("2003-02-01"))
    lat, lon = np.arange(-12.0, -5.0), np.arange(-82.0, -73.0)
    temperature = np.full((len(time), len(lat), len(lon)), 20.0)
    first = int(np.flatnonzero(time == np.datetime64("2003-01-01"))[0])
    temperature[first : first + 15, 2, 2:7] = 26.0
    temperature[first : first + 15, 4, 2:7] = 26.0
    temperature[first : first + 5, 3, 3:6] = 26.0
    temperature[first + 10 : first + 15, 3, 3:6] = 26.0
    temperature[first : first + 13, 0, 0] = 25.0
    temperature[first + 6, 0, 0] = 20.0  # one cool day between qualifying runs
    temperature[first : first + 13, 0, 8] = 25.0
    temperature[first + 6, 0, 8] = np.nan  # interior interpolation, with a separate flag
    # Diagonally touching cells: two rank-1 patches, one rank-2 patch.
    temperature[first + 5 : first + 10, 6, 7] = 25.0
    temperature[first + 5 : first + 10, 5, 8] = 25.0
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), temperature, {"units": "degC"}),
            "domain_mask": (("lat", "lon"), np.ones((7, 9), bool)),
        },
        coords={"time": time.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    ds = add_area(ds, "lat", "lon", 1.0)
    source = xhw.SourceRef(
        dataset_id="synthetic:split-merge-v1",
        variable="sst",
        accessed="2026-10-10",
        product_status="final",
        checksum="sha256:" + hashlib.sha256(temperature.tobytes()).hexdigest(),
    )
    return (
        ds,
        surface_grid(),
        fixed_baseline(source, "1973-01-01", "2002-12-31"),
        replace(marine_definition(), category_scheme=None),
    )


# --- A parametric reconstruction of the Peru SST field ------------------------------------
# Constants fitted once to OISST v2.1 (12S-4S, 82W-77W, 1991-2025): a smooth quadratic surface in
# (lat + 8, lon + 79.5) for the mean and for each seasonal harmonic, a coastline, and the
# statistics of the anomaly. No daily value of OISST is stored: everything random comes from a
# seeded generator, so the field is similar in structure and never a copy.
_PERU_SURFACE = {  # coefficients of 1, y, x, y^2, x^2, y*x with y = lat + 8, x = lon + 79.5
    "mean": (19.7428, -0.6472, -1.3668, -0.0269, -0.175, -0.1821),
    "cos1": (0.8451, -0.1601, -0.3181, 0.0008, -0.0725, -0.0641),
    "sin1": (2.0754, -0.304, -0.5551, -0.0153, -0.0648, -0.0325),
    "cos2": (0.0983, -0.0509, 0.0541, -0.0175, -0.003, -0.036),
    "sin2": (0.6354, 0.0533, 0.0824, -0.0021, -0.0008, 0.0065),
}
_PERU_COAST = (-84.071, -0.5841)  # easternmost ocean longitude = a + b * latitude
_PERU_TREND = -0.022  # degC per year before the warm pulses; the fitted OISST trend is -0.0093
_PERU_SLOW = (
    0.992,
    0.70,
)  # domain-wide anomaly, slow part: AR(1) coefficient per day, sd (degC)
_PERU_FAST = (0.94, 0.60)  # domain-wide anomaly, fast part
_PERU_LOCAL_PHI, _PERU_LOCAL_SD, _PERU_LOCAL_SIGMA = (
    0.938,
    0.68,
    2.15,
)  # weather; sigma in cells
# Warm pulses (centre, peak degC, width in days) near the dates of the strong events of the period.
_PERU_PULSES = (
    ("1992-03-01", 2.6, 70),
    ("1997-09-01", 5.4, 130),
    ("2015-10-01", 3.8, 110),
    ("2017-03-01", 3.2, 40),
    ("2023-04-15", 4.8, 90),
)


def synthetic_peru(seed: int = 7) -> xr.Dataset:
    """Daily SST on the 32 x 20 Peru grid, 1991-2025: a fast, seeded reconstruction of OISST.

    Same grid, coast, seasonal cycle, trend and anomaly statistics as the registered OISST case,
    with a domain-wide red-noise anomaly, five warm pulses and spatially correlated weather
    noise. It is statistically similar, not a copy of any observation, and is rounded to
    hundredths of a degree like OISST. Generating it takes about a second.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    time = np.arange(np.datetime64("1991-01-01"), np.datetime64("2026-01-01"))
    lat = np.arange(-11.875, -4.0, 0.25)
    lon = np.arange(-81.875, -77.0, 0.25)
    la, lo = np.meshgrid(lat + 8.0, lon + 79.5, indexing="ij")
    basis = np.stack([np.ones_like(la), la, lo, la**2, lo**2, la * lo])
    surface = {k: np.tensordot(np.array(c), basis, axes=1) for k, c in _PERU_SURFACE.items()}
    days = (time - time[0]).astype(float)
    phase = 2 * np.pi * days / 365.25
    seasonal = (
        surface["mean"][None]
        + _PERU_TREND * days[:, None, None] / 365.25
        + surface["cos1"][None] * np.cos(phase)[:, None, None]
        + surface["sin1"][None] * np.sin(phase)[:, None, None]
        + surface["cos2"][None] * np.cos(2 * phase)[:, None, None]
        + surface["sin2"][None] * np.sin(2 * phase)[:, None, None]
    )
    common = np.zeros(time.size)
    for phi, sd in (_PERU_SLOW, _PERU_FAST):
        part = lfilter([np.sqrt(1 - phi**2)], [1, -phi], rng.standard_normal(time.size))
        common += sd * part / part.std()
    for centre, peak, width in _PERU_PULSES:
        offset = (time - np.datetime64(centre)).astype(float)
        common += peak * np.exp(-0.5 * (offset / width) ** 2)
    weather = gaussian_filter(
        rng.standard_normal((time.size, lat.size, lon.size)),
        sigma=(0, _PERU_LOCAL_SIGMA, _PERU_LOCAL_SIGMA),
        mode="nearest",
    )
    weather = lfilter([np.sqrt(1 - _PERU_LOCAL_PHI**2)], [1, -_PERU_LOCAL_PHI], weather, axis=0)
    weather *= _PERU_LOCAL_SD / weather.std()
    sst = np.round(seasonal + common[:, None, None] + weather, 2)
    ocean = lon[None, :] <= _PERU_COAST[0] + _PERU_COAST[1] * lat[:, None] + 0.125
    sst = np.where(ocean[None], sst, np.nan)
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst, {"units": "degC"}),
            "domain_mask": (("lat", "lon"), ocean),
        },
        coords={"time": time.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    return add_area(ds, "lat", "lon", 0.25)


def synthetic_peru_inputs(seed: int = 7):
    """`synthetic_peru` with its grid, fixed 1991-2020 baseline and marine MHW definition."""
    ds = synthetic_peru(seed)
    source = xhw.SourceRef(
        dataset_id=f"synthetic:peru-reconstruction-seed{seed}",
        variable="sst",
        accessed="2026-10-11",
        product_status="final",
        checksum="sha256:" + hashlib.sha256(ds["sst"].values.tobytes()).hexdigest(),
    )
    grid = surface_grid(product="synthetic reconstruction", resolution="0.25 degree")
    return ds, grid, fixed_baseline(source), marine_definition()


def _load_oisst_box(data_dir: Path, key: str):
    """A registered OISST box (`sources/<key>.json`), original or compressed, with its inputs."""
    rec = record(key)
    if (data_dir / rec["file"]).is_file():
        path = checked_file(data_dir, rec["file"], rec["sha256"], rec.get("content_sha256"))
    else:  # the compressed copy has the same stored content under another name
        path = checked_test_dataset(data_dir, key)
    ds = open_loaded(path)
    ds = add_area(ds, "lat", "lon", 0.25)
    ds["domain_mask"] = ds["sst"].notnull().any("time")
    source = xhw.SourceRef(
        dataset_id=rec["dataset_id"],
        variable="sst",
        accessed=rec["accessed"],
        product_status="final",
        version=rec["version"],
        checksum="sha256:" + rec["sha256"],
    )
    grid = surface_grid(product="NOAA OISST v2.1 AVHRR-only", resolution="0.25 degree")
    return ds, grid, fixed_baseline(source), marine_definition()


def load_oisst(data_dir: Path):
    """OISST off Peru, 12S-4S and 82W-77W, 0.25 degree."""
    return _load_oisst_box(data_dir, "oisst")


def load_oisst_equatorial(data_dir: Path):
    """OISST over the equatorial Pacific (Nino 3.4 box), 5S-5N and 170W-120W, 0.25 degree."""
    return _load_oisst_box(data_dir, "oisst_equatorial")


def load_era5(data_dir: Path):
    rec = record("era5")
    ds = open_loaded(checked_file(data_dir, rec["file"], rec["sha256"], rec.get("content_sha256")))
    ds = add_area(ds, "latitude", "longitude", 0.25)
    land = ds["lsm"] >= 0.5
    ds["land_mask"] = land & ds["tx"].notnull().any("time")
    ds["ocean_mask"] = ~land & ds["tx"].notnull().any("time")
    source = xhw.SourceRef(
        dataset_id=rec["dataset_id"],
        variable="mx2t,mn2t",
        accessed=rec["accessed"],
        product_status="final",
        checksum="sha256:" + rec["sha256"],
    )
    return ds, fixed_baseline(source, atmospheric=True)


def era5_profiles() -> dict:
    common = dict(
        phenomenon="ahw",
        day_definition="offset:-05:00",
        tail="upper",
        comparator=">",
        threshold_kind="percentile",
        absolute_threshold=None,
        min_duration_days=3,
        max_gap_days=0,
        join_rule="qualify_then_join",
        missing_policy="as_non_exceedance",
        season_months=None,
        category_scheme=None,
    )
    result = {}
    for variable, statistic in (("tx", "max"), ("tn", "min")):
        result[f"{variable.upper()}90"] = (
            xhw.DefinitionSpec(
                profile=f"perkins2013_{variable}90",
                variable=f"{variable}_daily",
                combination="single",
                daily_statistic=statistic,
                percentile=90.0,
                **common,
            ),
            variable,
        )
    for lag in (0, 1):
        result[f"TX+TN lag {lag}"] = (
            xhw.DefinitionSpec(
                profile="xhw_txtn90" if lag == 0 else "xhw_tx_then_tn90",
                variable="tx_daily&tn_daily",
                combination="and_daily",
                daily_statistic="native",
                percentile=None,
                components=(
                    xhw.DailyComponent(
                        variable="tx_daily",
                        daily_statistic="max",
                        percentile=90.0,
                        lag_days=0,
                    ),
                    xhw.DailyComponent(
                        variable="tn_daily",
                        daily_statistic="min",
                        percentile=90.0,
                        lag_days=lag,
                    ),
                ),
                **common,
            ),
            ("tx", "tn"),
        )
    return result


def hourly_era5(data_dir: Path) -> xr.Dataset:
    """Four hourly days (2023-03-01..04): from the registered file, else the shipped sample."""
    rec = record("era5")
    original = data_dir / "era5_peru_hourly" / "2023.nc"
    if original.is_file():
        path = checked_file(
            data_dir / "era5_peru_hourly",
            "2023.nc",
            rec["hourly_files_sha256"]["2023.nc"],
        )
    else:
        path = checked_test_dataset(data_dir, "era5_hourly_sample")
    # Only four days are materialized to explain the aggregation, with complete interior days.
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        time_dim = "valid_time" if "valid_time" in opened.dims else "time"
        sub = (
            opened[["mx2t", "mn2t", "t2m"]]
            .sel({time_dim: slice("2023-03-01", "2023-03-04")})
            .load()
        )
    return sub.rename({time_dim: "time"}) if time_dim != "time" else sub


def level_thickness(depth: np.ndarray) -> np.ndarray:
    interfaces = np.concatenate(([0.0], (depth[:-1] + depth[1:]) / 2))
    bottom = depth[-1] + (depth[-1] - interfaces[-1])
    return np.diff(np.append(interfaces, bottom))


def _glorys_thetao(data_dir: Path, rec: dict) -> xr.DataArray:
    """The retained voxels as float32: from the 31 registered files, else the decimated dataset."""
    folder = data_dir / "glorys12v1_peru"
    if all((folder / name).is_file() for name in rec["files"]):
        pieces = []
        for name, entry in rec["files"].items():
            path = checked_file(folder, name, entry["sha256"])
            with xr.open_dataset(path, engine="h5netcdf") as opened:
                pieces.append(
                    opened["thetao"]
                    .isel(
                        depth=rec["decimation"]["levels"],
                        latitude=slice(None, None, 8),
                        longitude=slice(None, None, 8),
                    )
                    .load()
                    .astype("float32")
                )
        return xr.concat(pieces, dim="time")
    return open_loaded(checked_test_dataset(data_dir, "glorys_decimated"))["thetao"].astype(
        "float32"
    )


def load_glorys(data_dir: Path):
    rec = record("glorys")
    thetao = _glorys_thetao(data_dir, rec)
    thetao.attrs = {"units": "degC", "standard_name": "sea_water_potential_temperature"}
    ds = add_area(thetao.to_dataset(), "latitude", "longitude", 8 / 12)
    ds["depth"].attrs = {"units": "m", "positive": "down"}
    ds["dz"] = ("depth", level_thickness(ds["depth"].values), {"units": "m"})
    ds["domain_mask"] = ds["thetao"].isel(time=0).notnull()
    assert bool((ds["thetao"].notnull().any("time") == ds["domain_mask"]).all())
    digest = hashlib.sha256(json.dumps(rec["files"], sort_keys=True).encode()).hexdigest()
    source = xhw.SourceRef(
        dataset_id=rec["dataset_id"],
        variable="thetao",
        accessed=rec["accessed"],
        product_status="final",
        version="GLORYS12V1 reanalysis",
        checksum="sha256:" + digest,
    )
    grid = xhw.GridSpec(
        support="volume",
        topology="structured",
        time_dim="time",
        spatial_dims=("depth", "latitude", "longitude"),
        coords=xhw.CoordRoles(lat="latitude", lon="longitude", depth="depth"),
        calendar="standard",
        frequency="1D",
        time_axes="single",
        periodic_x=False,
        cell_measure=xhw.CellMeasure(kind="area", var="cell_area", units="m2"),
        vertical=xhw.VerticalSpec(
            coordinate="depth",
            thickness="dz",
            thickness_units="m",
            time_dependent=False,
        ),
        domain="ocean",
        domain_mask="domain_mask",
        domain_fraction=None,
        native=xhw.NativeGrid(
            product="GLORYS12V1 reanalysis",
            resolution="1/12 degree, decimated every 8th cell (2/3 degree)",
            is_native=True,
        ),
    )
    return (
        ds,
        grid,
        fixed_baseline(source, "1993-01-01", "2022-12-31"),
        marine_definition(variable="sea_water_potential_temperature"),
    )


def checked_test_dataset(data_dir: Path, key: str) -> Path:
    """A compressed test dataset, accepted if its bytes or its stored content match the manifest."""
    entry = record("test_datasets")["datasets"][key]
    path = data_dir / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Test dataset {entry['file']!r} is absent. Supply --data-dir with the compressed "
            "test datasets (see the README). Analysis never downloads data automatically."
        )
    if sha256(path) != entry["sha256"] and content_sha256(path) != entry["content_sha256"]:
        raise ValueError(
            f"{entry['file']!r} matches neither the registered bytes nor the registered content; "
            "do not relabel it as this test dataset."
        )
    return path


def _surface_arm(sst: xr.DataArray, step: float) -> xr.Dataset:
    ds = sst.rename("sst").to_dataset()
    ds["sst"].attrs = {"units": "degC"}
    ds = add_area(ds, "lat", "lon", step)
    ds["domain_mask"] = ds["sst"].notnull().any("time")
    return ds


def load_resolution_study(data_dir: Path, min_ocean_subcells: int = 13):
    """The three arms of the OISST/OSTIA resolution study on one 32 x 20 box (degC).

    ``oisst``: NOAA OISST at 0.25 degree. ``ostia_coarse``: OSTIA averaged over the 5 x 5
    sub-cells of every OISST cell, an ocean cell needing at least `min_ocean_subcells` of 25
    valid ones (13 = a majority). ``ostia_fine``: OSTIA at its native 0.05 degree. OSTIA is
    decoded from its stored integers as exact hundredths of a degree (scale 0.01, offset
    273.15 K), so no float32 rounding enters the comparison.
    """
    oisst_path = checked_test_dataset(data_dir, "oisst")
    ostia_path = checked_test_dataset(data_dir, "ostia")
    oisst = open_loaded(oisst_path)["sst"]
    # OISST stamps are at 12:00 and OSTIA's at 00:00: compare them by calendar day.
    oisst = oisst.assign_coords(time=oisst["time"].dt.floor("D").values)
    with xr.open_dataset(ostia_path, engine="h5netcdf") as decoded:
        coords = decoded[["latitude", "longitude", "time"]].load()
    with xr.open_dataset(
        ostia_path,
        engine="h5netcdf",
        decode_cf=False,
        mask_and_scale=False,
        decode_times=False,
    ) as raw:
        attrs = raw["analysed_sst"].attrs
        ints = raw["analysed_sst"].values
    assert abs(float(attrs["scale_factor"]) - 0.01) < 1e-6
    assert abs(float(attrs["add_offset"]) - 273.15) < 1e-4
    celsius = np.where(ints == attrs["_FillValue"], np.nan, ints.astype("float64") * 0.01)
    fine = xr.DataArray(
        celsius,
        dims=("time", "lat", "lon"),
        coords={
            "time": coords["time"].values,
            "lat": coords["latitude"].values.astype("float64"),
            "lon": coords["longitude"].values.astype("float64"),
        },
    )
    valid = fine.notnull().coarsen(lat=5, lon=5).sum()
    coarse = fine.coarsen(lat=5, lon=5).mean().where(valid >= min_ocean_subcells)
    for name in ("lat", "lon"):
        assert float(abs(coarse[name].values - oisst[name].values).max()) < 1e-4, name
    coarse = coarse.assign_coords(lat=oisst["lat"].values, lon=oisst["lon"].values)
    assert (coarse["time"].values == oisst["time"].values).all(), "calendar days differ"
    arms = {
        "oisst": _surface_arm(oisst, 0.25),
        "ostia_coarse": _surface_arm(coarse, 0.25),
        "ostia_fine": _surface_arm(fine, 0.05),
    }
    paths = {"oisst": oisst_path, "ostia": ostia_path}
    return arms, paths
