# Data, preparation and provenance

[README](../README.md) · [Tutorials](TUTORIALS.md)

## Where the data live

xHW never downloads data, and neither do the cases: they read a data directory and verify what
they find. The directory is `data/` by default (ignored by Git); choose another with `--data-dir`,
or with `XHW_DATA_DIR` for the notebooks. Generated figures and reports go to `--output-dir`
(`XHW_OUTPUT_DIR` for the notebooks).

| Shell | Example |
|---|---|
| bash | `export XHW_DATA_DIR="$HOME/datasets/xhw"` |
| PowerShell | `$env:XHW_DATA_DIR = Join-Path $HOME 'datasets/xhw'` |

The five standard cases need the **compressed test datasets** (about 299 MB), fetched with
`examples/fetch_data.py` and described in [the README](../README.md#datasets-data-release-assets):

```text
data/
  oisst_peru_1991_2025.nc
  oisst_equatorial_1991_2025.nc
  ostia_peru_1991_2025.nc
  era5_peru_daily_1991_2025.nc
  era5_peru_hourly_sample_2023_03.nc
  glorys12v1_peru_decimated_1993_2023.nc
```

The sixth notebook, North Pacific OISST, needs an additional 445 MB personal
file, `oisst_npacific_1991_2025.nc`, outside the standard fetcher/release assets.
Its original and compressed checksums are registered in
[oisst_npacific.json](../examples/sources/oisst_npacific.json).

Each loader accepts either the compressed dataset or, when present, the original registered
files (OISST `oisst_v21_peru_1991_2025.nc`, ERA5 `era5_peru_hourly/2023.nc`, the 31 GLORYS files in
`glorys12v1_peru/`), and checks the registered hash first. Both routes give identical results.

| Case | Registered input and selection | Physical meaning |
|---|---|---|
| OISST | [Manifest](../examples/sources/oisst.json); NOAA OISST v2.1 AVHRR-only daily, final, 1991–2025, 12°S–4°S × 82°W–77°W; 32 × 20 centres, 0.25° | Daily analysed SST in °C; ocean support is any finite observation during the selected period; spherical cell areas |
| OSTIA | [Manifest](../examples/sources/test_datasets.json); Met Office OSTIA reprocessed, `METOFFICE-GLO-SST-L4-REP-OBS-SST` v202003, 160 × 100 centres, 0.05° | Foundation SST in kelvin, decoded from stored integers to hundredths of a degree |
| ERA5 | [Manifest](../examples/sources/era5.json); CDS single-levels, final, 1991–2025; 33 × 21 centres, 0.25°; local-day TX/TN, plus a 2023-03-01 to 04 hourly sample | Air temperature in K; land/ocean air support split at lsm 0.5; daily extrema of period-ending mx2t/mn2t |
| GLORYS | [Manifest](../examples/sources/glorys.json); `cmems_mod_glo_phy_my_0.083deg_P1D-m`, `GLOBAL_MULTIYEAR_PHY_001_030`; 31 annual thetao files, 1993–2023 | Assimilating ocean reanalysis, potential temperature in °C; coarse retained voxels; approximate m³ weights |

The original files are much larger than their compressed copies: OISST 16 MB, ERA5 daily 284 MB,
GLORYS several GB. Loading the selected arrays and producing several products takes more memory
than the files suggest; the OSTIA native arm needs about 4 GB. The GLORYS cases select, before
`.load()`, every eighth latitude/longitude and every second original level among the top 24. This
yields 13 × 8 × 12 potential voxels, with a static ocean mask.

## Reproduction versus a new retrieval

The manifests identify **specific registered files**, including an OISST
Icechunk snapshot `1H5D09GQTBHSNHGT67TG`, ERA5 hourly and daily checksums, and
per-year GLORYS checksums. A fresh NetCDF retrieval can differ in bytes because of
container metadata or revised product values. It is not automatically this exact
registered case. The strict loader reports a mismatch rather than changing the
manifest or a regression reference.

To analyse a fresh retrieval, open it in a separate workflow and create a new
`SourceRef` with its actual product/version/query/hash; use the stages shown in
the notebooks with that input. Treat its output as a new, unreviewed run. The
saved tutorials can still be read without any input files.

```python
from pathlib import Path
import hashlib
import xarray as xr
import xhw

path = Path("data/my_temperature.nc")
ds = xr.open_dataset(path)  # lazy opening; manage the dataset's lifetime
with path.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
source = xhw.SourceRef(dataset_id="my-product-id", variable="sst",
                       accessed="2026-10-10", product_status="unknown",
                       version="actual-version", query="actual subset",
                       checksum=f"sha256:{digest}")
```

Combine this source
with an explicit grid and baseline (see [the tutorial helpers](../examples/_data.py)
and [API](https://github.com/DanielCamarena/xhw/blob/main/docs/API.md)); `SourceRef` itself does not validate scientific quality.

## Optional acquisition scripts

Five scripts in `examples/` fetch the registered regional selections again (OISST, ERA5,
GLORYS) or add the high-resolution SST products used to study the effect of resolution (OSTIA,
ESA SST CCI). They
need network access, source-specific clients and, for ERA5 and GLORYS, a free account.
None of these tools is an xHW dependency or used by the offline test suite. Install
them in a separate acquisition environment:

```bash
python -m pip install -r examples/requirements-download.txt
```

| Script | Source and access | Writes under `--data-dir` |
|---|---|---|
| [download_oisst.py](../examples/download_oisst.py) | [Source Cooperative Icechunk store](https://data.source.coop/ocean-icechunks/noaa-oisst/oisst.icechunk) of NOAA OISST v2.1 (`daily` group, anonymous S3); snapshot pinned with `--snapshot` | `oisst_v21_<region>_<years>.nc`, `oisst_partial_<region>/`, `oisst_<region>_download_manifest.json` (`--region peru` or `equatorial`) |
| [download_era5.py](../examples/download_era5.py) | [CDS ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) via `cdsapi`; key in `~/.cdsapirc` ([setup](https://cds.climate.copernicus.eu/en/how-to-api)); accept the dataset licence once | `era5_peru_hourly/<year>.nc`, `lsm.nc`, `era5_peru_daily_<years>.nc`, `era5_download_manifest.json` |
| [download_glorys.py](../examples/download_glorys.py) | `cmems_mod_glo_phy_my_0.083deg_P1D-m` via `copernicusmarine` ([subset docs](https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset)); `copernicusmarine login` | `glorys12v1_peru/thetao_<year>.nc`, `download_manifest.json` |
| [download_ostia.py](../examples/download_ostia.py) | Met Office OSTIA reprocessed SST (`METOFFICE-GLO-SST-L4-REP-OBS-SST`, 0.05°, kelvin, 1981-10-01 to 2026-03-31) via `copernicusmarine`; same account as GLORYS | `ostia_peru/analysed_sst_<year>.nc`, `download_manifest.json` |
| [download_sst_cci.py](../examples/download_sst_cci.py) | ESA SST CCI L4 CDR v3.0.1 (0.05°, kelvin) from [CEDA](https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/); HTTP Range reads of the box, no account | `sst_cci_v3_<start>_<end>.nc`, `sst_cci_partial/`, `sst_cci_download_manifest.json` |

All three share one behaviour:

- **Without an action flag they only print the request.** Nothing is transferred or written.
- They never overwrite an existing target; use a fresh data directory.
- They record the request, client version, UTC time and SHA-256 of what they wrote.
  OISST and ERA5 also report whether the result matches the registered checksum.
- They never change `examples/sources/*.json` or start a tutorial.

```bash
python examples/download_oisst.py --data-dir data                 # print the request
python examples/download_oisst.py --data-dir data --download      # fetch 1991-2025
python examples/download_era5.py  --data-dir data                 # print 72 CDS requests
python examples/download_era5.py  --data-dir data --download      # hourly files + lsm
python examples/download_era5.py  --data-dir data --daily         # local-day TX/TN file
python examples/download_glorys.py --data-dir data --years 1993-2023 --dry-run
python examples/download_glorys.py --data-dir data --years 1993-2023 --download
python examples/download_ostia.py --data-dir data --years 1991-2025 --download   # ~12 MB per year
python examples/download_sst_cci.py --data-dir data --start 2020-01-01 --end 2020-01-31 --download
```

**OISST.** Each store chunk holds one global day, so about 0.5 MB is transferred per
day even for this small box: several GB for 1991–2025, although the saved file is
about 16 MB. Months are read with retries and each year is kept in `oisst_partial/`,
so an interrupted run resumes. Preliminary days are rejected. Longitudes 278–283°E
are shifted by −360. The default snapshot is the registered one,
`1H5D09GQTBHSNHGT67TG`. Check the [NOAA product page](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
for the product description.

**ERA5.** The script requests `t2m`, `mx2t`, `mn2t` and the land-sea mask over
area `[-4, -82, -12, -77]` (north, west, south, east), hourly UTC. Each year is two
half-year requests, because the CDS cost limit rejects a full year of three
variables. The first seven UTC hours of the following year are added so the last
local day at UTC−05:00 is complete. Finished pieces are kept, so a rerun only
repeats unfinished requests; `--workers` sets how many run at once (default 4).
`--daily` then calls `xhw.aggregate_daily` year by year: TX is the maximum of
period-ending `mx2t`, TN the minimum of `mn2t`. The `t2m` extrema `tx_t2m`/`tn_t2m`
are kept only as a contrast. CDS can return several NetCDF parts in a zip; the
script merges them, keeping names, units and timestamps.

**GLORYS.** The script requests `thetao` over 12°S–4°S × 82°W–77°W and 0–500 m,
one annual file at a time. `--dry-run` asks the service to check the request
without transferring data. The tutorial then keeps every eighth horizontal cell
and every second of the top 24 levels; see [Tutorials](TUTORIALS.md#glorys).

**OSTIA.** `analysed_sst` is stored as `int16` (scale 0.01, offset 273.15) in **kelvin**, on a
0.05° grid (160 × 100 cells in this box), with timestamps at 00:00. The service call takes about
25 s per year; the reprocessed series covers all of 1991–2025, so no near-real-time product is
needed. The dataset version is pinned with `--dataset-version` (default `202003`, the label in the
catalogue on 2026-10-11); see the
[product page](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011).

**ESA SST CCI.** The archive has one global file per day; the script reads only the bytes of the
box, with one request per day spread over several processes. It took 25 s for 3 days with 3 processes, so 1991–2021 needs
several hours. The CDR series ends on 2021-12-31 and a different interim series (ICDR) continues
it, so the default end date is 2021-12-31. Timestamps are at 12:00, unlike OSTIA's 00:00, so
align both to the calendar day before comparing. For 1991–2025 use OSTIA.

A fresh retrieval can differ in bytes from the registered file, because of
container metadata or revised product values. Treat it as a new, unreviewed run,
as described above. The development repository also checked the ERA5 route
against the ARCO-ERA5 copy and an Earth Data Hub mirror. Those comparisons are
not part of these scripts.

## Compressed test datasets and content hashes

A file checksum identifies bytes, and recompressing a file changes them. `examples/content_hash.py`
therefore also defines `content_sha256`, a hash of what the file stores: each variable's name,
dimensions, dtype, shape and values, plus the attributes that decide how values are decoded
(`scale_factor`, `add_offset`, `_FillValue`, `missing_value`, `units`, `calendar`). Compression,
chunking and descriptive attributes (`history`, `source`) do not change it; a different dtype,
scale or value does. Values are hashed as stored, so packed integers are hashed as integers.

`examples/make_test_datasets.py` builds the single-file NetCDF4 datasets from the
raw inputs, with zlib level 9 and byte shuffling and **no re-quantisation**:

| Dataset | Source | Operation |
|---|---|---|
| `oisst_peru_1991_2025.nc` | NOAA OISST v2.1, 0.25°, Peru box | Lossless recompression |
| `oisst_equatorial_1991_2025.nc` | NOAA OISST v2.1, 0.25°, equatorial Pacific (Niño 3.4 box) 5°S–5°N, 170°W–120°W | Lossless recompression (205 MB to 100 MB) |
| `ostia_peru_1991_2025.nc` | Met Office OSTIA, 0.05° | Yearly files joined in time |
| `era5_peru_daily_1991_2025.nc` | ERA5 local-day TX/TN | Lossless recompression |
| `era5_peru_hourly_sample_2023_03.nc` | ERA5 hourly, 2023-03-01 to 04 | Time selection, values unchanged |
| `glorys12v1_peru_decimated_1993_2023.nc` | GLORYS12V1 `thetao` | Tutorial decimation (every 8th cell, retained levels), joined in time |

Every case reads these compressed datasets, and none needs the original files. Credit lines and the
list of modifications are in [Attribution](ATTRIBUTION.md).

Each build step checks itself: a recompressed file must have the content hash of its source, and a
joined file must equal each of its parts array by array. The results are recorded in
`examples/sources/test_datasets.json` with file hash, content hash, size and origin. Check a
data directory, or any file, with:

```bash
python examples/verify_data.py --data-dir data         # every registered file found there
python examples/verify_data.py test-data/oisst_peru_1991_2025.nc
```

Each file is reported as `identical bytes`, `same content` (bytes differ, information is the
same) or `MISMATCH`. The registered OISST and ERA5 manifests also carry `content_sha256`, so a
recompressed copy of either verifies as `same content`.

## Processed products

Run any case with `--products-dir products/<case>` to write what it computed, as compressed NetCDF
files with a `products_manifest.json` (sizes, `sha256`, `content_sha256`, dimensions, variables):

| File | Holds | Reload with |
|---|---|---|
| `<label>_baseline.nc` | Reference, threshold and sample counts of the fit | `BaselineResult.from_dataset` |
| `<label>_mask.nc` | Boolean and flag layers of the mask: exceedance, membership, joined gap, validity, invalid reason, filled | `xarray.open_dataset` (not a full `XMask`) |
| `<label>_catalog.nc` | One row per event and cell | `EventCatalog.from_dataset` |
| `<label>_objects.nc` | Daily labels, track IDs, processed mask and tables | `ObjectField.from_dataset` |
| `<label>_lineage.nc` | The lineage graph: nodes, transitions, branches, spacetime events | `LineageGraph.from_dataset` |

The mask file omits `temperature`, `reference` and `threshold`: they are float64 arrays that recompute
from the data and the baseline, and on a 640-cell, 35-year field the mask was about 48 MB with them against 0.4 MB
without them. Measured sizes of the products of each case are in the README. All these NetCDF files
are release assets (ignored by git); only the manifests are in git.

## Preparation decisions to retain

Keep instantaneous t2m separate from forecast-derived mx2t/mn2t. Max sampled t2m
can exceed max mx2t; they are different data streams, not continuous extrema of
one identical series. Air over the ocean is not a substitute for volumetric
seawater temperature. A fixed UTC−05:00 offset is a declared day convention, not
a timezone with changing daylight-saving rules.

GLORYS retained-level thickness is approximated by interfaces halfway between
retained depths, the upper interface at zero and the bottom extended by one half
spacing. Coarse spherical cell area times thickness gives volume. These are not
original model layer bounds, conservative remapping or vertically integrated
high-resolution truth. Decimation can alter connectivity and small-scale metrics.
No mandatory regridding is added to detection.

Every run report records the source checksum, selection, specs, method parameters,
versions, UTC date and review state. OISST/ERA5 manifests retain their retrieval
queries; GLORYS records per-file hashes and decimation. A checksum identifies
bytes; it does not establish that the physical selection is appropriate.
