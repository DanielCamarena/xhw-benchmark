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

The scripts described here are those of code version `v0.2.0` ([changes](../CHANGELOG.md)). They
do not change the registered data (`data-v1`): a new retrieval is a new run, as explained above.

Six scripts in `examples/` fetch the registered regional selections again (OISST, ERA5,
GLORYS), add the high-resolution SST products used to study the effect of resolution (OSTIA,
ESA SST CCI), or add ERA5 pressure levels. They need network access, source-specific clients
and, for ERA5, GLORYS and OSTIA, a free account. None of these tools is an xHW dependency or
used by the offline test suite. Install them in a separate acquisition environment:

```bash
python -m pip install -r examples/requirements-download.txt
```

| Script | Source and access | Writes under `--data-dir` |
|---|---|---|
| [download_oisst.py](../examples/download_oisst.py) | [Source Cooperative Icechunk store](https://data.source.coop/ocean-icechunks/noaa-oisst/oisst.icechunk) of NOAA OISST v2.1 (`daily` group, anonymous S3); snapshot pinned with `--snapshot` | yearly pieces per snapshot and box, `oisst_ledger.json`, one product and one manifest per box |
| [download_era5.py](../examples/download_era5.py) | [CDS ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) via `cdsapi`; key in `~/.cdsapirc` ([setup](https://cds.climate.copernicus.eu/en/how-to-api)); accept the dataset licence once | `era5_single/<region>/<variable>/<year>/` monthly files with sidecars and `ledger.json`; `era5_<region>_daily_*.nc`; with `--consolidate-years`, `era5_<region>_hourly/<year>.nc` |
| [download_era5_pressure.py](../examples/download_era5_pressure.py) | [CDS ERA5 pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels); same account and key | `era5_pressure/<region>/<variable>/<year>/`, `era5p_<region>_daily_*.nc` |
| [download_glorys.py](../examples/download_glorys.py) | `cmems_mod_glo_phy_my_0.083deg_P1D-m` via `copernicusmarine` ([subset docs](https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset)); `copernicusmarine login` | `glorys12v1_<region>/thetao_<year>.nc`, `ledger.json`, `glorys_manifest_<hash>_<UTC>.json` |
| [download_ostia.py](../examples/download_ostia.py) | Met Office OSTIA reprocessed SST (`METOFFICE-GLO-SST-L4-REP-OBS-SST` v202003, 0.05°, kelvin; the end of the series is asked from the service) via `copernicusmarine`; same account as GLORYS | `ostia_<region>[_<policy>]/analysed_sst_<year>.nc`, ledger and manifest |
| [download_sst_cci.py](../examples/download_sst_cci.py) | ESA SST CCI L4 CDR v3.0.1 (0.05°, kelvin) from [CEDA](https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/); HTTP Range reads of the box, no account | `sst_cci_partial/<hash>/<year>/`, `sst_cci_ledger_<hash>.json`, the combined file and its manifest |

All six share one behaviour, implemented once in `examples/_acquire.py`:

- **Without an action flag they only print the plan.** Nothing is transferred or written.
- **The data contain the requested box.** `--region` (`peru`, `equatorial`, `npacific`) or
  `--box S N W E` give the box; `--box-policy enclose` (default) asks for grid nodes on or
  beyond every edge, `cover-cells` for cells reaching every edge, and `inside` for the
  nodes inside only, which is how the registered inputs were requested. The CDS area is
  rounded outward to the grid, Copernicus Marine is asked for `outside`, and OISST and SST
  CCI select index ranges locally. The extent actually received is checked and recorded in
  the manifest; a run stops if it fails the policy. A box crossing the antimeridian
  (`npacific`, or `--box 25 55 160 -130`) is requested in two parts, each checked, and
  joined west to east keeping the stored longitudes (…, 179.75, 180, −179.75, …).
- **Every request has an identity** (a hash of the request) and a state in a ledger. A file
  is reused only if its SHA-256 is still the recorded one, so an extended or changed request
  never picks up a file of another one, and an interrupted run resumes where it stopped.
  Files are written to a temporary name and renamed when complete.
- **Damaged answers leave evidence.** Size, SHA-256 and error are recorded, the file is
  deleted and the request is retried once; `--keep-failed` moves it to quarantine instead.
  Only transient errors (network, 5xx, 429, some icechunk kinds) are retried; every failure
  is reported at the end and the exit code is not 0 while something is missing.
- **Manifests** are named after the request and the UTC time, so runs never overwrite each
  other. They hold the request, `sha256` and `content_sha256` of each file, software
  versions, the retrieval window and the box check. Where a registered input exists (OISST,
  ERA5 single levels, GLORYS, OSTIA) they compare with it by `content_sha256`.
- They never change `examples/sources/*.json` or start a tutorial.

```bash
python examples/download_oisst.py --data-dir data                       # print the plan
python examples/download_oisst.py --data-dir data --region peru equatorial --download
python examples/download_era5.py  --data-dir data                       # print the plan
python examples/download_era5.py  --data-dir data --download            # hourly files + lsm
python examples/download_era5.py  --data-dir data --daily               # local-day TX/TN file
python examples/download_era5.py  --data-dir data --years 2020-2020 --variables t2m d2m u10 v10 \
    --derived ws10 rh2m at2m --download    # the hours the derived products need
python examples/download_era5.py  --data-dir data --years 2020-2020 --variables t2m d2m u10 v10 \
    --derived ws10 rh2m at2m --daily       # the daily derived products
python examples/download_era5_pressure.py --data-dir data --years 2020-2020 --download
python examples/download_glorys.py --data-dir data --years 1993-2023 --dry-run
python examples/download_ostia.py --data-dir data --start 1991-01-01 --end 2025-12-31 --download
python examples/download_sst_cci.py --data-dir data --start 2020-01-01 --end 2020-01-31 --download
```

**OISST.** Each store chunk holds one global day of about 0.67 MB, transferred whole even for a
small box: about 8.5 GB for 1991–2025. Several boxes (`--region peru equatorial`) are read in one
pass, so the transfer does not grow with the number of boxes. The snapshot is part of each
piece's identity, so pieces of two snapshots are never mixed; the default snapshot is the
registered one, `1H5D09GQTBHSNHGT67TG`. Coverage and preliminary days are checked before any
`sst` is read, and a month with preliminary days is never saved. NOAA's attributes are kept;
`source`, `store`, `store_snapshot` and `history` are added. A box's pieces (one per year) are
joined one at a time, so memory holds one year of the box. `--no-join` (added after `v0.2.0`)
keeps the validated yearly pieces under `oisst_partial/` and writes no product or manifest; a
later run without it joins them without transferring again. Check the
[NOAA product page](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) for the
product description.

**ERA5 single levels.** By default the script requests `t2m`, `mx2t`, `mn2t` and the land-sea
mask, monthly and grouped by GRIB step type, under the CDS cost limit (121 000 =
6 × variables × hours × days). `--variables` takes any variable of its catalogue (surface
pressure, wind, dew point, radiation and heat fluxes, precipitation, gusts). The hours of the
neighbouring UTC days that the local day needs are derived from `--day-definition`. The
registered 72-request plan is reproduced with
`--chunk-period half-year --request-grouping all --boundary-hours 7`. Preliminary ERA5T data
(`expver` 0005) are rejected unless `--allow-era5t`, which marks them in every file and
product; data without `expver` need `--accept-unknown-expver-before DATE` and a
justification. `--daily` reads only the files of the current plan and calls
`xhw.aggregate_daily` month by month: TX is the maximum of period-ending `mx2t`, TN the minimum
of `mn2t`; the `t2m` extrema `tx_t2m`/`tn_t2m` are kept as a contrast (`--no-contrast` omits
them). Other variables get one daily file per statistic; accumulated ones become mean fluxes
in W m-2, or a daily total for `tp`. `--derived` adds the products of `examples/_derived.py`:
wind speed and direction, downward radiation, net heat flux, three versions of the daily mean
temperature for the EHF, humidity, apparent temperature, the NWS heat index and the humidex,
each with its formula, constants and order of computation in the attributes. UTCI, WBGT and
the mean radiant temperature (`--derived utci2m wbgt2m mrt`) use ECMWF's `thermofeel`
(optional, in `requirements-download.txt`) and need `t2m d2m u10 v10 ssrd ssr strd str fdir`;
the cosine of the solar zenith angle is computed by the script, averaged over the sunlit part
of the hour of each radiation accumulation. UTCI comes with the daily share of hours outside
the validity domain of its polynomial (`utci2m_outside_domain_fraction`); WBGT is an estimate
(psychrometric wet bulb, not the instrument's natural wet bulb) kept as a diagnostic. The CDS
name and units of `fdir` were not checked against a real answer.
`--consolidate-years` also writes one hourly file per year with the three default variables
(`era5_<region>_hourly/<year>.nc`), the layout of the registered input.

*Memory.* Each daily product (and each consolidated year) is built in blocks of local days
(`--daily-block month`, the default, or `year`): the block's hours are read, aggregated and
appended to the file before the next block, so memory holds about one block of one variable.
One month at 0.25° over 260 000 cells (150°E–30°W, 30°N–60°S) is 0.76 GB per variable in
`float32`; the measured peak was about four times the hourly data of the block. The block size
does not change any value: month and year blocks give the same `content_sha256`.

Pieces are brought to the encoding asked (see [Compression](#compression)) by `--download`,
with their sidecars; products by `--daily` and `--consolidate-years`.

**ERA5 pressure levels.** By default `temperature` at 850, 700 and 500 hPa, one request per
variable, month and group of levels (cost limit 60 000). ERA5 gives values below the terrain,
so `--daily` masks every hour before aggregating: a sample counts only where the surface
pressure is finite and `100 × level_hPa ≤ sp_Pa`. The hourly surface pressure comes from
`download_era5.py --variables sp` over the same box and period (`--surface-pressure` is its
folder); without it no daily file is built. Each level is a separate 2-D surface; detection on
levels is not part of these examples. Daily files are built by blocks, as for single levels.

**GLORYS.** The script requests `thetao` over 12°S–4°S × 82°W–77°W and 0–500 m
(`--max-depth`), one calendar year per request, dataset version `202311`. `--dry-run` asks the
service to check the request without transferring data; its size estimate overstates the
transfer about twelve times. A year the service covers only in part is rejected unless
`--allow-partial` keeps it as `thetao_<year>_partial.nc`. The tutorial then keeps every eighth
horizontal cell and every second of the top 24 levels; see [Tutorials](TUTORIALS.md#glorys).
A box crossing the antimeridian is joined slab by slab of days (bounded memory). The service
sends uncompressed files; each validated year (or part) is then rewritten with the encoding
asked.

**OSTIA.** One request for the whole period (`--block-days` splits it), split locally into one
file per year. `analysed_sst` is stored as `int16` (scale 0.01, offset 273.15) in **kelvin**, on
a 0.05° grid, with timestamps at 00:00. The dataset version is pinned (`202003`); the end of the
series is asked from the service, and a year it covers only in part needs `--allow-partial`.
Each year is written month by month from the blocks, values as stored, so memory holds one
month of the box. The blocks are deleted after splitting; with `--keep-blocks` they follow the
encoding asked like the years.
See the [product page](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011).

**ESA SST CCI.** The archive has one global file per day; the script reads only the bytes of the
box, one request per day over several processes: about 11 300 requests for 1991–2021. Three days
took 25 s with three processes in revision 6, so the full period is expected to take hours
(not measured with the new code). The CDR
series ends on 2021-12-31; the interim series (ICDR) that continues it is used only with
`--include-icdr`, and the manifest gives the series of each day. `--no-join` (added after
`v0.2.0`) keeps the validated daily parts and writes no combined file or manifest; a later run
without it combines them without fetching again. A reused day is checked against
the server's ETag (`--no-check-etag` skips it). Timestamps are at 12:00, unlike OSTIA's 00:00, so
align both to the calendar day before comparing. For 1991–2025 use OSTIA. The daily parts are
combined 31 days at a time into one file (values as stored), so memory holds a
month of the box instead of the whole period.

### Compression

Every downloader writes the NetCDF files it keeps (pieces and products) with zlib and byte
shuffle, level 4 by default (`--compression-level 1-9`); `--no-compress` writes them
uncompressed. On the short F3 downloads, level 4 left files at 32–68 % of their uncompressed
size. Only the layout changes: no option quantises values (no new `int16` packing), and
`content_sha256` is the same at any setting.

The encoding is part of each product's identity in the ledger. A run with another setting
rewrites in place, block by block, the files kept with the previous one: pieces first, then
the products built from them, which are not rebuilt and need neither the pieces nor the network
(OSTIA years whose blocks were deleted are rewritten alone). Each rewrite must give the
recorded `content_sha256` before it replaces the file. Products written before the encoding
was recorded are treated in the same way on the next run. OSTIA and GLORYS answers arrive
uncompressed; the files kept from them are rewritten with the encoding asked.

A fresh retrieval can differ in bytes from the registered file, because of
container metadata or revised product values. Treat it as a new, unreviewed run,
as described in [Reproduction versus a new retrieval](#reproduction-versus-a-new-retrieval). The development repository also checked the ERA5 route
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

`tools/regress_data_v1.py` checks that the current code still rebuilds the registered inputs
from the historical originals (the hourly ERA5 year files and the GLORYS, OSTIA and OISST files
kept with the development repository), without network:

```bash
python tools/regress_data_v1.py --originals ../xhw/data --work-dir <scratch folder>
```

It checks the `sha256` of every original against its registration, replays the CDS from the
ERA5 year files so that `download_era5.py` runs unchanged (plan, validation, ledger, `--daily`)
and compares the daily file with the registered `content_sha256`, rebuilds the six datasets of
`make_test_datasets.py`, and reports the OISST, GLORYS, OSTIA and SST CCI downloaders as
pending: their originals are the services' answers, so only a real download checks them. It
needs about 3 GB of scratch space and runs for about 25 minutes. Run on 2026-11-03 with xHW
0.0.1: 115 checks passed, none failed (`data_v1_regression.json` in the work folder).
`tests/test_regress_data_v1.py` runs the same regression when `XHW_DATA_V1_ORIGINALS` names
the originals folder, and is skipped otherwise.

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
