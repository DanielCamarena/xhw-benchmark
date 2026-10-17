# xhw-benchmark

The **full** working mode of [xHW](https://github.com/DanielCamarena/xhw): real data from several
sources, compressed NetCDF test datasets with their downloaders, the processed products, and the
scripts and notebooks that produce them. The xHW repository itself stays light and offline:

| | Fast mode: `xhw` | Full mode: `xhw-benchmark` (this repository) |
|---|---|---|
| Data | Synthetic only, generated in seconds | Real: OISST (Peru and equatorial Pacific), OSTIA, ERA5, GLORYS (and ESA SST CCI on request) |
| Network | None | Only to fetch the datasets (or to re-download from the sources) |
| Size | About 10 MB of source, docs and tests | Code, executed notebooks and figures; standard `data-v1` assets: datasets 299 MB, processed products about 329 MB |
| Time | Minutes | Real examples take seconds to minutes; larger domains need substantial memory or optional disk backing (see reproduction guide) |
| Purpose | Learn and test the API | Check results on real inputs and study their sensitivity |

## Quick start

Use xHW `0.0.1` with the optional example requirements. See
[reproduction instructions](docs/REPRODUCIBILITY.md) for environments and product checks.

```bash
python -m pip install ../xhw
python -m pip install -r examples/requirements-interactive.txt
python examples/fetch_data.py --release-url https://github.com/DanielCamarena/xhw-benchmark/releases/download/data-v1
python examples/verify_data.py --data-dir data    # every file: identical bytes or same content
python examples/oisst_peru.py --data-dir data --output-dir outputs/oisst
```

`fetch_data.py` copies from a local folder too (`--from-dir`) and keeps a file only if its bytes
or its stored values match the manifest. Nothing is ever overwritten.

## Cases

| Case | Script | Notebook | What it shows | Recorded run |
|---|---|---|---|---|
| OISST Peru | `examples/oisst_peru.py` | `oisst_peru.ipynb` | MHW and MCS, fixed versus detrended baseline, objects, genealogy (0.25°) | `docs/run_records/oisst.json` |
| OSTIA vs OISST | `examples/ostia_peru.py` | `ostia_peru.ipynb` | Product and resolution effects in three arms: OISST, OSTIA at 0.25°, OSTIA at 0.05° | `products/ostia_resolution/` |
| OISST equatorial Pacific | `examples/oisst_equatorial.py` | `oisst_equatorial.ipynb` | Niño 3.4 box, 8 000 ocean cells: MHW and MCS, fixed versus detrended baseline, and the 2015–16 El Niño as one large object with its genealogy | `docs/run_records/oisst_equatorial.json` |
| OISST North Pacific | `examples/oisst_npacific.py` | `oisst_npacific.ipynb` | Large domain and Blob objects in 2012–2018; separately registered personal input, outside standard release assets | `docs/run_records/oisst_npacific.json` |
| ERA5 Peru | `examples/era5_peru.py` | `era5_peru.ipynb` | Hourly to daily aggregation, TX90, TN90 and joint TX+TN over land and ocean air | `docs/run_records/era5.json` |
| GLORYS volume | `examples/glorys_volume.py` | `glorys_volume.ipynb` | Voxel detection, physical vertical connectivity, volume-weighted genealogy | `docs/run_records/glorys.json` |

Written guides for each case, with figures, are in [`docs/TUTORIALS.md`](docs/TUTORIALS.md). The
case reports describe results for the registered compressed datasets below. Reproducing a number is not a scientific review: the real-data
summaries are pending review, except the OISST temporal reference, which has prior approval.

## Datasets (`data/`, release assets)

| File | Source | Size | Obtained by |
|---|---|---|---|
| `oisst_peru_1991_2025.nc` | NOAA OISST v2.1 AVHRR-only daily SST, 0.25° | 5.1 MB | `download_oisst.py` |
| `oisst_equatorial_1991_2025.nc` | The same product over the equatorial Pacific (Niño 3.4 box), 5°S–5°N, 170°W–120°W | 100 MB | `download_oisst.py --region equatorial` |
| `ostia_peru_1991_2025.nc` | Met Office OSTIA reprocessed, dataset `METOFFICE-GLO-SST-L4-REP-OBS-SST` v202003, 0.05°, kelvin | 110 MB | `download_ostia.py` |
| `era5_peru_daily_1991_2025.nc` | ERA5 hourly single levels, local-day (UTC−05:00) TX/TN | 70 MB | `download_era5.py` then `--daily` |
| `era5_peru_hourly_sample_2023_03.nc` | ERA5 hourly, 2023-03-01 to 04 (aggregation demonstration) | 0.4 MB | `download_era5.py` |
| `glorys12v1_peru_decimated_1993_2023.nc` | GLORYS12V1 `thetao`, every 8th cell, 12 retained levels | 13 MB | `download_glorys.py` |

Credit lines and the list of modifications: [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).

All are lossless: variables keep the stored integers, dtype, scale and offset of the source and
only gain zlib level 9 with byte shuffling. They were built by `examples/make_test_datasets.py`,
which checks each result before accepting it, and are registered in
`examples/sources/test_datasets.json` with two hashes:

- `sha256`, the hash of the file bytes, which detects a damaged download;
- `content_sha256`, a hash of the stored values and their decoding attributes (see
  `examples/content_hash.py`), which stays the same when a file is recompressed and changes if
  a value, dtype or scale changes.

`verify_data.py` reports each file as `identical bytes`, `same content` or `MISMATCH`.

## Processed products (`products/`)

All xHW entity schemas use version `0.1`. Intensity graphs store valid cell count
and valid physical measure per patch; missing observations remain distinct from zero.
See [reproduction instructions](docs/REPRODUCIBILITY.md) and
[plotting and exploration](docs/PLOTTING.md).

The North Pacific example additionally requires the 445 MB personal file registered
in [oisst_npacific.json](examples/sources/oisst_npacific.json). The standard fetcher
does not supply it. See [the case guide](docs/TUTORIALS.md#north-pacific-oisst)
for selection and memory requirements.

Every case can write what it computed with `--products-dir products/<case>`: the fitted baseline, the
mask layers, the per-cell event catalog, the objects and the lineage graph, as compressed NetCDF files,
with a `products_manifest.json` of sizes and hashes (`examples/_report.py::export_products`). Mask files
keep only their boolean and flag layers. Typical product sizes for the registered selections:

| Case | Files | Size | Largest file |
|---|---:|---:|---|
| OISST Peru (`oisst_peru`) | 10 | 11.2 MB | catalog of MCS, 2.3 MB |
| OSTIA vs OISST (`ostia_resolution`) | 15 | 90.6 MB | OSTIA 0.05° baseline, 39.1 MB |
| OISST equatorial Pacific (`oisst_equatorial`) | 10 | 192.7 MB | catalog of MCS, 45.5 MB |
| ERA5 Peru (`era5_peru`) | 16 | 23.7 MB | catalog of one profile (ocean air, TX+TN lag 0), 3.3 MB |
| GLORYS volume (`glorys_volume`) | 5 | 11.1 MB | MHW baseline, 3.6 MB |
| **Total** | 56 | **329.3 MB** | no file above 50 MB |

The equatorial case is the largest because its catalogs hold every per-cell event of the 35 years (376 000 heatwaves, 580 000 cold spells and 374 000 detrended heatwaves). 

All of these are release assets; the manifests and each case's `report.json` and figures are in git.
Exporting products does not change any metric: the cases reproduce their recorded numbers with the
option on.

## Downloaders (`examples/download_*.py`)

Every downloader prints its request by default, never overwrites a file and writes a manifest of
what it fetched. Install `examples/requirements-download.txt` in a separate environment.

| Script | Source | Account |
|---|---|---|
| `download_oisst.py` | Source Cooperative Icechunk store of NOAA OISST (pinned snapshot) | none; about 100 s per year |
| `download_ostia.py` | Copernicus Marine, OSTIA reprocessed | free Copernicus Marine account; 13 min for 1991–2025 |
| `download_era5.py` | Copernicus Climate Data Store | free CDS account and licence |
| `download_glorys.py` | Copernicus Marine, GLORYS12V1 | free Copernicus Marine account |
| `download_sst_cci.py` | CEDA archive, ESA SST CCI v3.0.1 (CDR to 2021-12-31, interim series after) | none; hours for 31 years |

Source-by-source notes (products, versions, cautions, measured times) are in the development
guides of the xHW project, `docs/guides/`: OSTIA, ESA SST CCI, GLORYS, ERA5, and the CESM2-LE and
EERIE sources that xHW cannot use (monthly output; unstructured grid).

## Git or release assets?

Git keeps every version of every committed file forever, and a clone downloads all of it. A release
asset is a file attached to a tag, outside the history: it is fetched one file at a time,
can be up to 2 GiB, and can be replaced. So:

- **In git:** code, notebooks, manifests, `report.json` and tables, figures, the run records.
- **As release assets:** every NetCDF (the datasets above and any large processed product).
- The manifests in git hold the hashes, so an asset can always be checked and replaced safely.

GitHub allows less than 2 GiB per file, up to 1000 files per release and no total limit; a
repository is best kept under 1 GB. Zenodo, if a DOI is wanted, takes 100 files and 50 GB per record.

## Layout

```
examples/    scripts, notebooks, helpers (_data.py, _report.py, plotting.py), downloaders, tools
  sources/   manifests: oisst.json era5.json glorys.json test_datasets.json
data/        the datasets (ignored by git; fetched or built)
products/    processed outputs per case: report.json, figures (git); large NetCDF (assets)
docs/        TUTORIALS.md, DATA.md, run records and figures
tests/       content hash, comparison helpers, input contracts
```

`_data.py`, `_report.py` and `plotting.py` are copies of the helpers of the xHW `examples/`
folder, with the real-data loaders added; keep them in step by hand.

## Status and limits

- The datasets are modified copies of NOAA, Copernicus Climate Change Service and Copernicus
  Marine data. The credit lines to keep, and what was changed in each file, are in
  [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md); the licence agreements themselves were not read in
  full, so check them at publication.
- OSTIA and OISST are different analyses: comparing them mixes product and resolution effects; the
  three-arm design in `ostia_peru.py` separates them.
- Run `python -m pytest` (offline; real-data tests need `data/`) and `python -m ruff check .`.

## Licence

The code, notebooks and documentation of this repository are under the [Apache License 2.0](LICENSE),
like xHW. The datasets and products distributed as release assets are modified copies of third-party
data and keep their providers' terms and credit lines: see [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).

## Scientific references and example configurations

See the [methodological bibliography](docs/REFERENCES.md) for complete references, DOI,
Bonino STL attribution and the verified Schlegel cold-category source. Plotting helpers
are example modules, separate from the installed API; the notebooks retain static figures.
