# Changelog

Code and data are versioned apart. A code tag (`v0.2.0`) marks scripts, notebooks, tests and
documentation; a data tag (`data-v1`) marks the release assets: the registered input datasets and
the processed products, with their checksums. A new code version does not change the data unless
its notes say so, and `examples/fetch_data.py` keeps reading the data tag named in the
[README](README.md#quick-start).

## v0.2.0 (2026-11-04): rebuilt downloaders

Code only. **The data stay `data-v1`**: the same six input datasets and the same processed
products, byte for byte. The tutorials, notebooks and products do not change. Tested with xHW
0.0.1 (commit `aa44a80`).

### Downloaders

All six `examples/download_*.py` now share `examples/_acquire.py`:

- **The data contain the requested box.** `--region` (`peru`, `equatorial`, `npacific`) or
  `--box`, with `--box-policy enclose` by default (`inside` reproduces the registered requests).
  The extent received is checked and written to the manifest. Boxes crossing the antimeridian are
  requested in two parts and joined west to east.
- **Interrupted or extended runs resume.** A ledger per request keeps the state of each piece, and
  a file is reused only if its SHA-256 matches. A damaged answer is recorded (size, SHA-256,
  error), deleted and retried once; `--keep-failed` keeps it in quarantine instead. Only transient
  errors are retried. Every failure is reported, and the exit code says whether anything is
  missing.
- **Manifests are named after the request.** They hold `sha256`, `content_sha256`, software
  versions, the retrieval window and the box check, and are compared with the registered input
  where one exists.

New:

- `download_era5_pressure.py`: ERA5 pressure levels, with terrain masked hour by hour by surface
  pressure before the daily mean, maximum and minimum.
- `examples/_derived.py`: daily ERA5 derived variables (wind, radiation, net heat flux, three
  daily means for EHF, humidity, apparent temperature, NWS heat index, humidex), and UTCI, WBGT and
  mean radiant temperature through the optional `thermofeel`.

Per source:

- **ERA5 single levels:** a variable catalogue; monthly requests grouped by GRIB step type under
  the CDS cost limit (the registered plan stays available); ERA5T rejected unless
  `--allow-era5t`; daily files per variable.
- **Memory, every downloader:** ERA5 daily products and consolidated years (single and pressure
  levels) are built and written in month blocks (`--daily-block`), about one month of one
  variable in memory instead of a year of hourly data plus the whole daily product. OISST joins
  a box one year at a time, OSTIA writes years month by month and ESA SST CCI combines 31 days at
  a time, instead of the whole series, year or period; GLORYS already joined by slabs. Stored
  values, packing and attributes are those of a single join.
- **Compression, every downloader:** the NetCDF files each script keeps (pieces and products)
  are written with zlib and byte shuffle, level 4 by default (`--compression-level`);
  `--no-compress` writes them uncompressed. Before, only ERA5 compressed, with `--compress`;
  OISST, OSTIA and GLORYS never did, and SST CCI always used level 4. The encoding is part of
  each product's identity: another setting rewrites pieces and products in place, block by
  block, checking `content_sha256`, without rebuilding or downloading again. Values never
  change.
- **OISST:** the store snapshot is part of each piece's identity, and preliminary days are checked
  month by month.
- **GLORYS:** pinned dataset version and `--max-depth`.
- **OSTIA:** one request split into years.
- **ESA SST CCI:** `--include-icdr` and an ETag check.
- **GLORYS and OSTIA:** partial years only with `--allow-partial`.

The details are in [docs/DATA.md](docs/DATA.md#optional-acquisition-scripts).

### Checks

- **Offline tests:** 377 pass without network or credentials. CI tests against xHW 0.0.1 and,
  apart, against xHW's default branch; that CI has not run remotely yet.
- **`data-v1` regression:** [tools/regress_data_v1.py](tools/regress_data_v1.py) rebuilds the
  registered inputs from the historical originals without network.
  - ERA5 is replayed from the registered hourly files through `download_era5.py`.
  - Result: 115 checks passed, none failed, and the same `content_sha256` for every entry.
  - The OISST, GLORYS, OSTIA and ESA SST CCI downloaders are checked only against their services.
- **Short real downloads on 2026-11-03/04:** a box not aligned with any grid, 3 days to 1 month
  per source, and one antimeridian box each at the CDS and at Copernicus Marine. Every manifest
  passed the box check.
  - Found: a one-hour ERA5 answer carries a scalar `expver`, which the validator could not
    read. Fixed before this version.
  - Long downloads were not repeated.

## Initial release (2026-10-17, commit `58d2def`, tag `data-v1`)

Real-data cases for xHW 0.0.1 (OISST Peru, OSTIA versus OISST, OISST equatorial Pacific, OISST
North Pacific, ERA5 Peru, GLORYS volume), with notebooks, processed products, compressed test
datasets and their downloaders. The data were published as the `data-v1` release. The code had no
version tag.
