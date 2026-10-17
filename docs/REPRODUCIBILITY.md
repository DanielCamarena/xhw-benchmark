# Reproducing the cases

[README](../README.md) · [Tutorials](TUTORIALS.md) · [Data](DATA.md) · [Plotting](PLOTTING.md)

The examples use xHW `0.0.1` and registered regional datasets. They never download
implicitly. Saved notebooks contain static results for reading without the data;
interactive controls need a running Jupyter kernel.

## Installation and analysis

Install the public xHW package from its local checkout or built wheel:

```bash
python -m pip install ../xhw
python -m pip install -r examples/requirements-interactive.txt
python -m pip check
python examples/verify_data.py --data-dir data
python examples/oisst_peru.py --data-dir data --output-dir outputs/oisst --products-dir products/oisst_peru
python examples/ostia_peru.py --data-dir data --output-dir outputs/ostia --products-dir products/ostia_resolution
python examples/oisst_equatorial.py --data-dir data --output-dir outputs/equatorial --products-dir products/oisst_equatorial
python examples/era5_peru.py --data-dir data --output-dir outputs/era5 --products-dir products/era5_peru
python examples/glorys_volume.py --data-dir data --output-dir outputs/glorys --products-dir products/glorys_volume
```

If inputs are absent, obtain them separately with `examples/fetch_data.py`; see
[Data](DATA.md). Commands work in PowerShell and POSIX shells. Each script writes
figures, a `report.json` and, when requested, compressed NetCDF products with a
`products_manifest.json`. All xHW entity schemas use version `0.1`.

## Notebooks and reports

```bash
python examples/execute_notebooks.py --case oisst --data-dir data --output-dir outputs/notebooks
python examples/execute_notebooks.py --data-dir data --output-dir outputs/notebooks
```

The second command runs all six notebooks and additionally requires the separately
registered North Pacific input. The runner uses fresh kernels, updates notebooks in
place and checks format and error outputs. Script reports and saved notebook outputs
retain the actual software versions and parameters of their producing execution.
Do not infer execution metadata from the current installation instructions.

Reports record source/checksum, selection, full Specs, date, Python, software versions,
code fingerprints and scientific-review state. Dataset manifests and file checksums
identify the input. Product manifests identify generated files. Recomputing a result
does not grant scientific approval.

## Checks

```bash
python -m pytest
python -m ruff check examples tests tools
python tools/validate_docs.py
```

The small offline tests cover dataset verification, product export, plotting and
example helpers; they do not download or execute all real-data cases. Analysis scripts
also invoke public baseline, mask, catalog, object and graph validation methods.
A failed check is an error, except the explicitly reported catalog qualification
constraint described below.

## Prepare the standard data release

The standard data release contains six registered input datasets and five archives
of derived products. North Pacific is outside this bundle. With those six inputs
already in `outputs/release-data-v1`, run:

```bash
python tools/regenerate_release_data.py --release-dir outputs/release-data-v1 --work-dir outputs/release-data-run
```

Choose a new work directory for every run. The runner uses the current Python and
requires xHW `0.0.1` with entity schemas `0.1`. It verifies inputs and executes OISST
Peru, OSTIA, equatorial OISST, ERA5 and GLORYS sequentially. Working arrays are removed
after each successful case. Reserve at least 25 GiB free disk; optional disk backing
reduces memory pressure without changing the scientific selection.

Inspect `<work-dir>/status.json` and each case's `.log`. Creating an empty `STOP`
file in the work directory stops the active analysis. A failure retains its log
and working files and does not replace the release bundle. After all five cases
succeed, the runner installs the new ZIPs, notes and checksums; it also updates local
case products and reports. Completion means the files were generated, not that their
scientific interpretation was approved. Review them before publishing.

## Memory on large examples

OSTIA and the equatorial Pacific retain several full-period temperature, mask and
baseline arrays. On a memory-limited machine, opt into disk backing before starting
the script or notebook kernel:

```bash
export XHW_MAPPED_ARRAYS=1                 # Linux/macOS shell
# PowerShell: $env:XHW_MAPPED_ARRAYS = "1"
python examples/ostia_peru.py --data-dir data --output-dir results/ostia
```

The helper copies labelled arrays to read-only NPY mappings under
`<output-dir>/_arrays`. It preserves values, dtypes, resolution, dates, missing
flags and scientific Specs; it does not change the installed core. Temporary
copies during fitting, detection and spatial operations still require RAM.
Reserve at least 10 GiB of free disk for OSTIA and 20 GiB for the equatorial case
per execution. New executions use unique filenames, so space accumulates. Keep
these files while a live notebook uses them; they are local working files, not
publication assets. The helpers never remove them automatically. Run reports
record the requested memory option separately from scientific parameters.

The North Pacific example instead partitions the temporal work by latitude while
retaining each cell's complete time series. It assembles the full 2012–2018 window
before segmentation, tracking and genealogy; spatial connectivity crosses temporal
processing block boundaries normally. This reduces working memory without changing
the data selection or detector. The full-period mask exports are divided by latitude
block; their manifest records the resulting files.

## Scientific review and interpretation limits

- The fixed-baseline OISST temporal regression has scientific approval.
  The tutorial reproduces its registered 16,702 MHW events and 422,215 event
  cell-days. Its dynamic, object and lineage extensions, and the ERA5/GLORYS real
  summaries, remain **executed and pending scientific review**. Numerical
  reproducibility does not turn them into approved scientific conclusions.
- OISST tracking retains 118 identities, while genealogy contains 136 components
  of the daily patches, including 18 shorter than five days. The tracker filters
  identities without deleting those patches.
  `graph.to_event_catalog(mask)` marks all components qualified and raises
  `InvariantError: qualified events shorter than min_duration_days`. The tutorial
  reports this constraint and keeps the graph available; it does not silently
  discard or reclassify components. See [Objects](https://github.com/DanielCamarena/xhw/blob/main/docs/OBJECTS.md#catalog-export-qualification). Synthetic and
  GLORYS lineage-catalog exports pass their checks.
- GLORYS has approximate retained-level midpoint thicknesses, not exact native
  model interfaces. Decimation changes support, adjacency and effective
  resolution. Its volume estimates describe that supplied support. ERA5 over
  ocean describes air, not ocean water.
- Missing observations, joined temporal gaps, filled values and spatially added
  cells remain separate flags. A split/merge represents qualifying overlap;
  genealogy is not physical attribution. Graph panels limited to 18 branches
  are labelled as subsets.
- A fresh retrieval may differ from the registered datasets; source services and
  file metadata can change. Downloading is separate from analysis. See [Data](DATA.md).
- macOS and global-domain memory costs require additional evidence.

Next: [Scientific contract](https://github.com/DanielCamarena/xhw/blob/main/docs/SCIENTIFIC_CONTRACT.md)
