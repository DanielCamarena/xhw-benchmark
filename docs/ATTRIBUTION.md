# Attribution of the datasets

[README](../README.md) · [Data](DATA.md)

The files in `data/` and the products in `products/` are **modified copies** of data produced by
other institutions: subset to a box, decimated, averaged to a coarser grid, aggregated to local days,
or only recompressed. Every use or redistribution should keep the credit lines below. They were
assembled from each provider's published guidance on 11 October 2026; where a provider asks users to
copy a line from its current catalogue entry, the entry is named and its text should be checked at the
time of publication.

## What was changed

| File in `data/` | Provider's data | Modification |
|---|---|---|
| `oisst_peru_1991_2025.nc`, `oisst_equatorial_1991_2025.nc` | NOAA OISST v2.1 AVHRR-only, final | Spatial subset; stored values unchanged; lossless recompression |
| `ostia_peru_1991_2025.nc` | OSTIA reprocessed `analysed_sst` | Spatial subset; 35 yearly files joined in time; stored values unchanged; recompressed |
| `era5_peru_daily_1991_2025.nc` | ERA5 hourly single levels (`mx2t`, `mn2t`, `t2m`, `lsm`) | Spatial subset; aggregated by xHW to local-day maxima and minima (UTC−05:00); recompressed |
| `era5_peru_hourly_sample_2023_03.nc` | The same | Spatial subset; four days; values unchanged |
| `glorys12v1_peru_decimated_1993_2023.nc` | GLORYS12V1 `thetao` | Spatial subset; every 8th latitude and longitude; 12 retained levels; years joined; values unchanged |
| `products/**` | All of the above | Detection masks, event catalogs, objects and lineage graphs computed with xHW |

The tutorial `ostia_peru.py` additionally averages OSTIA over 5 × 5 sub-cells, in memory only.

## Credit lines

### NOAA OISST v2.1 (AVHRR-only)

- **Dataset.** NOAA National Centers for Environmental Information, *NOAA 0.25-degree Daily Optimum
  Interpolation Sea Surface Temperature (OISST), Version 2.1* (AVHRR-only, final data).
  DOI [10.25921/RE9P-PT57](https://doi.org/10.25921/RE9P-PT57), checked on 2026-10-12: it resolves to
  NCEI's Version 2.1 entry, released in March 2020. The other OISST DOI, 10.7289/V5SQ8XB5, is the
  earlier Version 2 and is not used here.
- **Method.** Huang, B., Liu, C., Banzon, V., Freeman, E., Graham, G., Hankins, B., Smith, T. and
  Zhang, H.-M. (2021). Improvements of the Daily Optimum Interpolation Sea Surface Temperature
  (DOISST) Version 2.1. *Journal of Climate*.
  [doi:10.1175/JCLI-D-20-0166.1](https://doi.org/10.1175/JCLI-D-20-0166.1).
- **Retrieved** through the public Source Cooperative Icechunk mirror, snapshot
  `1H5D09GQTBHSNHGT67TG`, on 2026-09-23 (Peru) and 2026-10-12 (equatorial Pacific); final data only.
- **Suggested line.** "Contains NOAA OISST v2.1 AVHRR-only data (Huang et al., 2021; NOAA/NCEI,
  doi:10.25921/RE9P-PT57), subset and recompressed."

### ERA5

- **Dataset.** Hersbach, H., Bell, B., Berrisford, P., et al. (2023). *ERA5 hourly data on single
  levels from 1940 to present*. Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
  DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47). Retrieved 2026-10-05.
- **Licence.** CC BY 4.0 for this dataset, as listed in its CDS catalogue entry (check the entry):
  link to the licence, credit the source and **indicate that the data were modified**.
- **Credit.** Copy the line under *Citation and attribution* of the CDS entry, which reads
  "Generated using Copernicus Climate Change Service information" or "Contains modified Copernicus
  Climate Change Service information" followed by a year (sources differ on the wording and year).
  Neither the European Commission nor ECMWF is responsible for any use made of the information.
- **Suggested line.** "Contains modified Copernicus Climate Change Service information (ERA5,
  Hersbach et al., 2023, doi:10.24381/cds.adbb2d47): subset, aggregated to local days and recompressed."

### GLORYS12V1 (Copernicus Marine)

- **Product.** `GLOBAL_MULTIYEAR_PHY_001_030`, dataset `cmems_mod_glo_phy_my_0.083deg_P1D-m`,
  DOI [10.48670/moi-00021](https://doi.org/10.48670/moi-00021) (Mercator Ocean International).
  Retrieved 2026-10-06 with the `copernicusmarine` client.
- **Credit for redistribution** (Copernicus Marine's template): "E.U. Copernicus Marine Service
  Information; https://doi.org/10.48670/moi-00021". Its source statement is "Source: European Union,
  Copernicus Marine Service Information or Data (year), ©Mercator Ocean."
- **For derived work:** "Generated using E.U. Copernicus Marine Service Information;
  https://doi.org/10.48670/moi-00021". **For a publication:** "This study has been conducted using
  E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00021".

### OSTIA reprocessed (Copernicus Marine)

- **Product.** `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`, dataset `METOFFICE-GLO-SST-L4-REP-OBS-SST`
  version 202003, produced by the UK Met Office with the OSTIA system. DOI
  [10.48670/moi-00168](https://doi.org/10.48670/moi-00168). Retrieved 2026-10-11 with the
  `copernicusmarine` client 2.5.0.
- **Credit lines.** The same three templates as above with this DOI. The ESA SST CCI and C3S
  analyses, a different product, carry DOI 10.48670/moi-00169: do not mix them.

### ESA SST CCI (downloader only)

`download_sst_cci.py` reads the ESA SST CCI Level-4 analysis from the CEDA archive. No file from it is
distributed here. Credit it as the
[CEDA catalogue entry](https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/) asks
when you use it.

## Products computed with xHW

The detection masks, catalogs, objects and lineage graphs in `products/` are derived from the data
above and carry the credit of each source: for GLORYS and OSTIA, "Generated using E.U. Copernicus
Marine Service Information" with the DOIs; for ERA5, "Contains modified Copernicus Climate Change
Service information"; for OISST, the NOAA line. xHW itself is Apache-2.0
([LICENSE](https://github.com/DanielCamarena/xhw/blob/main/LICENSE)).

## Text for the release page

> The datasets are modified subsets of NOAA OISST v2.1 (Huang et al., 2021; doi:10.25921/RE9P-PT57),
> ERA5 (Hersbach et al., 2023; doi:10.24381/cds.adbb2d47; contains modified Copernicus Climate Change
> Service information), GLORYS12V1 (doi:10.48670/moi-00021) and OSTIA (doi:10.48670/moi-00168), the
> latter two E.U. Copernicus Marine Service Information. Changes: spatial subsets, decimation,
> temporal aggregation or lossless recompression, as listed in `docs/ATTRIBUTION.md`. Neither the
> providers nor the European Commission or ECMWF are responsible for the use made of these data.

## What was and was not checked

The credit wording for Copernicus Marine and the CDS comes from their published guidance as read on
11 October 2026; the OISST DOI was checked against NCEI's entry; the ERA5 licence was read in secondary descriptions of the CDS entry. None of the licence
agreements was read in full, and the Copernicus Marine Licence Agreement is the authoritative text.
