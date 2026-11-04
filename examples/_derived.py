# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""ERA5 derived variables for the acquisition scripts (revision 2, 7.5; revision 1, L10).

xHW detects events on temperatures only and converts no units, so these quantities are
computed here and delivered as variables of their own, with their formula, constants,
source variables and order of computation in the attributes. Two uses are kept apart:

- detection inputs (temperatures, in the units the literature defines them): the daily
  mean temperature for the EHF, kept in three versions that are never presented as
  equivalent (decision 7), the apparent temperature and the NWS heat index;
- diagnostic fields (wind, radiation, heat flux, humidity, humidex, geopotential): their
  association with detected events does not imply causality.

Order of computation, declared per product: non-linear formulas are computed hour by hour
and then reduced to days (wind speed, humidity, apparent temperature, heat indices); the
mean of the extremes ``(TX + TN) / 2`` and the daily wind direction (from the mean
components) reduce first and combine after. Masks (incomplete days; the terrain mask of
pressure levels) are applied before.

UTCI and WBGT (decision 8) use ``thermofeel`` (ECMWF), an optional acquisition dependency
imported only when they are asked for. Both need the mean radiant temperature (Di Napoli
et al. 2020), from the hourly radiation fluxes, the direct solar radiation ``fdir`` and the
cosine of the solar zenith angle, which is computed here and averaged over the hour of each
radiation accumulation. The instantaneous temperature, dew point and wind at stamp ``t``
are paired with the radiation accumulated over the hour ending at ``t``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

G0 = 9.80665  # m s-2, standard gravity (geopotential height)
T0 = 273.16  # K, triple point of water (Tetens reference)
ES_A1, ES_A3, ES_A4 = 611.21, 17.502, 32.19  # Pa, -, K: Tetens over water, IFS CY47R3
EPSILON = 0.621981  # R_d / R_v
RD_CP = 0.285691  # R_d / c_p, IFS CY47R3
CALM = 0.1  # m s-1: below this speed the wind direction is undefined
KELVIN = 273.15  # K at 0 degC (unit conversion, not the Tetens reference)
CATALOGUE_VERSION = "era5-derived-2"
UTCI_DOMAIN = (
    "air temperature -50..50 degC, 10 m wind 0.5..17 m s-1, MRT - Ta -30..70 K, "
    "water vapour pressure <= 50 hPa"
)
TETENS = f"e_s(T) = {ES_A1} exp({ES_A3} (T - {T0}) / (T - {ES_A4})) Pa, T in K"
ROTHFUSZ = (
    "simple = 0.5 (T + 61 + 1.2 (T - 68) + 0.094 RH); if simple >= 80 degF: HI = -42.379 "
    "+ 2.04901523 T + 10.14333127 RH - 0.22475541 T RH - 0.00683783 T^2 - 0.05481717 RH^2 "
    "+ 0.00122874 T^2 RH + 0.00085282 T RH^2 - 0.00000199 T^2 RH^2, minus "
    "(13 - RH)/4 sqrt((17 - |T - 95|)/17) if RH < 13 and 80 <= T <= 112, plus "
    "(RH - 85)/10 (87 - T)/5 if RH > 85 and 80 <= T <= 87; T in degF, RH in %"
)
#: Constants and auxiliary formulas of each product, written into its attributes.
CONSTANTS = {
    "wdir10": f"calm threshold {CALM} m s-1",
    "rad_down": "3600 s per hourly accumulation",
    "qnet": "3600 s per hourly accumulation",
    "rh2m": TETENS,
    "q2m": f"{TETENS}; epsilon = {EPSILON}",
    "at2m": f"{TETENS}; e in hPa from e_s(d2m); 273.15 K = 0 degC",
    "hi2m": f"{ROTHFUSZ}; RH = 100 e_s(d2m)/e_s(t2m), {TETENS}",
    "humidex": "6.11 hPa, 5417.7530 K, 273.16 K; T in degC, Td in K",
    "mrt": "thermofeel.calculate_mean_radiant_temperature (Di Napoli et al. 2020); dsrp = "
    "thermofeel.approximate_dsrp(fdir, cossza, threshold 0.1); fluxes = accumulations / 3600 s; "
    "cossza = mean cos zenith over the sunlit part of the hour ending at the stamp (60 "
    "sub-steps; 0 if the sun stays down), NOAA solar position (Spencer 1971; 365/366 days)",
    "utci2m": "thermofeel.calculate_utci (Broede et al. 2012 polynomial, not clamped); "
    f"validity domain {UTCI_DOMAIN}; hours outside it are kept and counted in "
    "utci2m_outside_domain_fraction; wind = sqrt(u10**2 + v10**2) at 10 m; MRT as in mrt",
    "wbgt2m": "thermofeel.calculate_wbgt: Stull (2011) wet bulb from t2m and relative "
    "humidity (fitted at sea-level pressure, about -20..50 degC and 5..99 %), black globe "
    "from MRT and the 10 m wind; an estimate, not the instrumental natural wet bulb; "
    "provisional diagnostic, not validated for detection, especially at altitude or in sun",
}
REFERENCES = {
    "tetens": "Tetens over water with IFS CY47R3 constants (via earthkit-meteo 1.3.0)",
    "steadman": "Steadman 1994; Bureau of Meteorology version without radiation",
    "rothfusz": "NWS heat index: Rothfusz regression with the NWS adjustments and simple formula",
    "humidex": "Masterton and Richardson 1979 (Environment Canada)",
    "ehf": "Nairn and Fawcett 2015",
}


# --- Formulas (plain numpy or xarray arithmetic) -----------------------------------------


def saturation_vapour_pressure(t):
    """Pa, over water, for ``t`` in K: ``611.21 exp(17.502 (T - 273.16) / (T - 32.19))``."""
    return ES_A1 * np.exp(ES_A3 * (t - T0) / (t - ES_A4))


def relative_humidity(t, td):
    """Percent, from temperature and dew point in K (water phase, WMO convention)."""
    return 100.0 * saturation_vapour_pressure(td) / saturation_vapour_pressure(t)


def specific_humidity(td, sp):
    """kg kg-1 from the dew point (K) and surface pressure (Pa): ``eps e / (p - (1 - eps) e)``."""
    e = saturation_vapour_pressure(td)
    return EPSILON * e / (sp - (1.0 - EPSILON) * e)


def wind_speed(u, v):
    return np.sqrt(u * u + v * v)


def wind_direction(u, v, calm: float = CALM):
    """Degrees the wind blows from (0/360 north, 90 east); NaN where the speed <= ``calm``."""
    direction = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
    return np.where(wind_speed(u, v) > calm, direction, np.nan)


def apparent_temperature(t, td, u, v):
    """degC: ``Ta + 0.33 e - 0.70 ws - 4.00`` with Ta in degC, e in hPa, ws at 10 m in m s-1."""
    e_hpa = saturation_vapour_pressure(td) / 100.0
    return (t - KELVIN) + 0.33 * e_hpa - 0.70 * wind_speed(u, v) - 4.00


def heat_index(t, td):
    """degC, NWS: simple formula, or the Rothfusz regression with adjustments from 80 degF.

    The simple formula ``0.5 (T + 61 + 1.2 (T - 68) + 0.094 RH)`` is already the average of
    Steadman's approximation and the temperature; when it reaches 80 degF the regression is
    used instead.
    """
    tf = np.asarray((t - KELVIN) * 9.0 / 5.0 + 32.0, dtype=float)
    rh = np.asarray(relative_humidity(t, td), dtype=float)
    simple = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
    full = (
        -42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh
        - 0.00683783 * tf**2 - 0.05481717 * rh**2 + 0.00122874 * tf**2 * rh
        + 0.00085282 * tf * rh**2 - 0.00000199 * tf**2 * rh**2
    )  # fmt: skip
    dry = (rh < 13) & (tf >= 80) & (tf <= 112)
    with np.errstate(invalid="ignore"):
        full = np.where(dry, full - (13 - rh) / 4 * np.sqrt((17 - np.abs(tf - 95)) / 17), full)
    humid = (rh > 85) & (tf >= 80) & (tf <= 87)
    full = np.where(humid, full + (rh - 85) / 10 * (87 - tf) / 5, full)
    hi = np.where(simple >= 80, full, simple)
    return (hi - 32.0) * 5.0 / 9.0


def humidex(t, td):
    """Dimensionless: ``T + 0.5555 (6.11 exp(5417.7530 (1/273.16 - 1/Td)) - 10)``, T in degC."""
    return (t - KELVIN) + 0.5555 * (6.11 * np.exp(5417.7530 * (1 / T0 - 1 / td)) - 10.0)


def potential_temperature(t, p_hpa):
    """K: ``T (1000 / p) ** (R_d / c_p)`` with ``p`` in hPa."""
    return t * (1000.0 / p_hpa) ** RD_CP


def above_ground(p_hpa, sp):
    """True where the level is above the terrain: ``sp`` finite and ``100 p <= sp`` (Pa).

    A missing or non-finite ``sp`` is never above ground (revision 1, L6): the naive
    ``~(100 p > sp)`` would keep the sample, because comparisons with NaN are false.
    """
    sp = np.asarray(sp, dtype=float)
    return np.isfinite(sp) & (100.0 * np.asarray(p_hpa, dtype=float) <= sp)


def cos_solar_zenith_hour(times, lat, lon, substeps: int = 60) -> np.ndarray:
    """Mean cosine of the solar zenith angle over the sunlit part of the hour ending at each
    UTC stamp; 0 when the sun stays below the horizon for the whole hour.

    Di Napoli et al. (2020, eqs. 12-13) average over the illuminated part of the
    accumulation interval: averaging ``max(cos, 0)`` over the whole hour would bias the
    sunrise and sunset hours low. ``(time, lat, lon)`` array for 1-D ``lat``/``lon`` in
    degrees. Solar position from the NOAA general solar position equations (Spencer 1971
    declination and equation of time; 365 or 366 days in the year), sampled at ``substeps``
    mid-points of the hour.
    """
    stamps = np.asarray(times, dtype="datetime64[s]")
    phi = np.radians(np.asarray(lat, dtype=float))[None, :, None]
    lam = np.asarray(lon, dtype=float)[None, None, :]
    total = np.zeros((stamps.size, phi.shape[1], lam.shape[2]))
    lit = np.zeros_like(total)
    for k in range(substeps):
        offset = np.timedelta64(int(3600 * (k + 0.5) / substeps) - 3600, "s")
        t = stamps + offset
        day = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(float)  # 0-based
        hours = (t - t.astype("datetime64[D]")).astype(float) / 3600.0
        year = t.astype("datetime64[Y]")
        next_year = (year + np.timedelta64(1, "Y")).astype("datetime64[D]")
        length = (next_year - year.astype("datetime64[D]")).astype(float)
        g = (2 * np.pi / length * (day + (hours - 12) / 24))[:, None, None]
        eqtime = 229.18 * (
            0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
            - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
        )  # fmt: skip
        decl = (
            0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
        )  # fmt: skip
        solar_minutes = hours[:, None, None] * 60 + eqtime + 4 * lam
        hour_angle = np.radians(solar_minutes / 4 - 180)
        cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
        up = cosz > 0
        total += np.where(up, cosz, 0.0)
        lit += up
    return np.divide(total, lit, out=np.zeros_like(total), where=lit > 0)


def _thermofeel():
    try:
        import thermofeel
    except ImportError as exc:
        raise ImportError(
            "UTCI, WBGT and the mean radiant temperature need thermofeel (ECMWF); install it "
            "from examples/requirements-download.txt. It is not an xHW dependency."
        ) from exc
    return thermofeel


def mean_radiant_temperature(h):
    """K, from hourly ERA5 accumulations (J m-2) of ssrd, ssr, strd, str and fdir."""
    tf = _thermofeel()
    ref = h["ssrd"]
    flux = {
        v: np.asarray(h[v].values, dtype=float) / 3600.0
        for v in ("ssrd", "ssr", "strd", "str", "fdir")
    }
    cossza = cos_solar_zenith_hour(
        ref["time"].values, ref["latitude"].values, ref["longitude"].values
    )
    cossza = np.broadcast_to(cossza, flux["fdir"].shape).copy()
    dsrp = tf.approximate_dsrp(flux["fdir"], cossza)
    mrt = tf.calculate_mean_radiant_temperature(
        flux["ssrd"], flux["ssr"], dsrp, flux["strd"], flux["fdir"], flux["str"], cossza
    )
    return ref.copy(data=mrt)


def utci(h):
    """K, UTCI from t2m, d2m, the 10 m wind and the mean radiant temperature."""
    tf = _thermofeel()
    mrt = mean_radiant_temperature(h)
    va = np.asarray(wind_speed(h["u10"], h["v10"]).values, dtype=float)
    out = tf.calculate_utci(
        np.asarray(h["t2m"].values, dtype=float), va, np.asarray(mrt.values, dtype=float),
        td_k=np.asarray(h["d2m"].values, dtype=float),
    )  # fmt: skip
    return h["t2m"].copy(data=out)


def utci_outside_domain(h):
    """1.0 where an hourly UTCI input is outside the domain of the polynomial, 0.0 inside,
    NaN where an input is missing: an unknown hour is never counted as inside (C4)."""
    ta = h["t2m"] - KELVIN
    va = wind_speed(h["u10"], h["v10"])
    delta = mean_radiant_temperature(h) - h["t2m"]
    e_hpa = saturation_vapour_pressure(h["d2m"]) / 100.0
    outside = (
        (ta < -50) | (ta > 50) | (va < 0.5) | (va > 17) | (delta < -30) | (delta > 70)
        | (e_hpa > 50)
    ).astype("float64")  # fmt: skip
    known = ta.notnull() & va.notnull() & delta.notnull() & e_hpa.notnull()
    return outside.where(known)


def wbgt(h):
    """K, wet bulb globe temperature from t2m, d2m, the 10 m wind and the MRT."""
    tf = _thermofeel()
    mrt = mean_radiant_temperature(h)
    va = np.asarray(wind_speed(h["u10"], h["v10"]).values, dtype=float)
    out = tf.calculate_wbgt(
        np.asarray(h["t2m"].values, dtype=float), np.asarray(mrt.values, dtype=float), va,
        np.asarray(h["d2m"].values, dtype=float),
    )  # fmt: skip
    return h["t2m"].copy(data=out)


# --- Catalogue ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Derived:
    """A derived product: inputs, how it is computed and which daily values it allows."""

    inputs: tuple[str, ...]
    units: str
    daily: tuple[str, ...]
    sampling: str  # of the hourly series it is reduced from
    order: str  # "hourly then daily" or "daily then combined"
    formula: str
    use: str  # "detection" or "diagnostic"
    reference: str = ""


#: Surface products of revision 2, 7.5 (priorities A and B accepted in decision 8).
SURFACE = {
    "ws10": Derived(
        ("u10", "v10"), "m s-1", ("mean", "max"), "instantaneous", "hourly then daily",
        "sqrt(u10**2 + v10**2)", "diagnostic",
    ),
    "wdir10": Derived(
        ("u10", "v10"), "degree", ("mean",), "instantaneous", "daily then combined",
        f"(270 - atan2(mean v10, mean u10) in degrees) mod 360; undefined if speed <= {CALM}",
        "diagnostic",
    ),
    "rad_down": Derived(
        ("ssrd", "strd"), "W m-2", ("mean",), "period_ending", "hourly then daily",
        "(ssrd + strd) / 3600 s, positive downwards", "diagnostic",
    ),
    "qnet": Derived(
        ("ssr", "str", "sshf", "slhf"), "W m-2", ("mean",), "period_ending",
        "hourly then daily", "(ssr + str + sshf + slhf) / 3600 s, positive downwards (ECMWF)",
        "diagnostic",
    ),
    "tmean_t2m": Derived(
        ("t2m",), "K", ("mean",), "instantaneous", "hourly then daily",
        "mean of the hourly t2m of the local day", "detection", REFERENCES["ehf"],
    ),
    "tmean_txtn": Derived(
        ("mx2t", "mn2t"), "K", ("mean",), "period_ending", "daily then combined",
        "(TX + TN) / 2 on the same local day; an adaptation of the EHF daily mean",
        "detection", REFERENCES["ehf"],
    ),
    "tmean_9to9": Derived(
        ("mx2t", "mn2t"), "K", ("mean",), "period_ending", "daily then combined",
        "(TX + TN) / 2 over 09:00-09:00 local time, labelled with the day it starts",
        "detection", REFERENCES["ehf"],
    ),
    "rh2m": Derived(
        ("t2m", "d2m"), "%", ("mean", "min", "max"), "instantaneous", "hourly then daily",
        "100 e_s(d2m) / e_s(t2m)", "diagnostic", REFERENCES["tetens"],
    ),
    "q2m": Derived(
        ("d2m", "sp"), "kg kg-1", ("mean",), "instantaneous", "hourly then daily",
        f"{EPSILON} e / (sp - (1 - {EPSILON}) e), e = e_s(d2m)", "diagnostic",
        REFERENCES["tetens"],
    ),
    "at2m": Derived(
        ("t2m", "d2m", "u10", "v10"), "degC", ("max", "mean"), "instantaneous",
        "hourly then daily", "Ta + 0.33 e - 0.70 ws - 4.00 (degC, hPa, m s-1)", "detection",
        REFERENCES["steadman"],
    ),
    "hi2m": Derived(
        ("t2m", "d2m"), "degC", ("max", "mean"), "instantaneous", "hourly then daily",
        "NWS heat index in degF, converted to degC", "detection", REFERENCES["rothfusz"],
    ),
    "humidex": Derived(
        ("t2m", "d2m"), "1", ("max", "mean"), "instantaneous", "hourly then daily",
        "T + 0.5555 (6.11 exp(5417.7530 (1/273.16 - 1/Td)) - 10); a comfort scale, no units",
        "diagnostic", REFERENCES["humidex"],
    ),
    "mrt": Derived(
        ("ssrd", "ssr", "strd", "str", "fdir"), "K", ("max", "mean"), "period_ending",
        "hourly then daily", "mean radiant temperature from the radiation fluxes of the hour "
        "ending at each stamp", "diagnostic", "Di Napoli et al. 2020 (thermofeel)",
    ),
    "utci2m": Derived(
        ("t2m", "d2m", "u10", "v10", "ssrd", "ssr", "strd", "str", "fdir"), "K",
        ("max", "mean"), "instantaneous", "hourly then daily",
        "UTCI(t2m, wind at 10 m, MRT, d2m)", "detection", "Broede et al. 2012 (thermofeel)",
    ),
    "wbgt2m": Derived(
        ("t2m", "d2m", "u10", "v10", "ssrd", "ssr", "strd", "str", "fdir"), "K",
        ("max", "mean"), "instantaneous", "hourly then daily",
        "estimated WBGT = 0.7 Tw + 0.2 Tg + 0.1 Ta, with Stull's psychrometric wet bulb (not "
        "the natural wet bulb of the instrument) and the black globe from MRT", "diagnostic",
        "Stull 2011; ECMWF thermofeel",
    ),
}  # fmt: skip

HOURLY = {
    "ws10": lambda h: wind_speed(h["u10"], h["v10"]),
    "rad_down": lambda h: (h["ssrd"] + h["strd"]) / 3600.0,
    "qnet": lambda h: (h["ssr"] + h["str"] + h["sshf"] + h["slhf"]) / 3600.0,
    "tmean_t2m": lambda h: h["t2m"],
    "rh2m": lambda h: relative_humidity(h["t2m"], h["d2m"]),
    "q2m": lambda h: specific_humidity(h["d2m"], h["sp"]),
    "at2m": lambda h: apparent_temperature(h["t2m"], h["d2m"], h["u10"], h["v10"]),
    "hi2m": lambda h: heat_index(h["t2m"], h["d2m"]),
    "humidex": lambda h: humidex(h["t2m"], h["d2m"]),
    "mrt": lambda h: mean_radiant_temperature(h),
    "utci2m": lambda h: utci(h),
    "wbgt2m": lambda h: wbgt(h),
}


MAX_OFFSET = 14 * 60 + 59  # minutes: xHW accepts offset:+-HH:MM up to 14:59 (ADR-029)


def shifted_day(day_definition: str, hours: int) -> tuple[str, int]:
    """A day definition whose day starts ``hours`` later, and the label shift in days.

    09:00-09:00 local days use ``hours=9``. When the shifted clock passes the +-14:59 that
    xHW accepts, the equivalent clock 24 h away is used: its day D is the wanted day
    D - shift, so labels must be moved back by ``shift`` days.
    """
    if day_definition == "UTC":
        minutes = 0
    else:
        match = re.fullmatch(r"offset:([+-])(\d{2}):(\d{2})", day_definition)
        if not match:
            raise ValueError(f"unsupported day definition {day_definition!r}")
        sign = 1 if match.group(1) == "+" else -1
        minutes = sign * (int(match.group(2)) * 60 + int(match.group(3)))
    minutes -= hours * 60
    shift = 0
    if minutes < -MAX_OFFSET:
        minutes, shift = minutes + 1440, 1
    elif minutes > MAX_OFFSET:
        minutes, shift = minutes - 1440, -1
    sign = "+" if minutes >= 0 else "-"
    return f"offset:{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}", shift


def required(names) -> set[str]:
    """Hourly variables the derived products ``names`` need."""
    unknown = [n for n in names if n not in SURFACE]
    if unknown:
        raise ValueError(f"unknown derived products {unknown}; known: {', '.join(SURFACE)}")
    return {v for n in names for v in SURFACE[n].inputs}


def daily(
    name: str,
    get_hourly: Callable[[str, str, date, date, str], object],
    start: date,
    end: date,
    day_definition: str,
):
    """Daily values of a derived product over ``start..end`` (one dataset, one variable per
    statistic: ``<name>`` for the single statistic, ``<name>_<stat>`` otherwise).

    ``get_hourly(var, sampling, start, end, day_definition)`` returns the hourly dataset
    of ``var`` over the UTC hours those local days need for that sampling.
    """
    import xarray as xr

    import xhw

    spec = SURFACE[name]

    counts = {}

    def reduce(da, stat, sampling, definition, shift=0, count=None):
        """Daily values; the valid samples per day are kept as ``<count>_n_samples``."""
        ds = da.to_dataset(name="value")
        out = xhw.aggregate_daily(
            ds, variable="value", statistic=stat, sampling=sampling, input_clock="UTC",
            day_definition=definition,
        )  # fmt: skip
        if shift:
            out = out.assign_coords(time=out["time"] - np.timedelta64(shift, "D"))
        out = out.sel(time=slice(f"{start}", f"{end}"))
        if count:
            counts[f"{count}_n_samples"] = out["value_n_samples"].assign_attrs(
                {**out["value_n_samples"].attrs, "ancillary_of": count}
            )
        return out["value"]

    def series(var, sampling=spec.sampling, definition=day_definition, shift=0):
        days = timedelta(shift)
        return get_hourly(var, sampling, start + days, end + days, definition)[var]

    out, indicators = {}, {}
    if name in HOURLY:
        hourly = {v: series(v) for v in spec.inputs}
        aligned = xr.align(*hourly.values(), join="exact")
        values = HOURLY[name](dict(zip(hourly, aligned)))
        if not isinstance(values, xr.DataArray):  # numpy formulas: back on the input grid
            values = aligned[0].copy(data=np.asarray(values))
        for stat in spec.daily:
            key = name if len(spec.daily) == 1 else f"{name}_{stat}"
            out[key] = reduce(values, stat, spec.sampling, day_definition, count=name)
        if name == "utci2m":
            outside = utci_outside_domain(dict(zip(hourly, aligned)))
            indicators[f"{name}_outside_domain_fraction"] = reduce(
                outside, "mean", spec.sampling, day_definition
            ).assign_attrs(
                units="1", long_name="share of the hours outside the UTCI validity domain",
                domain=UTCI_DOMAIN,
            )  # fmt: skip
    elif name == "wdir10":
        u = reduce(series("u10"), "mean", "instantaneous", day_definition, count=f"{name}_u10")
        v = reduce(series("v10"), "mean", "instantaneous", day_definition, count=f"{name}_v10")
        out[name] = xr.apply_ufunc(wind_direction, u, v)
    else:  # tmean_txtn, tmean_9to9: reduce, then combine
        definition, shift = (
            (day_definition, 0) if name == "tmean_txtn" else shifted_day(day_definition, 9)
        )
        tx = series("mx2t", definition=definition, shift=shift)
        tn = series("mn2t", definition=definition, shift=shift)
        tx = reduce(tx, "max", "period_ending", definition, shift, count=f"{name}_mx2t")
        tn = reduce(tn, "min", "period_ending", definition, shift, count=f"{name}_mn2t")
        out[name] = (tx + tn) / 2
    ds = xr.Dataset(out)
    for key in list(ds.data_vars):
        ds[key].attrs = {
            "units": spec.units,
            "formula": spec.formula,
            "constants": CONSTANTS.get(name, "none"),
            "source_variables": " ".join(spec.inputs),
            "order": spec.order,
            "sampling": spec.sampling,
            "completeness": "a day needs every hourly sample (aggregate_daily default)",
            "use": spec.use,
            "day_definition": day_definition,
            "catalogue": CATALOGUE_VERSION,
            **({"reference": spec.reference} if spec.reference else {}),
        }
    for key, da in {**counts, **indicators}.items():
        ds[key] = da
    return ds
