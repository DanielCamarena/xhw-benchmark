# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Derived ERA5 variables: the acceptance cases of revision 2, 7.5 (and revision 1, L6/L10)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("xhw")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _derived as dv  # noqa: E402

K = 273.15


def dew_point(t, rh):
    """Invert the Tetens formula: the dew point (K) giving ``rh`` percent at ``t`` (K)."""
    log = np.log(rh / 100 * dv.saturation_vapour_pressure(t) / dv.ES_A1)
    return (dv.ES_A3 * dv.T0 - log * dv.ES_A4) / (dv.ES_A3 - log)


def fahrenheit_to_kelvin(f):
    return (f - 32) * 5 / 9 + K


# --- formulas ----------------------------------------------------------------------------


def test_wind_direction_cardinal_points_and_calm():
    u = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.05])
    v = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    got = dv.wind_direction(u, v)
    # from the west, south, east, north; calm and below the threshold are undefined
    assert got[:3].tolist() == [270.0, 180.0, 90.0]
    assert got[3] % 360 == 0.0
    assert np.isnan(got[4]) and np.isnan(got[5])  # atan2(0, 0) would give 270


def test_humidity():
    assert dv.saturation_vapour_pressure(dv.T0) == pytest.approx(611.21)
    t = np.array([263.0, 293.0, 308.0])
    assert np.allclose(dv.relative_humidity(t, t), 100.0)
    assert np.allclose(dv.relative_humidity(t, dew_point(t, 40.0)), 40.0)
    q = dv.specific_humidity(np.array([280.0, 295.0]), 101325.0)
    assert np.all((q > 0) & (q < 0.03)) and q[1] > q[0]


def test_humidex_published_case():
    # Environment Canada: 30 degC with a 15 degC dew point gives a humidex of 34.
    assert dv.humidex(30 + K, 15 + K) == pytest.approx(34.0, abs=0.1)


@pytest.mark.parametrize("tf, rh, chart", [(90, 70, 106), (100, 40, 109), (86, 50, 88)])
def test_heat_index_nws_chart(tf, rh, chart):
    # NWS heat index chart (integer degF): tolerance 1 degF, the rounding of the chart.
    t = fahrenheit_to_kelvin(tf)
    got = dv.heat_index(np.array([t]), np.array([dew_point(t, rh)]))[0] * 9 / 5 + 32
    assert got == pytest.approx(chart, abs=1.0)


def rothfusz(tf, rh):
    """The NWS regression with its published coefficients (no adjustment applies below)."""
    return (
        -42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh
        - 0.00683783 * tf**2 - 0.05481717 * rh**2 + 0.00122874 * tf**2 * rh
        + 0.00085282 * tf * rh**2 - 0.00000199 * tf**2 * rh**2
    )  # fmt: skip


def test_heat_index_branch_is_decided_by_the_simple_value():
    # 78 degF at 100 %: the simple value is 80.2 >= 80 although T < 80, so the regression
    # applies (no adjustment below 80 degF); at 50 % the simple value 77.85 stands.
    t = fahrenheit_to_kelvin(78.0)
    got = dv.heat_index(np.array([t]), np.array([dew_point(t, 100.0)]))[0] * 9 / 5 + 32
    assert got == pytest.approx(rothfusz(78.0, 100.0), abs=0.01)
    got = dv.heat_index(np.array([t]), np.array([dew_point(t, 50.0)]))[0] * 9 / 5 + 32
    assert got == pytest.approx(0.5 * (78 + 61 + 10 * 1.2 + 50 * 0.094), abs=0.01)


def test_heat_index_below_80f_uses_the_simple_formula():
    t = fahrenheit_to_kelvin(70.0)
    got = dv.heat_index(np.array([t]), np.array([dew_point(t, 50.0)]))[0] * 9 / 5 + 32
    assert got == pytest.approx(0.5 * (70 + 61 + 2 * 1.2 + 50 * 0.094), abs=1e-6)


def test_apparent_temperature_terms():
    t, td = np.array([30 + K]), np.array([20 + K])
    calm = dv.apparent_temperature(t, td, np.array([0.0]), np.array([0.0]))
    windy = dv.apparent_temperature(t, td, np.array([3.0]), np.array([4.0]))
    assert windy - calm == pytest.approx(-0.70 * 5)  # 5 m s-1 at 10 m
    e_hpa = dv.saturation_vapour_pressure(td) / 100
    assert calm == pytest.approx(30 + 0.33 * e_hpa - 4.0)


def test_potential_temperature_and_terrain_mask():
    assert dv.potential_temperature(280.0, 1000.0) == 280.0
    assert dv.potential_temperature(250.0, 500.0) == pytest.approx(250.0 * 2**0.285691)
    # revision 1, L6: 850 hPa is below ground at sp = 70 000 Pa; NaN sp is never above
    mask = dv.above_ground(np.array([850.0, 850.0, 850.0]), np.array([70000.0, 101325.0, np.nan]))
    assert mask.tolist() == [False, True, False]


def test_shifted_day_for_nine_to_nine():
    assert dv.shifted_day("offset:-05:00", 9) == ("offset:-14:00", 0)
    assert dv.shifted_day("offset:+05:00", 9) == ("offset:-04:00", 0)
    assert dv.shifted_day("UTC", 9) == ("offset:-09:00", 0)
    assert dv.shifted_day("offset:-08:00", 9) == ("offset:+07:00", 1)


def test_unknown_or_incomplete_derived():
    with pytest.raises(ValueError, match="unknown derived"):
        dv.required(["utci"])
    assert dv.required(["at2m"]) == {"t2m", "d2m", "u10", "v10"}


# --- daily products ----------------------------------------------------------------------


def hourly_source(series: dict, first="2023-03-01", hours=72):
    """``get_hourly`` over synthetic hourly series given as functions of the UTC stamp."""
    times = pd.date_range(first, periods=hours, freq="1h")

    def get_hourly(var, sampling, start, end, definition):
        values = np.array([series[var](t) for t in times], dtype="float64")[:, None, None]
        return xr.Dataset(
            {var: (("time", "latitude", "longitude"), values)},
            coords={"time": times, "latitude": [0.0], "longitude": [0.0]},
        )

    return get_hourly


def one_day(ds, var):
    return float(ds[var].sel(time="2023-03-02").squeeze())


def test_daily_mean_temperature_three_versions():
    # Decision 7: t2m with 23 h at 20 degC and one at 40 degC; mx2t/mn2t with extremes 40/20.
    def t2m(t):
        return K + (40.0 if t == pd.Timestamp("2023-03-02 12:00") else 20.0)

    def mx2t(t):
        return K + (40.0 if t == pd.Timestamp("2023-03-02 15:00") else 25.0)

    def mn2t(t):
        return K + (20.0 if t == pd.Timestamp("2023-03-02 05:00") else 22.0)

    source = hourly_source({"t2m": t2m, "mx2t": mx2t, "mn2t": mn2t})
    day = date(2023, 3, 2)
    a = dv.daily("tmean_t2m", source, day, day, "UTC")
    b = dv.daily("tmean_txtn", source, day, day, "UTC")
    assert one_day(a, "tmean_t2m") - K == pytest.approx(20 + 20 / 24)  # 20.83 degC
    assert one_day(b, "tmean_txtn") - K == pytest.approx(30.0)
    # (b) must use mx2t/mn2t, not t2m: other extremes give 28 degC
    other = hourly_source(
        {"mx2t": lambda t: K + (38.0 if t.hour == 15 else 25.0),
         "mn2t": lambda t: K + (18.0 if t.hour == 5 else 22.0)}
    )  # fmt: skip
    assert one_day(dv.daily("tmean_txtn", other, day, day, "UTC"), "tmean_txtn") - K == 28.0
    # (c) 09:00-09:00: the 38/18 extremes of 2 March 15 h and 3 March 05 h pair up
    c = dv.daily("tmean_9to9", other, day, day, "UTC")
    assert one_day(c, "tmean_9to9") - K == pytest.approx(28.0)
    assert c["tmean_9to9"].attrs["use"] == "detection"
    assert int(c["tmean_9to9_mx2t_n_samples"].sel(time="2023-03-02").squeeze()) == 24


def test_daily_wind_speed_is_not_the_norm_of_the_means():
    source = hourly_source({"u10": lambda t: 1.0 if t.hour % 2 else -1.0, "v10": lambda t: 0.0})
    day = date(2023, 3, 2)
    speed = dv.daily("ws10", source, day, day, "UTC")
    assert one_day(speed, "ws10_mean") == pytest.approx(1.0)
    assert one_day(speed, "ws10_max") == pytest.approx(1.0)
    direction = dv.daily("wdir10", source, day, day, "UTC")
    assert np.isnan(one_day(direction, "wdir10"))  # the mean vector is zero


def test_net_heat_flux_keeps_ecmwf_signs():
    hour = 3600.0
    source = hourly_source(
        {"ssr": lambda t: 100 * hour, "str": lambda t: -50 * hour,
         "sshf": lambda t: -20 * hour, "slhf": lambda t: -30 * hour}
    )  # fmt: skip
    day = date(2023, 3, 2)
    qnet = dv.daily("qnet", source, day, day, "UTC")
    assert one_day(qnet, "qnet") == pytest.approx(0.0)
    assert qnet["qnet"].attrs["units"] == "W m-2"


def test_hourly_then_daily_for_heat_indices():
    # The daily maximum is taken on the hourly index, not computed from daily maxima.
    def t2m(t):
        return K + (35.0 if t.hour == 15 else 25.0)

    def d2m(t):
        return K + (10.0 if t.hour == 15 else 22.0)

    source = hourly_source({"t2m": t2m, "d2m": d2m})
    day = date(2023, 3, 2)
    hi = dv.daily("hi2m", source, day, day, "UTC")
    assert "-42.379" in hi["hi2m_max"].attrs["constants"]
    assert hi["hi2m_max"].attrs["catalogue"] == dv.CATALOGUE_VERSION
    assert int(hi["hi2m_n_samples"].sel(time="2023-03-02").squeeze()) == 24
    stamps = pd.date_range("2023-03-02", periods=24, freq="1h")
    hourly = dv.heat_index(np.array([t2m(t) for t in stamps]), np.array([d2m(t) for t in stamps]))
    assert one_day(hi, "hi2m_max") == pytest.approx(hourly.max())
    from_extremes = dv.heat_index(np.array([K + 35.0]), np.array([K + 22.0]))[0]
    assert one_day(hi, "hi2m_max") != pytest.approx(from_extremes)


# --- UTCI, WBGT and the mean radiant temperature (decision 8) -----------------------------

SIGMA = 5.67e-8  # the Stefan-Boltzmann constant thermofeel uses


def test_cos_solar_zenith_over_the_hour():
    stamps = np.array(["2023-03-20T12:00", "2023-12-21T12:00"], dtype="datetime64[s]")
    cosz = dv.cos_solar_zenith_hour(stamps, [0.0, 80.0], [0.0])[:, :, 0]
    assert cosz[0, 0] == pytest.approx(0.983, abs=0.005)  # equinox, equator, 11-12 UTC
    assert cosz[1, 1] == 0.0  # polar night
    # review C1: the sunrise hour is averaged over its sunlit part only (Di Napoli et al.
    # 2020); a one-second quadrature of the sunlit part gives 0.1125
    sunrise = np.array(["2023-03-20T07:00"], dtype="datetime64[s]")
    assert dv.cos_solar_zenith_hour(sunrise, [0.0], [0.0]).item() == pytest.approx(
        0.1125, abs=0.002
    )
    # review C2: 366 days in a leap year (NOAA): 0.07806 at 65 N, 2024-09-22 06-07 UTC
    leap = np.array(["2024-09-22T07:00"], dtype="datetime64[s]")
    assert dv.cos_solar_zenith_hour(leap, [65.0], [0.0]).item() == pytest.approx(0.07806, abs=5e-4)
    # the hour ending at the stamp: 06 UTC at 90 E is just after local noon
    morning = dv.cos_solar_zenith_hour(
        np.array(["2023-03-20T06:00"], "datetime64[s]"), [0.0], [90.0]
    )
    assert morning[0, 0, 0] > 0.95


def one_hour(**values):
    times = pd.date_range("2023-03-21 00:00", periods=1, freq="1h")
    coords = {"time": times, "latitude": [0.0], "longitude": [0.0]}
    return {
        k: xr.DataArray(np.full((1, 1, 1), v, float), dims=tuple(coords), coords=coords)
        for k, v in values.items()
    }


def reference_hour(t=293.15, rh=50.0, wind=0.5):
    """Night (no shortwave), downward longwave of a black body at the air temperature."""
    return one_hour(
        t2m=t, d2m=dew_point(t, rh), u10=wind, v10=0.0, ssrd=0.0, ssr=0.0, fdir=0.0,
        strd=SIGMA * t**4 * 3600, str=0.0,
    )  # fmt: skip


def test_mean_radiant_temperature_of_a_black_body_sky():
    pytest.importorskip("thermofeel")
    h = reference_hour()
    assert float(dv.mean_radiant_temperature(h).squeeze()) == pytest.approx(293.15, abs=1e-6)


def test_utci_reference_condition_and_wind():
    pytest.importorskip("thermofeel")
    # Broede et al. 2012: at 0.5 m s-1, MRT = Ta and 50 % RH the UTCI equals Ta (the
    # polynomial approximates it within a few tenths)
    assert float(dv.utci(reference_hour()).squeeze()) - K == pytest.approx(20.0, abs=0.3)
    windy = float(dv.utci(reference_hour(wind=5.0)).squeeze()) - K
    assert windy < 19.0


def test_wbgt_in_saturated_air_is_the_air_temperature():
    pytest.importorskip("thermofeel")
    # with RH 100 % and MRT = Ta, wet bulb and globe are both near Ta
    assert float(dv.wbgt(reference_hour(rh=100.0)).squeeze()) - K == pytest.approx(20.0, abs=0.5)


def test_utci_domain_indicator():
    pytest.importorskip("thermofeel")
    # review C4: the reference hour is inside the domain; a calm hour (< 0.5 m s-1) is not
    assert not bool(dv.utci_outside_domain(reference_hour()).squeeze())
    assert bool(dv.utci_outside_domain(reference_hour(wind=0.2)).squeeze())
    # re-review R1: a missing input is unknown, not inside the domain
    assert np.isnan(float(dv.utci_outside_domain(reference_hour(t=np.nan)).squeeze()))
    assert dv.SURFACE["wbgt2m"].use == "diagnostic"  # review C5
    assert dv.SURFACE["mrt"].sampling == "period_ending"  # review C3


def test_missing_thermofeel_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "thermofeel", None)
    with pytest.raises(ImportError, match="requirements-download"):
        dv.utci(reference_hour())
