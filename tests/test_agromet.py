"""Tests for agro-meteorological primitives (FAO-56 equations)."""

import pytest

from canola_dt.simulation import agromet


def test_extraterrestrial_radiation_summer_gt_winter_at_50n():
    summer = agromet.extraterrestrial_radiation(172, 50.0)   # ~ Jun 21
    winter = agromet.extraterrestrial_radiation(355, 50.0)   # ~ Dec 21
    assert summer > winter
    # Mid-summer Ra at mid-latitude is ~40-43 MJ m-2 day-1.
    assert 38.0 < summer < 44.0
    assert winter < 12.0


def test_daylength_symmetry_and_equator():
    # Summer + winter solstice daylengths at a latitude sum to ~24 h.
    summer = agromet.daylength(172, 50.0)
    winter = agromet.daylength(355, 50.0)
    assert summer == pytest.approx(24.0 - winter, abs=0.3)
    assert summer > 15.0 and winter < 9.0
    # Equator ~ 12 h year-round.
    assert agromet.daylength(100, 0.0) == pytest.approx(12.0, abs=0.1)


def test_solar_radiation_capped_and_increasing_with_range():
    ra = agromet.extraterrestrial_radiation(172, 50.0)
    rs_small = agromet.solar_radiation(172, 50.0, tmax_c=18, tmin_c=14)
    rs_big = agromet.solar_radiation(172, 50.0, tmax_c=28, tmin_c=8)
    assert rs_big > rs_small
    assert rs_big <= 0.75 * ra + 1e-9


def test_thermal_time_cardinal_points():
    # Below base and above max -> 0; equals (t_opt - t_base) at the optimum.
    assert agromet.thermal_time(3.0, 5, 20, 30) == 0.0
    assert agromet.thermal_time(31.0, 5, 20, 30) == 0.0
    assert agromet.thermal_time(20.0, 5, 20, 30) == pytest.approx(15.0)
    assert agromet.thermal_time(12.0, 5, 20, 30) == pytest.approx(7.0)
    # Midway down the decline (t=25): 15 * (30-25)/(30-20) = 7.5
    assert agromet.thermal_time(25.0, 5, 20, 30) == pytest.approx(7.5)


def test_hargreaves_et0_positive_and_hotter_is_higher():
    cool = agromet.hargreaves_et0(172, 50.0, tmax_c=20, tmin_c=8)
    hot = agromet.hargreaves_et0(172, 50.0, tmax_c=34, tmin_c=16)
    assert cool > 0
    assert hot > cool


def test_photoperiod_factor_bounds():
    assert agromet.photoperiod_factor(9.0, 10, 16) == 0.0
    assert agromet.photoperiod_factor(17.0, 10, 16) == 1.0
    assert agromet.photoperiod_factor(13.0, 10, 16) == pytest.approx(0.5)
