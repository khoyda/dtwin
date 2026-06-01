"""Agro-meteorological primitives for the process-based crop model.

Pure functions (no state) implementing standard FAO-56 / crop-modelling equations:
extraterrestrial radiation, daylength, a temperature-based solar-radiation estimate,
Hargreaves reference ET, and cardinal-temperature thermal time. Kept separate so they
can be unit-tested against published reference values.

References: Allen et al. (1998) FAO Irrigation & Drainage Paper 56.
"""

from __future__ import annotations

import math

GSC = 0.0820          # solar constant, MJ m-2 min-1
MJ_TO_MM = 0.408      # latent-heat conversion: 1 MJ m-2 day-1 ~ 0.408 mm day-1
PAR_FRACTION = 0.48   # fraction of incoming shortwave that is photosynthetically active


def extraterrestrial_radiation(doy: int, latitude_deg: float) -> float:
    """Daily extraterrestrial radiation Ra (MJ m-2 day-1), FAO-56 eq. 21."""
    phi = math.radians(latitude_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)        # eq. 23
    decl = 0.409 * math.sin(2.0 * math.pi / 365.0 * doy - 1.39)     # eq. 24
    # Sunset hour angle (eq. 25), with clamping for polar day/night.
    x = -math.tan(phi) * math.tan(decl)
    x = max(-1.0, min(1.0, x))
    ws = math.acos(x)
    ra = (24.0 * 60.0 / math.pi) * GSC * dr * (
        ws * math.sin(phi) * math.sin(decl)
        + math.cos(phi) * math.cos(decl) * math.sin(ws)
    )
    return max(0.0, ra)


def daylength(doy: int, latitude_deg: float) -> float:
    """Astronomical daylength N (hours), FAO-56 eq. 34."""
    phi = math.radians(latitude_deg)
    decl = 0.409 * math.sin(2.0 * math.pi / 365.0 * doy - 1.39)
    x = -math.tan(phi) * math.tan(decl)
    x = max(-1.0, min(1.0, x))
    ws = math.acos(x)
    return 24.0 / math.pi * ws


def solar_radiation(doy: int, latitude_deg: float, tmax_c: float, tmin_c: float,
                    krs: float = 0.16) -> float:
    """Estimate incoming shortwave Rs from temperature range (Hargreaves, eq. 50).

    ``krs`` ~ 0.16 for interior locations, 0.19 for coastal. Capped at clear-sky
    radiation Rso = 0.75 * Ra.
    """
    ra = extraterrestrial_radiation(doy, latitude_deg)
    dt = max(0.0, tmax_c - tmin_c)
    rs = krs * math.sqrt(dt) * ra
    return min(rs, 0.75 * ra)


def hargreaves_et0(doy: int, latitude_deg: float, tmax_c: float, tmin_c: float,
                   tmean_c: float | None = None) -> float:
    """Reference evapotranspiration ET0 (mm day-1), Hargreaves (FAO-56 eq. 52).

    Temperature-only method — appropriate when humidity/wind/radiation are missing,
    as with ECCC daily station data.
    """
    if tmean_c is None:
        tmean_c = (tmax_c + tmin_c) / 2.0
    ra_mm = extraterrestrial_radiation(doy, latitude_deg) * MJ_TO_MM
    dt = max(0.0, tmax_c - tmin_c)
    return max(0.0, 0.0023 * (tmean_c + 17.8) * math.sqrt(dt) * ra_mm)


def thermal_time(tmean_c: float, t_base: float, t_opt: float, t_max: float) -> float:
    """Daily thermal time (deg-day) with cardinal temperatures (broken-linear).

    Rises linearly from ``t_base`` to ``t_opt`` (where it equals t_opt - t_base),
    then declines linearly to zero at ``t_max``. Below base or above max -> 0.
    """
    t = tmean_c
    if t <= t_base or t >= t_max:
        return 0.0
    if t <= t_opt:
        return t - t_base
    # Linear decline t_opt -> t_max, scaled so it is continuous at t_opt.
    return (t_opt - t_base) * (t_max - t) / (t_max - t_opt)


def photoperiod_factor(day_hours: float, pp_base: float, pp_opt: float) -> float:
    """Development multiplier for a long-day crop (canola), in [0, 1].

    0 at/below ``pp_base`` hours, 1 at/above ``pp_opt`` hours, linear between.
    """
    if pp_opt == pp_base:
        return 1.0
    return max(0.0, min(1.0, (day_hours - pp_base) / (pp_opt - pp_base)))
