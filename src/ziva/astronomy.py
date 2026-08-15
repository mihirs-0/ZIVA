"""Local astronomy backend.

Computes the physical state of the Sun and Moon for an observer at a given
location and time, using `astronomy-engine` (Don Cross's Astronomy Engine,
MIT-licensed). Astronomy Engine uses built-in analytic models (VSOP87 for the
Sun/planets, an ELP2000-derived lunar theory) accurate to roughly one
arcminute -- no ephemeris download or network access is required, which makes
scenario generation fully reproducible and offline.

The physical quantities computed here are the ground truth of the benchmark's
scenario records. Models under evaluation never get to alter them.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import astronomy

ASTRONOMY_LIBRARY = "astronomy-engine"


def astronomy_library_version() -> str:
    try:
        return importlib.metadata.version("astronomy-engine")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


@dataclass(frozen=True)
class PhysicalState:
    """The full computed sky state for one observer/time. All angles in degrees."""

    timestamp_utc: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    sun_altitude_deg: float
    sun_azimuth_deg: float
    moon_altitude_deg: float
    moon_azimuth_deg: float
    moon_illumination_fraction: float
    moon_phase_angle_deg: float          # Sun-Moon phase angle (0 = full, 180 = new)
    moon_phase_deg: float                # ecliptic longitude difference (0 = new, 180 = full)
    sun_moon_elongation_deg: float       # angular separation of Moon from Sun
    moon_distance_km: float
    moon_above_horizon: bool
    sun_above_horizon: bool
    sky_regime: str                      # daylight | civil_twilight | nautical_twilight |
    #                                      astronomical_twilight | night
    waxing: bool

    def to_dict(self) -> dict:
        return asdict(self)


def classify_sky_regime(sun_altitude_deg: float) -> str:
    """Deterministic sky-brightness regime from solar altitude (standard definitions)."""
    if sun_altitude_deg >= 0.0:
        return "daylight"
    if sun_altitude_deg >= -6.0:
        return "civil_twilight"
    if sun_altitude_deg >= -12.0:
        return "nautical_twilight"
    if sun_altitude_deg >= -18.0:
        return "astronomical_twilight"
    return "night"


def _to_astro_time(dt: datetime) -> astronomy.Time:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    dt = dt.astimezone(timezone.utc)
    return astronomy.Time.Make(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def compute_physical_state(
    latitude_deg: float,
    longitude_deg: float,
    when_utc: datetime,
    elevation_m: float = 0.0,
) -> PhysicalState:
    """Compute the topocentric Sun/Moon state for an observer.

    Altitudes use normal atmospheric refraction, matching what a human
    observer would experience near the horizon.
    """
    t = _to_astro_time(when_utc)
    obs = astronomy.Observer(latitude_deg, longitude_deg, elevation_m)

    def horizon(body: astronomy.Body) -> tuple[float, float, float]:
        eq = astronomy.Equator(body, t, obs, ofdate=True, aberration=True)
        hor = astronomy.Horizon(t, obs, eq.ra, eq.dec, astronomy.Refraction.Normal)
        return hor.altitude, hor.azimuth, eq.dist

    sun_alt, sun_az, _ = horizon(astronomy.Body.Sun)
    moon_alt, moon_az, moon_dist_au = horizon(astronomy.Body.Moon)

    illum = astronomy.Illumination(astronomy.Body.Moon, t)
    elongation = astronomy.AngleFromSun(astronomy.Body.Moon, t)
    phase_deg = astronomy.MoonPhase(t)  # 0 = new, 90 = first quarter, 180 = full

    return PhysicalState(
        timestamp_utc=when_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        latitude_deg=round(latitude_deg, 5),
        longitude_deg=round(longitude_deg, 5),
        elevation_m=round(elevation_m, 1),
        sun_altitude_deg=round(sun_alt, 3),
        sun_azimuth_deg=round(sun_az, 3),
        moon_altitude_deg=round(moon_alt, 3),
        moon_azimuth_deg=round(moon_az, 3),
        moon_illumination_fraction=round(illum.phase_fraction, 5),
        moon_phase_angle_deg=round(illum.phase_angle, 3),
        moon_phase_deg=round(phase_deg, 3),
        sun_moon_elongation_deg=round(elongation, 3),
        moon_distance_km=round(moon_dist_au * astronomy.KM_PER_AU, 1),
        moon_above_horizon=moon_alt > 0.0,
        sun_above_horizon=sun_alt > 0.0,
        sky_regime=classify_sky_regime(sun_alt),
        waxing=phase_deg < 180.0,
    )
