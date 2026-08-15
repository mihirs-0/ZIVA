from datetime import datetime, timezone

import pytest

from ziva.astronomy import classify_sky_regime, compute_physical_state
from ziva.visibility import (
    anchor_label,
    difficulty_score,
    evaluate_crescent_criteria,
    naive_visibility_index,
    scenario_category,
    yallop_code,
)


def test_equatorial_noon_sun_high():
    st = compute_physical_state(0.0, 0.0, datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    assert st.sun_altitude_deg > 80  # near-zenith sun at the equinox on the equator/meridian
    assert st.sky_regime == "daylight"


def test_midnight_sun_below_horizon():
    st = compute_physical_state(0.0, 0.0, datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc))
    assert st.sun_altitude_deg < -60
    assert st.sky_regime == "night"


def test_full_moon_near_180_elongation():
    # 2026-08-28 is close to full moon
    st = compute_physical_state(38.5, -121.7, datetime(2026, 8, 28, 8, 0, tzinfo=timezone.utc))
    assert st.moon_illumination_fraction > 0.95
    assert st.sun_moon_elongation_deg > 150


def test_davis_case_reconstruction():
    """The motivating scenario: thin waxing crescent in bright daylight."""
    st = compute_physical_state(38.5449, -121.7405, datetime(2026, 8, 14, 22, 0, tzinfo=timezone.utc), 16)
    assert 0.03 < st.moon_illumination_fraction < 0.10
    assert st.waxing
    assert st.sky_regime == "daylight"
    assert 20 < st.sun_moon_elongation_deg < 40
    assert st.moon_above_horizon


def test_illumination_and_ranges():
    st = compute_physical_state(51.5, 0.0, datetime(2026, 6, 1, 3, 0, tzinfo=timezone.utc))
    assert 0.0 <= st.moon_illumination_fraction <= 1.0
    assert -90 <= st.moon_altitude_deg <= 90
    assert 0 <= st.moon_azimuth_deg < 360
    assert 350_000 < st.moon_distance_km < 410_000


def test_sky_regime_boundaries():
    assert classify_sky_regime(5) == "daylight"
    assert classify_sky_regime(-3) == "civil_twilight"
    assert classify_sky_regime(-9) == "nautical_twilight"
    assert classify_sky_regime(-15) == "astronomical_twilight"
    assert classify_sky_regime(-30) == "night"


def _state(**over):
    from ziva.astronomy import PhysicalState

    base = dict(
        timestamp_utc="2026-01-01T00:00:00Z", latitude_deg=0, longitude_deg=0, elevation_m=0,
        sun_altitude_deg=-30, sun_azimuth_deg=90, moon_altitude_deg=45, moon_azimuth_deg=180,
        moon_illumination_fraction=0.9, moon_phase_angle_deg=30, moon_phase_deg=150,
        sun_moon_elongation_deg=150, moon_distance_km=384400.0, moon_above_horizon=True,
        sun_above_horizon=False, sky_regime="night", waxing=True,
    )
    base.update(over)
    return PhysicalState(**base)


def test_moon_below_horizon_anchor():
    st = _state(moon_altitude_deg=-10, moon_above_horizon=False)
    assert anchor_label(st) == "invisible"
    assert naive_visibility_index(st) == 0.0
    assert difficulty_score(st) == 0.0
    assert scenario_category(st) == "trivial_invisible"


def test_bright_night_moon_anchor():
    st = _state()
    assert anchor_label(st) == "visible"
    assert scenario_category(st) == "night_easy_visible"


def test_danjon_limit_anchor():
    st = _state(sun_moon_elongation_deg=3, moon_illumination_fraction=0.001,
                sun_altitude_deg=10, sky_regime="daylight")
    assert anchor_label(st) == "invisible"


def test_crescent_criteria_regime_gating():
    # daylight -> not applicable
    day = _state(sun_altitude_deg=40, sky_regime="daylight", moon_illumination_fraction=0.06,
                 sun_moon_elongation_deg=25, moon_altitude_deg=30)
    assert not evaluate_crescent_criteria(day).applicable
    # twilight thin crescent -> applicable, sane values
    tw = _state(sun_altitude_deg=-5, sky_regime="civil_twilight", moon_illumination_fraction=0.03,
                sun_moon_elongation_deg=15, moon_altitude_deg=8)
    cc = evaluate_crescent_criteria(tw)
    assert cc.applicable
    assert cc.yallop_code in list("ABCDEF")
    assert cc.odeh_zone in list("ABCD")


@pytest.mark.parametrize("q,code", [(0.3, "A"), (0.1, "B"), (-0.1, "C"), (-0.2, "D"), (-0.25, "E"), (-0.5, "F")])
def test_yallop_bands(q, code):
    assert yallop_code(q) == code
