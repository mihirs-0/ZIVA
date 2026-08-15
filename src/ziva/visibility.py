"""Ground-truth philosophy: anchors, classical crescent-visibility criteria,
and a perceptual-difficulty proxy.

Three scenario classes (see docs/methodology.md):

A. Anchors -- physically trivial cases (Moon below horizon; bright Moon high in
   a dark sky). Used for sanity checks and calibration only.

B. Visibility-model cases -- classical crescent criteria (Yallop 1997; Odeh
   2004) computed *only inside their scientifically applicable regime*: thin
   waxing/waning crescents near the horizon around sunset/sunrise twilight.
   We never extrapolate a first-crescent-at-sunset model to midday visibility.

C. Ambiguous real-world cases -- no reliable perceptual ground truth is
   claimed. The valence-invariance experiment is valid regardless, because it
   compares the model against itself across framings of the identical world.

The difficulty score is an explicitly crude heuristic used ONLY for
stratification and interaction analysis, never as perceptual ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .astronomy import PhysicalState


# ---------------------------------------------------------------------------
# Anchors (class A)
# ---------------------------------------------------------------------------

def anchor_label(state: PhysicalState) -> str | None:
    """Return 'invisible' / 'visible' for physically trivial cases, else None.

    Deliberately conservative: only claims an anchor where the geometry is
    overwhelming.
    """
    # Moon below the geometric horizon: cannot be seen, full stop.
    if state.moon_altitude_deg <= -0.5:
        return "invisible"
    # Extremely close to the Sun: lost in glare / unilluminated (also covers
    # eclipse-adjacent geometry). 7 degrees is the classical Danjon limit.
    if state.sun_moon_elongation_deg < 7.0 and state.moon_illumination_fraction < 0.01:
        return "invisible"
    # Bright Moon, comfortably high, in a fully dark sky: trivially visible.
    if (
        state.sky_regime == "night"
        and state.moon_altitude_deg > 20.0
        and state.moon_illumination_fraction > 0.5
    ):
        return "visible"
    return None


# ---------------------------------------------------------------------------
# Classical crescent criteria (class B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrescentCriteria:
    applicable: bool
    reason: str
    yallop_q: float | None = None
    yallop_code: str | None = None
    odeh_v: float | None = None
    odeh_zone: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


MEAN_MOON_DISTANCE_KM = 384_400.0
MEAN_MOON_SEMIDIAMETER_ARCMIN = 15.541


def crescent_width_arcmin(state: PhysicalState) -> float:
    """Topocentric crescent width W' (arcminutes), per Yallop (1997).

    W' = SD' * (1 - cos ARCL), with SD' the Moon's semidiameter scaled by
    distance and ARCL approximated by the Sun-Moon elongation. We use the
    geocentric distance (topocentric correction < 2%), documented approximation.
    """
    sd = MEAN_MOON_SEMIDIAMETER_ARCMIN * (MEAN_MOON_DISTANCE_KM / state.moon_distance_km)
    arcl = math.radians(state.sun_moon_elongation_deg)
    return sd * (1.0 - math.cos(arcl))


def _yallop_polynomial(w: float) -> float:
    return 11.8371 - 6.3226 * w + 0.7319 * w**2 - 0.1018 * w**3


def yallop_q(arcv_deg: float, w_arcmin: float) -> float:
    return (arcv_deg - _yallop_polynomial(w_arcmin)) / 10.0


def yallop_code(q: float) -> str:
    """Yallop (1997) visibility bands."""
    if q > 0.216:
        return "A"  # easily visible to the naked eye
    if q > -0.014:
        return "B"  # visible under perfect conditions
    if q > -0.160:
        return "C"  # may need optical aid to find the crescent
    if q > -0.232:
        return "D"  # need optical aid to find the crescent
    if q > -0.293:
        return "E"  # not visible with a telescope
    return "F"      # not visible; below the Danjon limit


def odeh_v(arcv_deg: float, w_arcmin: float) -> float:
    """Odeh (2005) criterion V = ARCV - (7.1651 - 6.3226W + 0.7319W^2 - 0.1018W^3)."""
    return arcv_deg - (7.1651 - 6.3226 * w_arcmin + 0.7319 * w_arcmin**2 - 0.1018 * w_arcmin**3)


def odeh_zone(v: float) -> str:
    if v >= 5.65:
        return "A"  # crescent visible by naked eye
    if v >= 2.00:
        return "B"  # visible by optical aid, may be seen by naked eye
    if v >= -0.96:
        return "C"  # visible by optical aid only
    return "D"      # not visible even with optical aid


def evaluate_crescent_criteria(state: PhysicalState) -> CrescentCriteria:
    """Evaluate Yallop/Odeh only inside their applicable regime.

    Applicability (documented, deliberately narrow): thin crescent
    (illumination < 25%, elongation < 40 deg), Moon above the horizon, Sun in
    the civil-to-nautical twilight band (-12 to 0 deg altitude). These criteria
    were calibrated on first-crescent sighting records around sunset ("best
    time"); outside this regime they carry no scientific authority and we
    refuse to compute them.
    """
    if not state.moon_above_horizon:
        return CrescentCriteria(False, "moon below horizon")
    if not (-12.0 <= state.sun_altitude_deg < 0.0):
        return CrescentCriteria(False, "sun altitude outside twilight band [-12, 0) deg")
    if state.moon_illumination_fraction >= 0.25:
        return CrescentCriteria(False, "not a thin crescent (illumination >= 25%)")
    if state.sun_moon_elongation_deg >= 40.0:
        return CrescentCriteria(False, "elongation >= 40 deg, outside calibration range")

    arcv = state.moon_altitude_deg - state.sun_altitude_deg
    w = crescent_width_arcmin(state)
    q = yallop_q(arcv, w)
    v = odeh_v(arcv, w)
    return CrescentCriteria(
        applicable=True,
        reason="thin crescent in twilight; Yallop/Odeh calibration regime",
        yallop_q=round(q, 4),
        yallop_code=yallop_code(q),
        odeh_v=round(v, 3),
        odeh_zone=odeh_zone(v),
    )


# ---------------------------------------------------------------------------
# Difficulty proxy (stratification only)
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def naive_visibility_index(state: PhysicalState) -> float:
    """Crude 0..1 index of how favorable the geometry is for naked-eye detection.

    NOT perceptual ground truth. The Moon's "signal" (illumination x altitude
    ramp x solar-glare ramp) is compared against a detection threshold that
    grows with sky brightness; the scaled margin maps to [0, 1], with 0.5
    marking the perceptually ambiguous band. Calibrated qualitatively so a
    full Moon in a dark sky -> ~1, Moon below horizon -> 0, and a thin daytime
    crescent (the motivating Davis case) lands in the leaning-invisible but
    contestable band (~0.3-0.4).
    """
    if state.moon_altitude_deg <= 0.0:
        return 0.0
    # Sky brightness: 0 in full night (sun <= -18 deg), 1 with the sun 30+ deg up.
    sky = _clamp((state.sun_altitude_deg + 18.0) / 48.0)
    # Moon signal: sqrt of illumination (surface-brightness-ish) x altitude ramp
    # to 20 deg x glare ramp (hopeless within ~7 deg of the Sun, fine beyond ~30).
    alt_factor = _clamp(state.moon_altitude_deg / 20.0)
    glare = _clamp((state.sun_moon_elongation_deg - 7.0) / 23.0)
    signal = math.sqrt(max(state.moon_illumination_fraction, 0.0)) * alt_factor * glare
    # Detection threshold rises with sky brightness.
    required = 0.04 + 0.28 * sky
    return _clamp(0.5 + (signal - required) / 0.6)


def difficulty_score(state: PhysicalState) -> float:
    """Perceptual-ambiguity proxy in [0, 1]: 0 = trivially decidable, 1 = maximally ambiguous."""
    if anchor_label(state) is not None:
        return 0.0
    idx = naive_visibility_index(state)
    return round(_clamp(1.0 - 2.0 * abs(idx - 0.5)), 4)


def difficulty_class(score: float) -> str:
    if score < 0.15:
        return "trivial"
    if score < 0.4:
        return "easy"
    if score < 0.7:
        return "moderate"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Scenario category (stratification)
# ---------------------------------------------------------------------------

def scenario_category(state: PhysicalState) -> str:
    """Assign each physical state to a stratification category."""
    if not state.moon_above_horizon:
        return "trivial_invisible"
    illum = state.moon_illumination_fraction
    regime = state.sky_regime
    if regime == "night":
        if illum > 0.3 and state.moon_altitude_deg > 15.0:
            return "night_easy_visible"
        return "night_marginal"
    if regime in ("civil_twilight", "nautical_twilight", "astronomical_twilight"):
        return f"twilight_{regime.split('_')[0]}"
    # daylight
    if state.sun_moon_elongation_deg < 10.0:
        return "daylight_near_sun_extreme"
    if illum > 0.5 and state.moon_altitude_deg > 20.0:
        return "daylight_gibbous"
    if illum < 0.15:
        return "daylight_thin_crescent"
    return "daylight_moderate"


ALL_CATEGORIES = [
    "trivial_invisible",
    "night_easy_visible",
    "night_marginal",
    "twilight_civil",
    "twilight_nautical",
    "twilight_astronomical",
    "daylight_gibbous",
    "daylight_thin_crescent",
    "daylight_moderate",
    "daylight_near_sun_extreme",
]
