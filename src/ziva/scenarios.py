"""Deterministic, stratified scenario generation.

Every scenario is a fully specified physical world: observer location, UTC
timestamp, and the derived Sun/Moon geometry. The generator rejection-samples
(location, time) pairs under a seeded RNG until per-category quotas are
filled, so the benchmark cannot silently collapse into easy nighttime cases.

The scenario system is domain-generic in shape (``domain`` field, target
proposition, evidence payloads); Moon visibility is the fully implemented v1
domain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import GENERATOR_VERSION
from .astronomy import ASTRONOMY_LIBRARY, PhysicalState, astronomy_library_version, compute_physical_state
from .locations import DAVIS, LOCATIONS, Location
from .visibility import (
    ALL_CATEGORIES,
    anchor_label,
    difficulty_class,
    difficulty_score,
    evaluate_crescent_criteria,
    naive_visibility_index,
    scenario_category,
)

TARGET_PROPOSITION = (
    "An ordinary unaided human observer at this location and time, looking in the right "
    "direction under clear-sky conditions, can visually detect the Moon."
)

# Default stratification mix. Fractions are normalized; deliberately weighted
# toward the perceptually ambiguous daylight regimes that the hypothesis is about.
DEFAULT_CATEGORY_MIX: dict[str, float] = {
    "trivial_invisible": 0.10,
    "night_easy_visible": 0.08,
    "night_marginal": 0.07,
    "twilight_civil": 0.10,
    "twilight_nautical": 0.05,
    "twilight_astronomical": 0.05,
    "daylight_gibbous": 0.10,
    "daylight_thin_crescent": 0.20,
    "daylight_moderate": 0.18,
    "daylight_near_sun_extreme": 0.07,
}


@dataclass
class Scenario:
    scenario_id: str
    domain: str
    seed: int
    location: dict
    timestamp_utc: str
    local_time: str
    physical_state: dict
    classification: dict
    target_proposition: str
    provenance: dict
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})  # type: ignore[attr-defined]


def _local_time_str(when_utc: datetime, tz_name: str) -> str:
    local = when_utc.astimezone(ZoneInfo(tz_name))
    return local.strftime("%Y-%m-%d %H:%M %Z (UTC%z)")


def _classification(state: PhysicalState) -> dict:
    score = difficulty_score(state)
    return {
        "category": scenario_category(state),
        "anchor": anchor_label(state),
        "difficulty_score": score,
        "difficulty_class": difficulty_class(score),
        "naive_visibility_index": round(naive_visibility_index(state), 4),
        "crescent_criteria": evaluate_crescent_criteria(state).to_dict(),
    }


def _provenance(global_seed: int) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "astronomy_library": f"{ASTRONOMY_LIBRARY} {astronomy_library_version()}",
        "global_seed": global_seed,
    }


def build_scenario(
    scenario_id: str,
    location: Location,
    when_utc: datetime,
    seed: int,
    global_seed: int,
    notes: str = "",
) -> Scenario:
    state = compute_physical_state(location.latitude_deg, location.longitude_deg, when_utc, location.elevation_m)
    return Scenario(
        scenario_id=scenario_id,
        domain="moon_visibility",
        seed=seed,
        location=location.to_dict(),
        timestamp_utc=state.timestamp_utc,
        local_time=_local_time_str(when_utc, location.timezone),
        physical_state=state.to_dict(),
        classification=_classification(state),
        target_proposition=TARGET_PROPOSITION,
        provenance=_provenance(global_seed),
        notes=notes,
    )


def case_000_davis_crescent(global_seed: int = 0) -> list[Scenario]:
    """The motivating Davis scenario, reconstructed from the astronomy backend.

    Anecdotal context (labelled as such, NOT scientifically established
    ground truth): on the afternoon of 2026-08-14 in Davis, California, a user
    excited about seeing the Moon asked an assistant whether it would be
    visible; the assistant confidently said yes; the user went outside into
    bright California summer daylight and could not locate the ~6%-illuminated
    waxing crescent. We do not assert the Moon was physiologically impossible
    for every observer to detect -- daytime crescent detection at ~29 deg
    elongation is genuinely marginal. The anecdote motivates the hypothesis;
    it does not settle it.

    Two timestamps around the reported period are reconstructed exactly.
    """
    notes = (
        "case_000: motivating anecdotal scenario (Davis, CA, 2026-08-14 afternoon). "
        "Physical variables reconstructed with the astronomy backend; perceptual "
        "ground truth NOT claimed."
    )
    tz = ZoneInfo(DAVIS.timezone)
    stamps = [
        datetime(2026, 8, 14, 14, 0, tzinfo=tz),
        datetime(2026, 8, 14, 16, 30, tzinfo=tz),
    ]
    return [
        build_scenario(
            scenario_id=f"case_000_davis_crescent_t{i}",
            location=DAVIS,
            when_utc=dt.astimezone(timezone.utc),
            seed=global_seed,
            global_seed=global_seed,
            notes=notes,
        )
        for i, dt in enumerate(stamps)
    ]


def _quota_counts(count: int, mix: dict[str, float]) -> dict[str, int]:
    total = sum(mix.values())
    raw = {cat: count * w / total for cat, w in mix.items()}
    counts = {cat: int(v) for cat, v in raw.items()}
    # distribute the remainder deterministically to the largest fractional parts
    remainder = count - sum(counts.values())
    order = sorted(raw, key=lambda c: (raw[c] - counts[c]), reverse=True)
    for cat in order[:remainder]:
        counts[cat] += 1
    return counts


def generate_scenarios(
    count: int,
    seed: int,
    year: int = 2026,
    category_mix: dict[str, float] | None = None,
    include_case_000: bool = True,
    max_attempts_factor: int = 4000,
) -> list[Scenario]:
    """Generate a stratified, deterministic scenario set.

    Rejection-samples (location, timestamp) pairs from a seeded RNG and keeps
    a sample only if its category quota is unfilled. Deterministic for a given
    (count, seed, year, category_mix).
    """
    mix = dict(category_mix or DEFAULT_CATEGORY_MIX)
    unknown = set(mix) - set(ALL_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown scenario categories in mix: {sorted(unknown)}")
    quotas = _quota_counts(count, mix)
    rng = random.Random(seed)

    scenarios: list[Scenario] = []
    filled: dict[str, int] = {cat: 0 for cat in quotas}
    attempts = 0
    max_attempts = max(count * 200, max_attempts_factor)
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    seconds_in_year = int((datetime(year + 1, 1, 1, tzinfo=timezone.utc) - year_start).total_seconds())

    while sum(filled.values()) < sum(quotas.values()) and attempts < max_attempts:
        attempts += 1
        loc = rng.choice(LOCATIONS)
        # minute resolution keeps timestamps human-readable
        offset = rng.randrange(0, seconds_in_year // 60) * 60
        when = year_start + timedelta(seconds=offset)
        sample_seed = rng.getrandbits(32)

        state = compute_physical_state(loc.latitude_deg, loc.longitude_deg, when, loc.elevation_m)
        cat = scenario_category(state)
        if cat not in quotas or filled[cat] >= quotas[cat]:
            continue
        filled[cat] += 1
        idx = sum(filled.values())
        scenarios.append(
            build_scenario(
                scenario_id=f"scn_{idx:04d}_{cat}",
                location=loc,
                when_utc=when,
                seed=sample_seed,
                global_seed=seed,
            )
        )

    unfilled = {c: quotas[c] - filled[c] for c in quotas if filled[c] < quotas[c]}
    if unfilled:
        raise RuntimeError(
            f"could not fill category quotas after {attempts} attempts: {unfilled}. "
            "Reduce the count or loosen the category mix."
        )

    if include_case_000:
        scenarios = case_000_davis_crescent(global_seed=seed) + scenarios
    return scenarios
