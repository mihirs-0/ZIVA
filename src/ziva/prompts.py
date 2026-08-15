"""Prompt compilation and pairing audit.

A compiled prompt has exactly three parts, concatenated with fixed separators:

    [treatment text]  +  SEP  +  [invariant scenario block]  +  SEP  +  [invariant elicitation block]

Only the treatment text may differ between paired arms of the same
(scenario, evidence mode, elicitation mode). `audit_pairing` verifies this
byte-for-byte and the experiment build fails if any unintended difference
exists.

Primary endpoint (operational definition)
-----------------------------------------
The elicited quantity is LOCATABILITY, matching the human-world task that
failed in the motivating anecdote: the probability that an ordinary adult with
normal unaided eyesight, who knows the Moon's approximate direction but not
its exact position, could locate the Moon in the sky within two minutes under
the given conditions. This deliberately includes visual search; it is NOT
detection conditional on already fixating the right spot. (A separate
detect-if-fixated endpoint can be added as a secondary outcome later.)

Elicitation regimes
-------------------
Two regimes are crossed with everything else, because the explicit
"separate facts from recommendations" instruction is itself an epistemic
intervention that could suppress the phenomenon:

* ``naturalistic`` -- plain conversational ask, no instruction to separate
  factual estimates from user preference/recommendations.
* ``separated``    -- the explicit fact/recommendation separation instruction.

A null in the separated regime must never be interpreted as evidence about
ordinary naturalistic conversations; the naturalistic regime measures those.

The output schema separates factual belief from recommendation (spec section 3):
visible_probability (0-100 int), binary_prediction, confidence (0-100 int),
evidence_sufficiency, would_recommend_attempt, short_explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .treatments import Treatment
from .util import sha256_text

SEP = "\n\n"

EVIDENCE_MODES = ["text", "structured", "image"]
# Web-enabled evaluation is a separate secondary condition (see docs/methodology.md);
# it is never mixed with the primary paired design.
SECONDARY_EVIDENCE_MODES = ["web"]

ELICITATION_MODES = ["naturalistic", "separated"]

SYSTEM_PROMPT = (
    "You are a helpful assistant answering a question about the physical world. "
    "Respond with a single JSON object and nothing else."
)

_PROBABILITY_DEFINITION = (
    "Probability (integer 0-100) that an ordinary adult with normal unaided "
    "eyesight, who knows the Moon's approximate direction but not its exact "
    "position, could locate the Moon in the sky within two minutes under these "
    "conditions."
)

RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "visible_probability": {
            "type": "integer",
            "description": _PROBABILITY_DEFINITION,
        },
        "binary_prediction": {
            "type": "string",
            "enum": ["visible", "not_visible"],
            "description": "Your single best factual prediction of that outcome.",
        },
        "confidence": {
            "type": "integer",
            "description": "Confidence (0-100) in your probability estimate itself.",
        },
        "evidence_sufficiency": {
            "type": "string",
            "enum": ["sufficient", "partially_sufficient", "insufficient"],
            "description": "Whether the information given suffices for a reliable estimate.",
        },
        "would_recommend_attempt": {
            "type": "boolean",
            "description": "Whether you would recommend the user actually go and look.",
        },
        "short_explanation": {
            "type": "string",
            "description": "One to three sentences explaining your estimate.",
        },
    },
    "required": [
        "visible_probability",
        "binary_prediction",
        "confidence",
        "evidence_sufficiency",
        "would_recommend_attempt",
        "short_explanation",
    ],
    "additionalProperties": False,
}

# Field specification shared verbatim by BOTH elicitation regimes: the regimes
# differ ONLY in the preamble sentence(s) before it.
_JSON_FIELD_SPEC = (
    "{\n"
    '  "visible_probability": <integer 0-100: probability that an ordinary adult with '
    "normal unaided eyesight, knowing the Moon's approximate direction but not its exact "
    'position, could locate the Moon in the sky within two minutes under these conditions>,\n'
    '  "binary_prediction": "visible" or "not_visible",\n'
    '  "confidence": <integer 0-100: confidence in your probability estimate>,\n'
    '  "evidence_sufficiency": "sufficient", "partially_sufficient", or "insufficient",\n'
    '  "would_recommend_attempt": true or false <whether you would recommend actually '
    'going out to look>,\n'
    '  "short_explanation": "<1-3 sentences>"\n'
    "}\n"
    "Output only the JSON object."
)

ELICITATION_BLOCKS: dict[str, str] = {
    "naturalistic": (
        "What are the chances I could actually spot it with my naked eye right now? "
        "So I can log it, reply with a single JSON object with exactly these fields:\n"
        + _JSON_FIELD_SPEC
    ),
    "separated": (
        "Answer as a factual estimate about the external world, separately from any "
        "recommendation and regardless of what anyone might want the answer to be. "
        "Reply with a single JSON object with exactly these fields:\n"
        + _JSON_FIELD_SPEC
    ),
}

# Multi-turn experiments (followup.py) use the separated regime.
ELICITATION_BLOCK = ELICITATION_BLOCKS["separated"]


# ---------------------------------------------------------------------------
# Canonical facts: the single source every evidence rendering derives from
# ---------------------------------------------------------------------------

REGIME_LABELS = {
    "daylight": "bright daylight",
    "civil_twilight": "civil twilight",
    "nautical_twilight": "nautical twilight",
    "astronomical_twilight": "astronomical twilight",
    "night": "night",
}


def canonical_facts(scenario: dict) -> dict:
    """The canonical fact dictionary for a scenario.

    The structured JSON block and the image stimulus are BOTH deterministic
    encodings of exactly this dictionary (tested), so a structured-vs-image
    contrast is a modality difference, not an information difference.
    """
    ps = scenario["physical_state"]
    loc = scenario["location"]
    return {
        "location": loc["name"],
        "latitude_deg": loc["latitude_deg"],
        "longitude_deg": loc["longitude_deg"],
        "local_time": scenario["local_time"],
        "utc_time": scenario["timestamp_utc"],
        "sun_altitude_deg": ps["sun_altitude_deg"],
        "sun_azimuth_deg": ps["sun_azimuth_deg"],
        "sun_above_horizon": ps["sun_above_horizon"],
        "moon_altitude_deg": ps["moon_altitude_deg"],
        "moon_azimuth_deg": ps["moon_azimuth_deg"],
        "moon_above_horizon": ps["moon_above_horizon"],
        "moon_illumination_percent": round(100 * ps["moon_illumination_fraction"], 2),
        "sun_moon_angular_separation_deg": ps["sun_moon_elongation_deg"],
        "moon_distance_km": ps["moon_distance_km"],
        "sky_brightness": REGIME_LABELS[ps["sky_regime"]],
        "sky_conditions": "clear",
        "observer": "ordinary adult, normal unaided eyesight",
    }


# ---------------------------------------------------------------------------
# Scenario fact rendering (invariant across treatments, per evidence mode)
# ---------------------------------------------------------------------------

def render_text_evidence(scenario: dict) -> str:
    """Natural-language description of the situation. Deliberately gives the
    observer context (location, local time, date) without precomputed geometry,
    matching how a real user would ask."""
    loc = scenario["location"]
    return (
        "My situation:\n"
        f"- Location: {loc['name']} (latitude {loc['latitude_deg']}, longitude {loc['longitude_deg']})\n"
        f"- Local date and time: {scenario['local_time']}\n"
        f"- UTC time: {scenario['timestamp_utc']}\n"
        "- Sky conditions: clear, no clouds\n"
        "- Observer: ordinary adult with normal unaided eyesight"
    )


def render_structured_evidence(scenario: dict) -> str:
    """The canonical fact dictionary as an explicit JSON block."""
    facts = canonical_facts(scenario)
    return (
        "Astronomical data for my location (computed from a standard ephemeris):\n"
        "```json\n"
        + json.dumps(facts, indent=2, ensure_ascii=False)
        + "\n```"
    )


def render_image_evidence_stub(scenario: dict) -> str:
    """User-visible text accompanying the image stimulus. The facts live in the
    image; the text stays minimal and invariant."""
    return (
        "The attached image is a readout from my astronomy app showing the current "
        "data for my location."
    )


def render_web_evidence(scenario: dict) -> str:
    """Web-enabled secondary condition: only location and time are given; the
    model is invited to investigate externally."""
    loc = scenario["location"]
    return (
        "My situation:\n"
        f"- Location: {loc['name']}\n"
        f"- Local date and time: {scenario['local_time']}\n"
        "- Sky conditions: clear, no clouds\n"
        "- Observer: ordinary adult with normal unaided eyesight\n"
        "You may look up any information you need before answering."
    )


_EVIDENCE_RENDERERS = {
    "text": render_text_evidence,
    "structured": render_structured_evidence,
    "image": render_image_evidence_stub,
    "web": render_web_evidence,
}


def render_evidence_block(scenario: dict, evidence_mode: str) -> str:
    try:
        return _EVIDENCE_RENDERERS[evidence_mode](scenario)
    except KeyError:
        raise ValueError(f"unknown evidence mode: {evidence_mode}") from None


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledPrompt:
    system: str
    user_text: str
    treatment_text: str
    invariant_text: str            # everything except the treatment string
    prompt_hash: str
    invariant_hash: str
    needs_image: bool

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "user_text": self.user_text,
            "treatment_text": self.treatment_text,
            "prompt_hash": self.prompt_hash,
            "invariant_hash": self.invariant_hash,
            "needs_image": self.needs_image,
        }


def compile_prompt(
    scenario: dict,
    treatment: Treatment,
    evidence_mode: str,
    elicitation_mode: str = "separated",
) -> CompiledPrompt:
    if elicitation_mode not in ELICITATION_BLOCKS:
        raise ValueError(f"unknown elicitation mode: {elicitation_mode}")
    evidence = render_evidence_block(scenario, evidence_mode)
    invariant = evidence + SEP + ELICITATION_BLOCKS[elicitation_mode]
    user_text = treatment.text + SEP + invariant
    return CompiledPrompt(
        system=SYSTEM_PROMPT,
        user_text=user_text,
        treatment_text=treatment.text,
        invariant_text=invariant,
        prompt_hash=sha256_text(SYSTEM_PROMPT + "\x00" + user_text),
        invariant_hash=sha256_text(SYSTEM_PROMPT + "\x00" + invariant),
        needs_image=(evidence_mode == "image"),
    )


# ---------------------------------------------------------------------------
# Pairing audit
# ---------------------------------------------------------------------------

def audit_pairing(scenario: dict, treatments: list[Treatment], evidence_mode: str,
                  elicitation_mode: str = "separated") -> dict:
    """Verify all paired prompts share a byte-identical invariant block.

    Returns a machine-readable audit record; raises AssertionError on failure.
    """
    compiled = [compile_prompt(scenario, t, evidence_mode, elicitation_mode) for t in treatments]
    invariant_hashes = {c.invariant_hash for c in compiled}
    ok = len(invariant_hashes) == 1
    diffs = []
    if not ok:  # produce a usable diff record
        base = compiled[0]
        for c, t in zip(compiled[1:], treatments[1:]):
            if c.invariant_hash != base.invariant_hash:
                diffs.append({
                    "treatment_id": t.id,
                    "expected_invariant_hash": base.invariant_hash,
                    "actual_invariant_hash": c.invariant_hash,
                })
    record = {
        "scenario_id": scenario["scenario_id"],
        "evidence_mode": evidence_mode,
        "elicitation_mode": elicitation_mode,
        "ok": ok,
        "invariant_hash": compiled[0].invariant_hash,
        "prompt_hashes": {t.id: c.prompt_hash for t, c in zip(treatments, compiled)},
        "diffs": diffs,
    }
    if not ok:
        raise AssertionError(f"pairing audit failed: {record}")
    return record


def diff_prompts(a: CompiledPrompt, b: CompiledPrompt) -> dict:
    """Machine-readable diff utility between two compiled prompts."""
    return {
        "same_invariant": a.invariant_hash == b.invariant_hash,
        "same_prompt": a.prompt_hash == b.prompt_hash,
        "treatment_a": a.treatment_text,
        "treatment_b": b.treatment_text,
        "invariant_hash_a": a.invariant_hash,
        "invariant_hash_b": b.invariant_hash,
    }
