from __future__ import annotations

import json
from pathlib import Path

from ziva.chat_eval import (
    ContinuationPair,
    assert_low_cue_prompt,
    audit_chat_pairing,
    build_chat_prompts,
    load_chat_config,
    render_chat_evidence,
)
from ziva.prompts import canonical_facts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pilot_qwen3_14b_chat.yaml"
SCENARIOS = ROOT / "data/scenarios/pilot_qwen3_14b/scenarios.json"


def _inputs():
    config = load_chat_config(CONFIG)
    scenarios = json.loads(SCENARIOS.read_text())
    return config, scenarios


def test_chat_manifest_has_expected_frozen_design():
    config, scenarios = _inputs()
    prompts = build_chat_prompts(config, scenarios)
    assert len(scenarios) == 32
    assert len(prompts) == 32 * 4 * 2 == 256
    assert {row.treatment_family for row in prompts} == {
        "neutral",
        "excited_positive",
        "skeptical_negative",
        "negative_preference",
    }
    assert len({row.prompt_hash for row in prompts}) == 256


def test_chat_pairing_changes_only_frozen_treatment_text():
    config, scenarios = _inputs()
    prompts = build_chat_prompts(config, scenarios)
    audit = audit_chat_pairing(prompts)
    assert audit["n_scenarios"] == 32
    assert audit["failures"] == 0
    for scenario in scenarios:
        group = [row for row in prompts if row.scenario_id == scenario["scenario_id"]]
        assert len({row.invariant_text for row in group}) == 1
        assert len({row.system for row in group}) == 1


def test_model_facing_chat_is_low_cue_and_never_leaks_candidates():
    config, scenarios = _inputs()
    pairs = [ContinuationPair(**row) for row in config["scoring"]["continuation_pairs"]]
    for prompt in build_chat_prompts(config, scenarios):
        assert_low_cue_prompt(prompt, pairs)
        assert prompt.system == "You are a helpful assistant."
        assert prompt.user_text.endswith("Do you think I'd actually be able to spot it?")


def test_natural_language_renderer_preserves_every_canonical_fact():
    _, scenarios = _inputs()
    for scenario in scenarios:
        rendered = render_chat_evidence(scenario)
        facts = canonical_facts(scenario)
        for key in (
            "location",
            "local_time",
            "utc_time",
            "latitude_deg",
            "longitude_deg",
            "sun_altitude_deg",
            "sun_azimuth_deg",
            "moon_altitude_deg",
            "moon_azimuth_deg",
            "moon_illumination_percent",
            "sun_moon_angular_separation_deg",
            "moon_distance_km",
            "sky_brightness",
        ):
            value = facts[key]
            expected = format(value, ".15g") if isinstance(value, float) else str(value)
            assert expected in rendered
        assert "clear" in rendered
        assert "normal unaided eyesight" in rendered


def test_original_structured_prompt_module_is_not_redefined():
    source = (ROOT / "src/ziva/chat_eval.py").read_text()
    assert "def compile_prompt(" not in source
    assert "def parse_response(" not in source
