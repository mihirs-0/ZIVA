from __future__ import annotations

import json
from pathlib import Path

import pytest

from ziva.chat_percentage import audit_percentage_pairing, build_percentage_trials, load_percentage_config
from ziva.chat_percentage_concise import choose_token_budget, parse_concise_percentage

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pilot_qwen3_14b_chat_percentage_concise.yaml"
SCENARIOS = ROOT / "data/scenarios/pilot_qwen3_14b/scenarios.json"
FOLLOWUP = "Okay — if you had to give me just a rough percentage, what would you say? No explanation needed."


@pytest.mark.parametrize(
    ("text", "status", "value", "original_range"),
    [
        ("30%", "ok_single", 30.0, None),
        ("30 percent", "ok_single", 30.0, None),
        ("Around 30%.", "ok_single", 30.0, None),
        ("20–30%", "ok_range", 25.0, [20.0, 30.0]),
        ("20 to 30 percent", "ok_range", 25.0, [20.0, 30.0]),
        ("30%, roughly 30 percent.", "ok_single", 30.0, None),
    ],
)
def test_concise_parser_accepts_only_explicit_scalars_and_ranges(
    text: str, status: str, value: float, original_range: list[float] | None
):
    parsed = parse_concise_percentage(text)
    assert parsed.status == status
    assert parsed.value == value
    assert parsed.original_range == original_range


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("", "empty"),
        ("Probably quite low.", "no_percentage"),
        ("Maybe 0.3.", "no_percentage"),
        ("30% or 70%.", "ambiguous"),
        ("20–30%, perhaps 50%.", "ambiguous"),
        ("120%.", "no_percentage"),
    ],
)
def test_concise_parser_rejects_missing_or_incompatible_values(text: str, status: str):
    parsed = parse_concise_percentage(text)
    assert parsed.status == status
    assert parsed.value is None


def test_token_budget_chooser_uses_smallest_candidate_that_passes_every_gate():
    summaries = [
        {
            "max_tokens": 16,
            "parse_success_rate": 1.0,
            "truncation_rate": 0.25,
            "anchor_order_sensible": True,
            "no_think_content": True,
        },
        {
            "max_tokens": 32,
            "parse_success_rate": 1.0,
            "truncation_rate": 0.0,
            "anchor_order_sensible": True,
            "no_think_content": True,
        },
        {
            "max_tokens": 64,
            "parse_success_rate": 1.0,
            "truncation_rate": 0.0,
            "anchor_order_sensible": True,
            "no_think_content": True,
        },
    ]
    assert choose_token_budget(summaries, 0.9, 0.1) == 32


def test_manifest_remains_512_trials_and_changes_only_the_followup():
    config = load_percentage_config(CONFIG)
    scenarios = json.loads(SCENARIOS.read_text())
    trials = build_percentage_trials(config, scenarios)
    audit = audit_percentage_pairing(trials)
    assert len(trials) == 512
    assert audit["failures"] == 0
    assert {row.turn_2_user for row in trials} == {FOLLOWUP}
    assert all(row.system == "You are a helpful assistant." for row in trials)


def test_concise_experiment_is_separate_and_prior_artifacts_exist():
    config = load_percentage_config(CONFIG)
    assert config["experiment_name"] != config["source_experiment"]
    assert (ROOT / config["previous_chat_summary"]).exists()
    assert (ROOT / config["previous_chat_scenario_effects"]).exists()
    assert (ROOT / config["structured_scenario_effects"]).exists()
