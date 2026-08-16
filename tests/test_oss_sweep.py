from __future__ import annotations

import json
from pathlib import Path

import yaml

from ziva.chat_percentage import audit_percentage_pairing, build_percentage_trials

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/oss_chat_v1_protocol.yaml"
QUEUE = ROOT / "configs/oss_model_queue.yaml"
CONCISE = ROOT / "configs/pilot_qwen3_14b_chat_percentage_concise.yaml"
FOLLOWUP = "Okay — if you had to give me just a rough percentage, what would you say? No explanation needed."


def _yaml(path: Path):
    return yaml.safe_load(path.read_text())


def test_fixed_model_queue_order_and_precision_policy():
    queue = _yaml(QUEUE)
    assert [row["key"] for row in queue["models"]] == [
        "qwen3_14b",
        "qwen3_4b",
        "qwen3_8b",
        "ministral3_8b",
        "gemma4_12b",
        "ministral3_14b",
    ]
    assert queue["runtime"]["dtype"] == "bfloat16"
    assert queue["runtime"]["quantization"] == "none"
    assert queue["runtime"]["tensor_parallel_size"] == 2
    assert queue["models"][0]["required_revision"] == "40c069824f4251a91eefaf281ebe4c544efd3e18"
    for row in queue["models"][:3]:
        assert row["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_v1_changes_only_turn_1_budget_from_concise_protocol():
    protocol = _yaml(PROTOCOL)
    concise = _yaml(CONCISE)
    assert protocol["conversation"] == concise["conversation"]
    assert protocol["treatments"] == concise["treatments"]
    assert protocol["run"]["shuffle_seed"] == concise["run"]["shuffle_seed"]
    assert protocol["sampling"]["turn_1_max_tokens"] == 2048
    assert protocol["sampling"]["turn_2_max_tokens"] == concise["sampling"]["turn_2_max_tokens"] == 16
    for key in ("repeats", "temperature", "top_p", "top_k", "min_p", "presence_penalty"):
        assert protocol["sampling"][key] == concise["sampling"][key]


def test_chat_v1_manifest_is_512_and_low_cue():
    config = _yaml(PROTOCOL)
    scenarios = json.loads((ROOT / config["scenario_file"]).read_text())
    trials = build_percentage_trials(config, scenarios)
    audit = audit_percentage_pairing(trials)
    assert len(trials) == 512
    assert audit["failures"] == 0
    assert {row.turn_2_user for row in trials} == {FOLLOWUP}
    assert {row.system for row in trials} == {"You are a helpful assistant."}
    prohibited = ("json", "benchmark", "evaluation", "scoring", "calibration", "logging")
    assert all(
        not any(cue in (row.system + "\n" + row.turn_1_user + "\n" + row.turn_2_user).lower() for cue in prohibited)
        for row in trials
    )


def test_structured_and_chat_trial_count_formulas_remain_fixed():
    scenarios = json.loads((ROOT / "data/scenarios/pilot_qwen3_14b/scenarios.json").read_text())
    assert len(scenarios) == 32
    assert 32 * 3 * 2 * 2 * 2 == 768
    assert 32 * 4 * 2 * 2 == 512
