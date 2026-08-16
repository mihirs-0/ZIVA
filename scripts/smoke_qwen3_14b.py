#!/usr/bin/env python3
"""Run a deterministic engineering-only smoke subset of the Qwen pilot.

The subset is selected from the full shuffled 768-trial manifest. Successful
records remain valid completed trials for the subsequent resumable full run.
This script reports only engineering acceptance checks, never treatment effects.
"""

from __future__ import annotations

import json
from collections import Counter

from ziva.config import load_experiment_config, load_models_config
from ziva.freeze import experiment_fingerprint, verify_freeze
from ziva.manifest import manifest_path
from ziva.prompts import SYSTEM_PROMPT, compile_prompt
from ziva.runner import run_manifest
from ziva.treatments import get_treatment
from ziva.util import read_json, read_jsonl, sha256_json

CONFIG_PATH = "configs/pilot_qwen3_14b.yaml"


def select_smoke_rows(rows: list[dict]) -> list[dict]:
    """One row per family x elicitation x treatment template (12 total)."""
    selected: dict[tuple[str, str, str], dict] = {}
    for row in sorted(rows, key=lambda item: item["order_index"]):
        key = (row["treatment_family"], row["elicitation_mode"], row["treatment_id"])
        selected.setdefault(key, row)
    result = sorted(selected.values(), key=lambda item: item["order_index"])
    if len(result) != 12:
        raise RuntimeError(f"expected 12 balanced smoke rows, found {len(result)}")
    return result


def expected_messages_hash(row: dict, scenario: dict) -> str:
    prompt = compile_prompt(
        scenario,
        get_treatment(row["treatment_id"]),
        row["evidence_mode"],
        row["elicitation_mode"],
    )
    return sha256_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.user_text},
        ]
    )


def main() -> None:
    cfg = load_experiment_config(CONFIG_PATH)
    models = load_models_config(cfg.models_config)
    scenarios = read_json(cfg.scenario_dir / "scenarios.json")
    scenario_by_id = {item["scenario_id"]: item for item in scenarios}
    rows = read_jsonl(manifest_path(cfg))

    freeze_check = verify_freeze(cfg, cfg.scenario_dir / "scenarios.json", manifest_path(cfg), cfg.stimuli_dir, models)
    if not freeze_check["ok"]:
        raise RuntimeError(f"freeze verification failed: {freeze_check['violations']}")

    smoke_rows = select_smoke_rows(rows)
    fingerprint = experiment_fingerprint(cfg, models)
    run_manifest(
        cfg,
        smoke_rows,
        scenarios,
        models,
        SYSTEM_PROMPT,
        fingerprint=fingerprint,
        ran_dirty=False,
    )

    records = [read_json(cfg.raw_dir / f"{row['trial_id']}.json") for row in smoke_rows]
    statuses = Counter(record["final_parse"]["status"] for record in records)
    texts = [record["steps"][-1]["result"]["text"] for record in records]
    think_count = sum("<think" in text.lower() or "</think" in text.lower() for text in texts)
    settings_ok = all(
        record["steps"][-1]["result"]["raw"]["generation_settings"]["chat_template_kwargs"]
        == {"enable_thinking": False}
        for record in records
    )
    prompt_hash_ok = all(
        record["steps"][-1]["result"]["raw"]["request_messages_sha256"]
        == expected_messages_hash(record, scenario_by_id[record["scenario_id"]])
        for record in records
    )
    system_prompt_ok = all(record["system_prompt_used"] == SYSTEM_PROMPT for record in records)
    parse_rate = statuses.get("ok", 0) / len(records)

    report = {
        "purpose": "engineering smoke only; no treatment effects inspected",
        "n_trials": len(records),
        "parse_status_counts": dict(statuses),
        "parse_success_rate": parse_rate,
        "think_tag_count": think_count,
        "enable_thinking_false_recorded": settings_ok,
        "exact_outbound_messages_hash_match": prompt_hash_ok,
        "system_prompt_exact_match": system_prompt_ok,
        "acceptance": {
            "parse_rate_at_least_0_90": parse_rate >= 0.90,
            "no_think_tags": think_count == 0,
            "settings_ok": settings_ok,
            "prompt_fidelity_ok": prompt_hash_ok and system_prompt_ok,
        },
    }
    report["passed"] = all(report["acceptance"].values())
    out_path = cfg.manifest_dir / "smoke_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
