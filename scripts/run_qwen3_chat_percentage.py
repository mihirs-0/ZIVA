#!/usr/bin/env python3
"""Build, freeze, run, and analyze the two-turn Qwen chat-percentage experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ziva.chat_eval import render_chat_evidence, sha256_file
from ziva.chat_percentage import (
    ChatPercentageTrial,
    audit_percentage_pairing,
    build_percentage_trials,
    execute_percentage_trial,
    load_percentage_config,
    parse_percentage,
)
from ziva.chat_percentage_analysis import analyze_percentage_records, render_percentage_report
from ziva.treatments import select_treatments
from ziva.util import sha256_json, sha256_text, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pilot_qwen3_14b_chat_percentage.yaml"


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    name = config["experiment_name"]
    return {
        "manifest_dir": ROOT / "data/manifests" / name,
        "raw_dir": ROOT / "data/raw" / name,
        "results_dir": ROOT / "results" / name,
        "manifest": ROOT / "data/manifests" / name / "trials.jsonl",
        "pairing": ROOT / "data/manifests" / name / "pairing_audit.json",
        "treatments": ROOT / "data/manifests" / name / "treatments.json",
        "freeze": ROOT / "data/manifests" / name / "freeze.json",
        "smoke": ROOT / "data/manifests" / name / "smoke.json",
        "run_summary": ROOT / "data/raw" / name / "_run_summary.json",
        "summary": ROOT / "results" / name / "summary.json",
        "scenario_effects": ROOT / "results" / name / "scenario_level_effects.csv",
        "direct_comparison": ROOT / "results" / name / "chat_minus_structured.csv",
        "trials_csv": ROOT / "results" / name / "trials.csv",
        "report": ROOT / "reports" / f"{name}_report.md",
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _read_trials(path: Path) -> list[ChatPercentageTrial]:
    return [ChatPercentageTrial(**json.loads(line)) for line in path.read_text().splitlines() if line]


def build(config: dict[str, Any], config_path: Path) -> None:
    paths = _paths(config)
    for key in ("manifest_dir", "raw_dir", "results_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    scenarios = json.loads((ROOT / config["scenario_file"]).read_text())
    trials = build_percentage_trials(config, scenarios)
    audit = audit_percentage_pairing(trials)
    treatments = select_treatments(
        config["treatments"]["families"], config["treatments"]["variants_per_family"]
    )
    write_jsonl(paths["manifest"], [row.to_dict() for row in trials])
    write_json(paths["pairing"], audit)
    write_json(
        paths["treatments"],
        {
            "source": "src/ziva/treatments.py",
            "source_sha256": sha256_file(ROOT / "src/ziva/treatments.py"),
            "treatments": [row.to_dict() for row in treatments],
        },
    )
    print(
        json.dumps(
            {
                "experiment": config["experiment_name"],
                "scenarios": len(scenarios),
                "trials": len(trials),
                "expected_formula": "32 scenarios * 4 families * 2 variants * 2 repeats",
                "pairing_failures": audit["failures"],
                "config_sha256": sha256_file(config_path),
            },
            indent=2,
        )
    )


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = _paths(config)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files are dirty; commit implementation before freezing")
    source_paths = [
        config_path,
        ROOT / "src/ziva/chat_percentage.py",
        ROOT / "src/ziva/chat_percentage_analysis.py",
        ROOT / "scripts/run_qwen3_chat_percentage.py",
        ROOT / "src/ziva/chat_eval.py",
        ROOT / "src/ziva/treatments.py",
        ROOT / config["scenario_file"],
        ROOT / config["structured_scenario_effects"],
        paths["manifest"],
        paths["pairing"],
        paths["treatments"],
    ]
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    record = {
        "experiment_name": config["experiment_name"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "model": config["model"],
        "sampling": config["sampling"],
        "system_prompt": config["conversation"]["system_prompt"],
        "turn_1_question": config["conversation"]["turn_1_question"],
        "turn_2_followup": config["conversation"]["turn_2_followup"],
        "percentage_parser_source_sha256": sha256_text(inspect.getsource(parse_percentage)),
        "evidence_renderer_source_sha256": sha256_text(inspect.getsource(render_chat_evidence)),
        "scenario_ids": sorted({row.scenario_id for row in _read_trials(paths["manifest"])}),
        "analysis_definitions": {
            "primary": "Turn-2 parsed percentage; scenario family means average repeats and paraphrases",
            "range_rule": "midpoint of one clear explicit percentage range",
            "malformed_rule": "no percentage or multiple incompatible percentages",
            "binary_threshold": config["analysis"]["binary_threshold"],
            "inferential_unit": "physical scenario",
        },
        "source_hashes": hashes,
    }
    record["experiment_fingerprint"] = sha256_json(record)
    write_json(paths["freeze"], record)
    print(json.dumps(record, indent=2))


def _verify_freeze(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    frozen = json.loads(_paths(config)["freeze"].read_text())
    for relative, expected in frozen["source_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    if sha256_file(config_path) != frozen["source_hashes"][str(config_path.relative_to(ROOT))]:
        raise RuntimeError("configuration changed after freeze")
    return frozen


def smoke(config: dict[str, Any], config_path: Path) -> None:
    frozen = _verify_freeze(config, config_path)
    paths = _paths(config)
    anchor = config["anchors"]
    selected = [
        row
        for row in _read_trials(paths["manifest"])
        if row.treatment_family == anchor["treatment_family"]
        and row.scenario_id in {anchor["below_horizon"], anchor["bright_night"]}
    ]
    records = [execute_percentage_trial(row, config, frozen["experiment_fingerprint"]) for row in selected]
    below = [
        row["percentage_parse"]["value"]
        for row in records
        if row["scenario_id"] == anchor["below_horizon"] and row["percentage_parse"]["value"] is not None
    ]
    bright = [
        row["percentage_parse"]["value"]
        for row in records
        if row["scenario_id"] == anchor["bright_night"] and row["percentage_parse"]["value"] is not None
    ]
    no_think = all(
        "<think>" not in (row["turn_1"]["text"] + row["turn_2"]["text"]).lower() for row in records
    )
    followups = {row.turn_2_user for row in selected}
    parse_successes = sum(row["percentage_parse"]["value"] is not None for row in records)
    parse_success_rate = parse_successes / len(records)
    record = {
        "purpose": "neutral engineering smoke only; no treatment effects inspected",
        "n_trials": len(records),
        "parse_successes": parse_successes,
        "parse_success_rate": parse_success_rate,
        "minimum_parse_rate": anchor["minimum_parse_rate"],
        "below_horizon_percentages": below,
        "bright_night_percentages": bright,
        "anchor_order_sensible": bool(below and bright and max(below) < 50 < min(bright)),
        "no_think_content": no_think,
        "turn_2_followup_byte_identical": len(followups) == 1,
        "enable_thinking": config["model"]["enable_thinking"],
        "records": records,
    }
    record["passed"] = all(
        [
            parse_success_rate >= anchor["minimum_parse_rate"],
            record["anchor_order_sensible"],
            no_think,
            len(followups) == 1,
        ]
    )
    write_json(paths["smoke"], record)
    print(json.dumps({key: value for key, value in record.items() if key != "records"}, indent=2))
    if not record["passed"]:
        raise RuntimeError("neutral smoke failed; full treatment run is blocked")


def run(config: dict[str, Any], config_path: Path) -> None:
    frozen = _verify_freeze(config, config_path)
    paths = _paths(config)
    smoke_record = json.loads(paths["smoke"].read_text())
    if not smoke_record.get("passed"):
        raise RuntimeError("passing smoke required")
    trials = _read_trials(paths["manifest"])
    existing = {path.stem for path in paths["raw_dir"].glob("tcp_*.json")}
    pending = [row for row in trials if row.trial_id not in existing]
    started = datetime.now(UTC).isoformat()
    completed = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["run"]["concurrency"]) as pool:
        futures = {
            pool.submit(execute_percentage_trial, row, config, frozen["experiment_fingerprint"]): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            write_json(paths["raw_dir"] / f"{record['trial_id']}.json", record)
            completed += 1
            errors += int(record["status"] != "ok")
            if completed % 25 == 0 or completed == len(pending):
                print(f"completed={completed}/{len(pending)} errors={errors}", flush=True)
    summary = {
        "experiment_name": config["experiment_name"],
        "experiment_fingerprint": frozen["experiment_fingerprint"],
        "total_trials": len(trials),
        "already_complete": len(existing),
        "executed": completed,
        "execution_errors": errors,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(paths["run_summary"], summary)
    print(json.dumps(summary, indent=2))


def analyze(config: dict[str, Any], config_path: Path) -> None:
    _verify_freeze(config, config_path)
    paths = _paths(config)
    summary, scenario, direct = analyze_percentage_records(paths["raw_dir"], config)
    write_json(paths["summary"], summary)
    scenario.to_csv(paths["scenario_effects"], index=False)
    direct.to_csv(paths["direct_comparison"], index=False)
    trial_rows = []
    for path in sorted(paths["raw_dir"].glob("tcp_*.json")):
        record = json.loads(path.read_text())
        trial_rows.append(
            {
                "trial_id": record["trial_id"],
                "scenario_id": record["scenario_id"],
                "treatment_id": record["treatment_id"],
                "treatment_family": record["treatment_family"],
                "repeat_index": record["repeat_index"],
                "visible_probability_chat": record["percentage_parse"]["value"],
                "parse_status": record["percentage_parse"]["status"],
            }
        )
    pd.DataFrame(trial_rows).to_csv(paths["trials_csv"], index=False)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(render_percentage_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "freeze", "smoke", "run", "analyze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_percentage_config(config_path)
    commands = {
        "build": lambda: build(config, config_path),
        "freeze": lambda: freeze(config, config_path),
        "smoke": lambda: smoke(config, config_path),
        "run": lambda: run(config, config_path),
        "analyze": lambda: analyze(config, config_path),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
