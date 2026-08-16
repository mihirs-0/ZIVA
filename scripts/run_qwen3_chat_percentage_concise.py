#!/usr/bin/env python3
"""Build, tune neutrally, freeze, run, and analyze the concise percentage rerun."""

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
    LocalVLLMChatClient,
    audit_percentage_pairing,
    build_percentage_trials,
    load_percentage_config,
)
from ziva.chat_percentage_concise import (
    choose_token_budget,
    execute_concise_trial,
    parse_concise_percentage,
)
from ziva.chat_percentage_concise_analysis import analyze_concise_records, render_concise_report
from ziva.treatments import select_treatments
from ziva.util import sha256_json, sha256_text, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pilot_qwen3_14b_chat_percentage_concise.yaml"


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    name = config["experiment_name"]
    return {
        "manifest_dir": ROOT / "data/manifests" / name,
        "raw_dir": ROOT / "data/raw" / name,
        "results_dir": ROOT / "results" / name,
        "manifest": ROOT / "data/manifests" / name / "trials.jsonl",
        "pairing": ROOT / "data/manifests" / name / "pairing_audit.json",
        "treatments": ROOT / "data/manifests" / name / "treatments.json",
        "token_audit": ROOT / "data/manifests" / name / "token_budget_audit.json",
        "freeze": ROOT / "data/manifests" / name / "freeze.json",
        "smoke": ROOT / "data/manifests" / name / "smoke.json",
        "run_summary": ROOT / "data/raw" / name / "_run_summary.json",
        "summary": ROOT / "results" / name / "summary.json",
        "scenario_effects": ROOT / "results" / name / "scenario_level_effects.csv",
        "structured_comparison": ROOT / "results" / name / "chat_minus_structured.csv",
        "previous_comparison": ROOT / "results" / name / "concise_minus_previous_chat.csv",
        "trials_csv": ROOT / "results" / name / "trials.csv",
        "report": ROOT / "reports" / f"{name}_report.md",
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _read_trials(path: Path) -> list[ChatPercentageTrial]:
    return [ChatPercentageTrial(**json.loads(line)) for line in path.read_text().splitlines() if line]


def _neutral_anchors(config: dict[str, Any], manifest: Path) -> list[ChatPercentageTrial]:
    anchor = config["anchors"]
    selected = [
        row
        for row in _read_trials(manifest)
        if row.treatment_family == anchor["treatment_family"]
        and row.scenario_id in {anchor["below_horizon"], anchor["bright_night"]}
    ]
    if len(selected) != 8:
        raise RuntimeError(f"expected 8 neutral anchor trials, found {len(selected)}")
    return selected


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


def _budget_trial(trial: ChatPercentageTrial, config: dict[str, Any]) -> dict[str, Any]:
    client = LocalVLLMChatClient(config)
    messages_1 = [
        {"role": "system", "content": trial.system},
        {"role": "user", "content": trial.turn_1_user},
    ]
    turn_1 = client.complete(messages_1, trial.turn_1_seed, config["sampling"]["turn_1_max_tokens"])
    messages_2 = messages_1 + [
        {"role": "assistant", "content": turn_1["text"]},
        {"role": "user", "content": trial.turn_2_user},
    ]
    budgets = []
    for max_tokens in sorted(config["token_budget_development"]["candidates"]):
        turn_2 = client.complete(messages_2, trial.turn_2_seed, max_tokens)
        budgets.append(
            {
                "max_tokens": max_tokens,
                "turn_2": turn_2,
                "percentage_parse": parse_concise_percentage(turn_2["text"]).to_dict(),
            }
        )
    return {**trial.to_dict(), "turn_1": turn_1, "budgets": budgets}


def token_audit(config: dict[str, Any]) -> None:
    paths = _paths(config)
    selected = _neutral_anchors(config, paths["manifest"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["run"]["concurrency"]) as pool:
        records = list(pool.map(lambda row: _budget_trial(row, config), selected))
    anchor = config["anchors"]
    summaries = []
    for max_tokens in sorted(config["token_budget_development"]["candidates"]):
        rows = [
            {**record, "budget": next(row for row in record["budgets"] if row["max_tokens"] == max_tokens)}
            for record in records
        ]
        values = [row["budget"]["percentage_parse"]["value"] for row in rows]
        below = [
            value
            for row, value in zip(rows, values, strict=True)
            if row["scenario_id"] == anchor["below_horizon"] and value is not None
        ]
        bright = [
            value
            for row, value in zip(rows, values, strict=True)
            if row["scenario_id"] == anchor["bright_night"] and value is not None
        ]
        summaries.append(
            {
                "max_tokens": max_tokens,
                "n_trials": len(rows),
                "parse_successes": sum(value is not None for value in values),
                "parse_success_rate": sum(value is not None for value in values) / len(rows),
                "truncations": sum(row["budget"]["turn_2"]["finish_reason"] == "length" for row in rows),
                "truncation_rate": sum(row["budget"]["turn_2"]["finish_reason"] == "length" for row in rows)
                / len(rows),
                "below_horizon_percentages": below,
                "bright_night_percentages": bright,
                "anchor_order_sensible": bool(below and bright and max(below) < 50 < min(bright)),
                "no_think_content": all(
                    "<think>" not in (row["turn_1"]["text"] + row["budget"]["turn_2"]["text"]).lower()
                    for row in rows
                ),
            }
        )
    development = config["token_budget_development"]
    selected_budget = choose_token_budget(
        summaries, development["minimum_parse_rate"], development["maximum_truncation_rate"]
    )
    audit = {
        "purpose": "neutral-only Turn-2 token-budget engineering; no treatment effects inspected",
        "selection_rule": (
            "smallest candidate with parse rate >= minimum, truncation rate <= maximum, "
            "sensible anchors, and no thinking content"
        ),
        "minimum_parse_rate": development["minimum_parse_rate"],
        "maximum_truncation_rate": development["maximum_truncation_rate"],
        "selected_budget": selected_budget,
        "summaries": summaries,
        "records": records,
    }
    write_json(paths["token_audit"], audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, indent=2))
    if selected_budget is None:
        raise RuntimeError("no candidate token budget passed the neutral-only engineering gates")


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = _paths(config)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files are dirty; commit the final token budget before freezing")
    token_record = json.loads(paths["token_audit"].read_text())
    configured_budget = config["sampling"]["turn_2_max_tokens"]
    if configured_budget != token_record["selected_budget"]:
        raise RuntimeError(
            f"configured Turn-2 cap {configured_budget!r} does not match neutral selection "
            f"{token_record['selected_budget']!r}"
        )
    source_paths = [
        config_path,
        ROOT / "src/ziva/chat_percentage_concise.py",
        ROOT / "src/ziva/chat_percentage_concise_analysis.py",
        ROOT / "scripts/run_qwen3_chat_percentage_concise.py",
        ROOT / "src/ziva/chat_percentage.py",
        ROOT / "src/ziva/chat_percentage_analysis.py",
        ROOT / "src/ziva/chat_eval.py",
        ROOT / "src/ziva/treatments.py",
        ROOT / config["scenario_file"],
        ROOT / config["structured_scenario_effects"],
        ROOT / config["previous_chat_scenario_effects"],
        ROOT / config["previous_chat_summary"],
        paths["manifest"],
        paths["pairing"],
        paths["treatments"],
        paths["token_audit"],
    ]
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    record = {
        "experiment_name": config["experiment_name"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "model": config["model"],
        "sampling": config["sampling"],
        "token_budget_selection": {
            "selected_budget": token_record["selected_budget"],
            "selection_rule": token_record["selection_rule"],
            "neutral_summaries": token_record["summaries"],
        },
        "system_prompt": config["conversation"]["system_prompt"],
        "turn_1_question": config["conversation"]["turn_1_question"],
        "turn_2_followup": config["conversation"]["turn_2_followup"],
        "percentage_parser_source_sha256": sha256_text(inspect.getsource(parse_concise_percentage)),
        "evidence_renderer_source_sha256": sha256_text(inspect.getsource(render_chat_evidence)),
        "scenario_ids": sorted({row.scenario_id for row in _read_trials(paths["manifest"])}),
        "analysis_definitions": {
            "primary": "Turn-2 parsed percentage; scenario family means average repeats and paraphrases",
            "range_rule": "midpoint of one explicit hyphen/dash/to percentage range",
            "malformed_rule": "no explicit percentage or multiple incompatible expressions",
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
    selected = _neutral_anchors(config, paths["manifest"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["run"]["concurrency"]) as pool:
        records = list(
            pool.map(
                lambda row: execute_concise_trial(row, config, frozen["experiment_fingerprint"]),
                selected,
            )
        )
    anchor = config["anchors"]
    valid = [row["percentage_parse"]["value"] for row in records]
    below = [
        value
        for row, value in zip(records, valid, strict=True)
        if row["scenario_id"] == anchor["below_horizon"] and value is not None
    ]
    bright = [
        value
        for row, value in zip(records, valid, strict=True)
        if row["scenario_id"] == anchor["bright_night"] and value is not None
    ]
    parse_rate = sum(value is not None for value in valid) / len(records)
    truncation_rate = sum(row["turn_2"]["finish_reason"] == "length" for row in records) / len(records)
    no_think = all(
        "<think>" not in (row["turn_1"]["text"] + row["turn_2"]["text"]).lower() for row in records
    )
    development = config["token_budget_development"]
    record = {
        "purpose": "neutral engineering smoke only; no treatment effects inspected",
        "n_trials": len(records),
        "parse_success_rate": parse_rate,
        "turn_2_truncation_rate": truncation_rate,
        "below_horizon_percentages": below,
        "bright_night_percentages": bright,
        "anchor_order_sensible": bool(below and bright and max(below) < 50 < min(bright)),
        "no_think_content": no_think,
        "turn_2_followup_byte_identical": len({row.turn_2_user for row in selected}) == 1,
        "enable_thinking": config["model"]["enable_thinking"],
        "turn_2_max_tokens": config["sampling"]["turn_2_max_tokens"],
        "records": records,
    }
    record["passed"] = all(
        [
            parse_rate >= development["minimum_parse_rate"],
            truncation_rate <= development["maximum_truncation_rate"],
            record["anchor_order_sensible"],
            no_think,
            record["turn_2_followup_byte_identical"],
        ]
    )
    write_json(paths["smoke"], record)
    print(json.dumps({key: value for key, value in record.items() if key != "records"}, indent=2))
    if not record["passed"]:
        raise RuntimeError("neutral smoke failed; full treatment run is blocked")


def run(config: dict[str, Any], config_path: Path) -> None:
    frozen = _verify_freeze(config, config_path)
    paths = _paths(config)
    if not json.loads(paths["smoke"].read_text()).get("passed"):
        raise RuntimeError("passing smoke required")
    trials = _read_trials(paths["manifest"])
    existing = {path.stem for path in paths["raw_dir"].glob("tcp_*.json")}
    pending = [row for row in trials if row.trial_id not in existing]
    started = datetime.now(UTC).isoformat()
    completed = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["run"]["concurrency"]) as pool:
        futures = {
            pool.submit(execute_concise_trial, row, config, frozen["experiment_fingerprint"]): row
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
    summary, scenario, structured_direct, previous_direct = analyze_concise_records(paths["raw_dir"], config)
    write_json(paths["summary"], summary)
    scenario.to_csv(paths["scenario_effects"], index=False)
    structured_direct.to_csv(paths["structured_comparison"], index=False)
    previous_direct.to_csv(paths["previous_comparison"], index=False)
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
                "turn_1_finish_reason": record["turn_1"]["finish_reason"],
                "turn_2_finish_reason": record["turn_2"]["finish_reason"],
            }
        )
    pd.DataFrame(trial_rows).to_csv(paths["trials_csv"], index=False)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(render_concise_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "token-audit", "freeze", "smoke", "run", "analyze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_percentage_config(config_path)
    commands = {
        "build": lambda: build(config, config_path),
        "token-audit": lambda: token_audit(config),
        "freeze": lambda: freeze(config, config_path),
        "smoke": lambda: smoke(config, config_path),
        "run": lambda: run(config, config_path),
        "analyze": lambda: analyze(config, config_path),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
