#!/usr/bin/env python3
"""Build, freeze, smoke, run, and analyze the globally frozen ZIVA-Chat-v1."""

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
import yaml

from ziva.chat_eval import render_chat_evidence, sha256_file
from ziva.chat_percentage import (
    ChatPercentageTrial,
    audit_percentage_pairing,
    build_percentage_trials,
)
from ziva.chat_percentage_concise import execute_concise_trial, parse_concise_percentage
from ziva.oss_chat_v1_analysis import analyze_chat_v1_records, render_chat_v1_report
from ziva.treatments import select_treatments
from ziva.util import sha256_json, sha256_text, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/oss_chat_v1_protocol.yaml"
QUEUE_PATH = ROOT / "configs/oss_model_queue.yaml"
PROTOCOL_DIR = ROOT / "data/manifests/oss_chat_v1_protocol"
PROTOCOL_FREEZE = PROTOCOL_DIR / "freeze.json"
REGISTRY_DIR = ROOT / "data/manifests/oss_model_registry"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_spec(key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = _load_yaml(QUEUE_PATH)
    matches = [row for row in queue["models"] if row["key"] == key]
    if len(matches) != 1:
        raise RuntimeError(f"model key {key!r} is absent or duplicated")
    return matches[0], queue["runtime"]


def _registry(key: str) -> dict[str, Any]:
    path = REGISTRY_DIR / f"{key}.json"
    if not path.exists():
        raise RuntimeError(f"missing resolved model registry: {path}")
    row = json.loads(path.read_text())
    if row.get("status") != "RESOLVED":
        raise RuntimeError(f"model {key} is not resolved: {row.get('status')}")
    return row


def _structured_name(key: str) -> str:
    return "pilot_qwen3_14b" if key == "qwen3_14b" else f"oss_{key}_structured"


def _config(key: str, turn_1_override: int | None = None) -> dict[str, Any]:
    protocol = _load_yaml(PROTOCOL_PATH)
    spec, runtime = _model_spec(key)
    registry = _registry(key)
    sampling = dict(protocol["sampling"])
    if turn_1_override is not None:
        sampling["turn_1_max_tokens"] = turn_1_override
    structured_name = _structured_name(key)
    config = {
        **protocol,
        "experiment_name": f"oss_{key}_chat_v1",
        "sampling": sampling,
        "structured_scenario_effects": f"results/{structured_name}/scenario_level_diffs.csv",
        "structured_paired_table": f"results/{structured_name}/paired_table.csv",
        "model": {
            "id": key,
            "checkpoint": spec["checkpoint"],
            "revision": registry["revision"],
            "tokenizer_revision": registry["tokenizer_revision"],
            "parameter_scale": spec["parameter_scale"],
            "model_family": spec["model_family"],
            "dtype": runtime["dtype"],
            "quantization": runtime["quantization"],
            "framework": runtime["framework"],
            "framework_version": runtime["framework_version"],
            "endpoint": runtime["endpoint"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "gpu_configuration": runtime["gpu_configuration"],
            "chat_template_kwargs": spec.get("chat_template_kwargs", {}),
            "enable_thinking": spec.get("chat_template_kwargs", {}).get("enable_thinking", False),
        },
    }
    if key == "qwen3_14b":
        config["previous_chat_scenario_effects"] = (
            "results/pilot_qwen3_14b_chat_percentage_concise/scenario_level_effects.csv"
        )
    return config


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    name = config["experiment_name"]
    manifest_dir = ROOT / "data/manifests" / name
    results_dir = ROOT / "results" / name
    return {
        "manifest_dir": manifest_dir,
        "raw_dir": ROOT / "data/raw" / name,
        "results_dir": results_dir,
        "manifest": manifest_dir / "trials.jsonl",
        "pairing": manifest_dir / "pairing_audit.json",
        "treatments": manifest_dir / "treatments.json",
        "resolved_config": manifest_dir / "resolved_config.json",
        "freeze": manifest_dir / "freeze.json",
        "smoke": manifest_dir / "neutral_smoke.json",
        "gpu_log": manifest_dir / "full_gpu_dmon.txt",
        "run_summary": ROOT / "data/raw" / name / "_run_summary.json",
        "summary": results_dir / "summary.json",
        "scenario_effects": results_dir / "scenario_level_effects.csv",
        "trials_csv": results_dir / "trials.csv",
        "runtime_audit": results_dir / "runtime_audit.json",
        "report": ROOT / "reports" / f"{name}_report.md",
    }


def _read_trials(path: Path) -> list[ChatPercentageTrial]:
    return [ChatPercentageTrial(**json.loads(line)) for line in path.read_text().splitlines() if line]


def _neutral_smoke_trials(config: dict[str, Any], manifest: Path) -> list[ChatPercentageTrial]:
    scenarios = json.loads((ROOT / config["scenario_file"]).read_text())
    by_class: dict[str, str] = {}
    for scenario in scenarios:
        label = scenario["classification"]["difficulty_class"]
        by_class.setdefault(label, scenario["scenario_id"])
    ids = {
        config["anchors"]["below_horizon"],
        config["anchors"]["bright_night"],
        by_class["moderate"],
        by_class["ambiguous"],
    }
    selected = [
        row
        for row in _read_trials(manifest)
        if row.treatment_family == "neutral" and row.scenario_id in ids
    ]
    if len(selected) != 16:
        raise RuntimeError(f"expected 16 neutral smoke trials, found {len(selected)}")
    return selected


def build(config: dict[str, Any]) -> None:
    paths = _paths(config)
    for key in ("manifest_dir", "raw_dir", "results_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    scenarios = json.loads((ROOT / config["scenario_file"]).read_text())
    trials = build_percentage_trials(config, scenarios)
    pairing = audit_percentage_pairing(trials)
    treatments = select_treatments(
        config["treatments"]["families"], config["treatments"]["variants_per_family"]
    )
    write_jsonl(paths["manifest"], [row.to_dict() for row in trials])
    write_json(paths["pairing"], pairing)
    write_json(
        paths["treatments"],
        {
            "source": "src/ziva/treatments.py",
            "source_sha256": sha256_file(ROOT / "src/ziva/treatments.py"),
            "treatments": [row.to_dict() for row in treatments],
        },
    )
    write_json(paths["resolved_config"], config)
    if len(trials) != 512 or pairing["failures"]:
        raise RuntimeError(f"Chat-v1 design mismatch: trials={len(trials)}, pairing={pairing['failures']}")
    print(json.dumps({"experiment": config["experiment_name"], "trials": 512, "pairing_failures": 0}, indent=2))


def _run_neutral_smoke(config: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(config)
    selected = _neutral_smoke_trials(config, paths["manifest"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["run"]["concurrency"]) as pool:
        records = list(
            pool.map(lambda row: execute_concise_trial(row, config, "neutral-engineering-only"), selected)
        )
    scenarios = json.loads((ROOT / config["scenario_file"]).read_text())
    scenario_by_id = {row["scenario_id"]: row for row in scenarios}
    parse_rate = sum(row["percentage_parse"]["value"] is not None for row in records) / len(records)
    turn_1_truncation = sum(row["turn_1"]["finish_reason"] == "length" for row in records) / len(records)
    turn_2_truncation = sum(row["turn_2"]["finish_reason"] == "length" for row in records) / len(records)
    below = [
        row["percentage_parse"]["value"]
        for row in records
        if row["scenario_id"] == config["anchors"]["below_horizon"]
        and row["percentage_parse"]["value"] is not None
    ]
    bright = [
        row["percentage_parse"]["value"]
        for row in records
        if row["scenario_id"] == config["anchors"]["bright_night"]
        and row["percentage_parse"]["value"] is not None
    ]
    no_think = all(
        "<think" not in (row["turn_1"]["text"] + row["turn_2"]["text"]).lower()
        for row in records
    )
    fact_fidelity = all(
        row["invariant_text"]
        == render_chat_evidence(scenario_by_id[row["scenario_id"]])
        + "\n\n"
        + config["conversation"]["turn_1_question"]
        for row in records
    )
    all_trials = _read_trials(paths["manifest"])
    cues = ("json", "benchmark", "evaluation", "scoring", "calibration", "logging")
    prompt_cues_absent = all(
        not any(cue in (row.system + "\n" + row.turn_1_user + "\n" + row.turn_2_user).lower() for cue in cues)
        for row in all_trials
    )
    engineering = config["turn_1_engineering"]
    qwen = config["model"]["model_family"] == "Qwen3"
    record = {
        "purpose": "neutral-only engineering; no treatment effects inspected",
        "model": config["model"],
        "turn_1_max_tokens": config["sampling"]["turn_1_max_tokens"],
        "turn_2_max_tokens": config["sampling"]["turn_2_max_tokens"],
        "n_trials": len(records),
        "scenario_classes": sorted({row["difficulty_class"] for row in records}),
        "parse_success_rate": parse_rate,
        "turn_1_truncation_rate": turn_1_truncation,
        "turn_2_truncation_rate": turn_2_truncation,
        "below_horizon_percentages": below,
        "bright_night_percentages": bright,
        "anchor_order_sensible": bool(below and bright and max(below) < 50 < min(bright)),
        "no_think_content": no_think,
        "qwen_thinking_disabled": (not qwen) or config["model"]["chat_template_kwargs"] == {"enable_thinking": False},
        "canonical_fact_rendering_exact": fact_fidelity,
        "prompt_cues_absent": prompt_cues_absent,
        "pairing_failures": json.loads(paths["pairing"].read_text())["failures"],
        "turn_2_followup_byte_identical": len({row.turn_2_user for row in all_trials}) == 1,
        "records": records,
    }
    record["passed"] = all(
        [
            parse_rate >= engineering["minimum_parse_rate"],
            turn_1_truncation <= engineering["maximum_truncation_rate"],
            turn_2_truncation <= engineering["maximum_turn_2_truncation_rate"],
            record["anchor_order_sensible"],
            (not qwen) or no_think,
            record["qwen_thinking_disabled"],
            fact_fidelity,
            prompt_cues_absent,
            record["pairing_failures"] == 0,
            record["turn_2_followup_byte_identical"],
        ]
    )
    return record


def engineering_smoke(config: dict[str, Any]) -> None:
    record = _run_neutral_smoke(config)
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    out = PROTOCOL_DIR / f"turn1_audit_{config['sampling']['turn_1_max_tokens']}.json"
    write_json(out, record)
    print(json.dumps({key: value for key, value in record.items() if key != "records"}, indent=2))


def freeze_protocol(config: dict[str, Any]) -> None:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files are dirty; commit protocol code before freezing")
    cap = config["sampling"]["turn_1_max_tokens"]
    audit_path = PROTOCOL_DIR / f"turn1_audit_{cap}.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed"):
        raise RuntimeError("passing neutral Turn-1 engineering audit required")
    sources = [
        PROTOCOL_PATH,
        QUEUE_PATH,
        ROOT / "src/ziva/chat_percentage.py",
        ROOT / "src/ziva/chat_percentage_concise.py",
        ROOT / "src/ziva/chat_percentage_analysis.py",
        ROOT / "src/ziva/oss_chat_v1_analysis.py",
        ROOT / "src/ziva/chat_eval.py",
        ROOT / "src/ziva/treatments.py",
        ROOT / "scripts/run_oss_chat_v1.py",
        ROOT / config["scenario_file"],
        audit_path,
    ]
    record = {
        "protocol_name": config["protocol_name"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "system_prompt": config["conversation"]["system_prompt"],
        "turn_1_question": config["conversation"]["turn_1_question"],
        "turn_2_followup": config["conversation"]["turn_2_followup"],
        "sampling": config["sampling"],
        "turn_1_selection": {
            "rule": "use 2048 unless neutral Turn-1 truncation exceeds 5%; then test 4096 once",
            "selected_budget": cap,
            "neutral_audit": str(audit_path.relative_to(ROOT)),
            "neutral_turn_1_truncation_rate": audit["turn_1_truncation_rate"],
        },
        "treatments": config["treatments"],
        "scenario_ids": sorted({row.scenario_id for row in _read_trials(_paths(config)["manifest"])}),
        "percentage_parser_source_sha256": sha256_text(inspect.getsource(parse_concise_percentage)),
        "evidence_renderer_source_sha256": sha256_text(inspect.getsource(render_chat_evidence)),
        "analysis": config["analysis"],
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
    }
    record["protocol_fingerprint"] = sha256_json(record)
    write_json(PROTOCOL_FREEZE, record)
    print(json.dumps(record, indent=2))


def _verify_protocol() -> dict[str, Any]:
    frozen = json.loads(PROTOCOL_FREEZE.read_text())
    for relative, expected in frozen["source_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"globally frozen Chat-v1 source changed: {relative}")
    return frozen


def freeze_model(config: dict[str, Any]) -> None:
    protocol = _verify_protocol()
    paths = _paths(config)
    sources = [
        paths["manifest"],
        paths["pairing"],
        paths["treatments"],
        paths["resolved_config"],
        REGISTRY_DIR / f"{config['model']['id']}.json",
        PROTOCOL_FREEZE,
    ]
    record = {
        "experiment_name": config["experiment_name"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "model": config["model"],
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
    }
    record["experiment_fingerprint"] = sha256_json(record)
    write_json(paths["freeze"], record)
    print(json.dumps(record, indent=2))


def _verify_model(config: dict[str, Any]) -> dict[str, Any]:
    protocol = _verify_protocol()
    frozen = json.loads(_paths(config)["freeze"].read_text())
    if frozen["protocol_fingerprint"] != protocol["protocol_fingerprint"]:
        raise RuntimeError("per-model freeze references a different Chat-v1 protocol")
    for relative, expected in frozen["source_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"frozen per-model source changed: {relative}")
    return frozen


def smoke(config: dict[str, Any]) -> None:
    _verify_model(config)
    record = _run_neutral_smoke(config)
    write_json(_paths(config)["smoke"], record)
    print(json.dumps({key: value for key, value in record.items() if key != "records"}, indent=2))
    if not record["passed"]:
        raise RuntimeError("neutral Chat-v1 smoke failed; treatment run blocked")


def _gpu_monitor(path: Path) -> subprocess.Popen[str] | None:
    try:
        handle = path.open("w", encoding="utf-8")
        return subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return None


def run(config: dict[str, Any]) -> None:
    frozen = _verify_model(config)
    paths = _paths(config)
    if not json.loads(paths["smoke"].read_text()).get("passed"):
        raise RuntimeError("passing neutral smoke required")
    trials = _read_trials(paths["manifest"])
    existing = {path.stem for path in paths["raw_dir"].glob("tcp_*.json")}
    pending = [row for row in trials if row.trial_id not in existing]
    started = datetime.now(UTC).isoformat()
    completed = errors = 0
    monitor = _gpu_monitor(paths["gpu_log"])
    try:
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
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
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


def analyze(config: dict[str, Any]) -> None:
    frozen = _verify_model(config)
    paths = _paths(config)
    summary, scenario, comparisons, previous = analyze_chat_v1_records(paths["raw_dir"], config)
    write_json(paths["summary"], summary)
    scenario.to_csv(paths["scenario_effects"], index=False)
    for name, frame in comparisons.items():
        frame.to_csv(paths["results_dir"] / f"{name}.csv", index=False)
    if previous is not None:
        previous.to_csv(paths["results_dir"] / "chat_v1_minus_previous_chat.csv", index=False)
    rows = []
    raw_records = []
    for path in sorted(paths["raw_dir"].glob("tcp_*.json")):
        row = json.loads(path.read_text())
        raw_records.append(row)
        rows.append(
            {
                "trial_id": row["trial_id"],
                "scenario_id": row["scenario_id"],
                "treatment_id": row["treatment_id"],
                "treatment_family": row["treatment_family"],
                "repeat_index": row["repeat_index"],
                "visible_probability_chat": row["percentage_parse"]["value"],
                "parse_status": row["percentage_parse"]["status"],
                "turn_1_finish_reason": row["turn_1"]["finish_reason"],
                "turn_2_finish_reason": row["turn_2"]["finish_reason"],
            }
        )
    pd.DataFrame(rows).to_csv(paths["trials_csv"], index=False)
    run_summary = json.loads(paths["run_summary"].read_text())
    parse_counts = pd.Series([row["percentage_parse"]["status"] for row in raw_records]).value_counts()
    audit = {
        "experiment_name": config["experiment_name"],
        "experiment_fingerprint": frozen["experiment_fingerprint"],
        "frozen_git_commit": frozen["git_commit"],
        "run_started_at_utc": run_summary["started_at_utc"],
        "run_finished_at_utc": run_summary["finished_at_utc"],
        "integrity": {
            "manifest_trials": len(_read_trials(paths["manifest"])),
            "raw_records": len(raw_records),
            "unique_trial_ids": len({row["trial_id"] for row in raw_records}),
            "execution_errors": sum(row["status"] != "ok" for row in raw_records),
            "records_matching_fingerprint": sum(
                row["experiment_fingerprint"] == frozen["experiment_fingerprint"] for row in raw_records
            ),
            "complete_five_message_transcripts": sum(len(row.get("transcript", [])) == 5 for row in raw_records),
            "turn_2_followup_mismatches": sum(
                row["turn_2_user"] != config["conversation"]["turn_2_followup"] for row in raw_records
            ),
            "think_marker_records": sum("<think" in json.dumps(row).lower() for row in raw_records),
            "parser_recomputation_mismatches": sum(
                parse_concise_percentage(row["turn_2"]["text"]).to_dict() != row["percentage_parse"]
                for row in raw_records
            ),
        },
        "model": config["model"],
        "sampling": config["sampling"],
        "generation": {
            "turn_1_finish_reasons": pd.Series([row["turn_1"]["finish_reason"] for row in raw_records]).value_counts().to_dict(),
            "turn_2_finish_reasons": pd.Series([row["turn_2"]["finish_reason"] for row in raw_records]).value_counts().to_dict(),
        },
        "parsing": {
            "valid": summary["valid_percentages"],
            "malformed": summary["malformed_percentages"],
            "malformed_rate": summary["malformed_rate"],
            "status_counts": parse_counts.to_dict(),
            "malformed_by_family": summary["reliability"]["malformed_by_family"],
            "malformed_by_template": summary["reliability"]["malformed_by_template"],
        },
    }
    write_json(paths["runtime_audit"], audit)
    paths["report"].write_text(render_chat_v1_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("build", "engineering-smoke", "freeze-protocol", "freeze", "smoke", "run", "analyze"),
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--turn-1-max-tokens", type=int)
    args = parser.parse_args()
    config = _config(args.model_key, args.turn_1_max_tokens)
    commands = {
        "build": lambda: build(config),
        "engineering-smoke": lambda: engineering_smoke(config),
        "freeze-protocol": lambda: freeze_protocol(config),
        "freeze": lambda: freeze_model(config),
        "smoke": lambda: smoke(config),
        "run": lambda: run(config),
        "analyze": lambda: analyze(config),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
