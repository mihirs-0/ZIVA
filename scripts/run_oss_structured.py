#!/usr/bin/env python3
"""Execute the unchanged ZIVA-Structured protocol for one resolved OSS model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ziva.analysis import analyze as analyze_structured
from ziva.cli import _scoring_spec
from ziva.config import ExperimentConfig, ModelConfig, load_experiment_config, load_models_config
from ziva.freeze import HYPOTHESES, build_freeze_record, experiment_fingerprint, verify_freeze
from ziva.manifest import build_manifest, manifest_path
from ziva.prompts import SYSTEM_PROMPT, audit_pairing, compile_prompt
from ziva.report import write_report
from ziva.runner import run_manifest
from ziva.treatments import get_treatment, select_treatments
from ziva.util import read_json, read_jsonl, sha256_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "configs/oss_model_queue.yaml"
REGISTRY_DIR = ROOT / "data/manifests/oss_model_registry"
SOURCE_EXPERIMENT = "pilot_qwen3_14b"


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
        raise RuntimeError(f"missing resolved registry {path}")
    row = json.loads(path.read_text())
    if row.get("status") != "RESOLVED":
        raise RuntimeError(f"model {key} is not resolved")
    return row


def _name(key: str) -> str:
    return f"oss_{key}_structured"


def _manifest_dir(key: str) -> Path:
    return ROOT / "data/manifests" / _name(key)


def _config_path(key: str) -> Path:
    return _manifest_dir(key) / "experiment_config.yaml"


def _models_path(key: str) -> Path:
    return _manifest_dir(key) / "models.yaml"


def _cfg(key: str) -> ExperimentConfig:
    return load_experiment_config(_config_path(key))


def _models(key: str) -> list[ModelConfig]:
    os.environ.setdefault("VLLM_API_KEY", "local")
    return load_models_config(_models_path(key))


def prepare(key: str) -> None:
    spec, runtime = _model_spec(key)
    registry = _registry(key)
    name = _name(key)
    manifest_dir = _manifest_dir(key)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    model_data = {
        "models": [
            {
                "id": key,
                "provider": "openai_compatible",
                "model": spec["checkpoint"],
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key_env": "VLLM_API_KEY",
                "pricing": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
                "supports": {"json_mode": True, "images": False, "web_search": False, "temperature": True},
                "max_concurrent": 8,
                "min_interval_s": 0.0,
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "chat_template_kwargs": spec.get("chat_template_kwargs", {}),
                "runtime_metadata": {
                    "checkpoint": spec["checkpoint"],
                    "revision": registry["revision"],
                    "tokenizer_revision": registry["tokenizer_revision"],
                    "weights_dtype": runtime["dtype"],
                    "quantization": runtime["quantization"],
                    "framework": runtime["framework"],
                    "framework_version": runtime["framework_version"],
                    "gpu_configuration": runtime["gpu_configuration"],
                    "tensor_parallel_size": runtime["tensor_parallel_size"],
                    "max_model_len": runtime["max_model_len"],
                    "gpu_memory_utilization": runtime["gpu_memory_utilization"],
                    "generation_config_source": runtime["generation_config_source"],
                },
            }
        ]
    }
    config_data = {
        "experiment_name": name,
        "scenarios": {"count": 30, "year": 2026, "seed": 20260815, "include_case_000": True},
        "treatments": {
            "families": ["neutral", "excited_positive", "skeptical_negative"],
            "variants_per_family": 2,
        },
        "evidence_modes": ["structured"],
        "elicitation_modes": ["naturalistic", "separated"],
        "sampling": {"repeats": 2, "temperature": 0.7, "max_tokens": 2000},
        "run": {
            "max_cost_usd": 1.0,
            "concurrency": 8,
            "retries": 3,
            "retry_backoff_s": 2.0,
            "shuffle_seed": 7,
        },
        "experiments": {"primary": True, "evidence_update": False, "commitment": False, "web": False},
        "models_config": str(_models_path(key).relative_to(ROOT)),
        "data_dir": "data",
        "results_dir": "results",
    }
    _models_path(key).write_text(yaml.safe_dump(model_data, sort_keys=False), encoding="utf-8")
    _config_path(key).write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    cfg = _cfg(key)
    models = _models(key)

    source_scenarios = ROOT / "data/scenarios" / SOURCE_EXPERIMENT / "scenarios.json"
    cfg.scenario_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_scenarios, cfg.scenario_dir / "scenarios.json")
    source_stimuli = ROOT / "data/stimuli" / SOURCE_EXPERIMENT
    cfg.stimuli_dir.mkdir(parents=True, exist_ok=True)
    for path in source_stimuli.iterdir():
        if path.is_file():
            shutil.copy2(path, cfg.stimuli_dir / path.name)
    scenarios = read_json(cfg.scenario_dir / "scenarios.json")
    image_hashes = read_json(cfg.stimuli_dir / "hashes.json")
    rows = build_manifest(cfg, scenarios, models, image_hashes)
    write_jsonl(manifest_path(cfg), rows)
    treatments = select_treatments(cfg.treatments.families, cfg.treatments.variants_per_family)
    audits = [
        audit_pairing(scenario, treatments, mode, elicitation)
        for scenario in scenarios
        for mode in cfg.evidence_modes
        for elicitation in cfg.elicitation_modes
    ]
    write_json(cfg.manifest_dir / "pairing_audit.json", audits)
    if len(rows) != 768:
        raise RuntimeError(f"expected 768 structured trials, found {len(rows)}")
    if sha256_json(scenarios) != sha256_json(read_json(source_scenarios)):
        raise RuntimeError("structured scenario copy differs from banked source")
    print(json.dumps({"experiment": name, "trials": len(rows), "pairing_audits": len(audits)}, indent=2))


def freeze(key: str) -> None:
    cfg = _cfg(key)
    models = _models(key)
    scenarios = cfg.scenario_dir / "scenarios.json"
    rows = manifest_path(cfg)
    image_hashes = read_json(cfg.stimuli_dir / "hashes.json")
    record = build_freeze_record(cfg, models, scenarios, rows, image_hashes)
    write_json(cfg.freeze_path, record)
    (cfg.manifest_dir / "hypotheses.yaml").write_text(
        yaml.safe_dump(HYPOTHESES, sort_keys=False), encoding="utf-8"
    )
    (cfg.manifest_dir / "scoring_spec.md").write_text(_scoring_spec(), encoding="utf-8")
    print(json.dumps(record, indent=2))


def _verify(key: str) -> tuple[ExperimentConfig, list[ModelConfig], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = _cfg(key)
    models = _models(key)
    scenarios = read_json(cfg.scenario_dir / "scenarios.json")
    rows = read_jsonl(manifest_path(cfg))
    check = verify_freeze(cfg, cfg.scenario_dir / "scenarios.json", manifest_path(cfg), cfg.stimuli_dir, models)
    if not check["ok"]:
        raise RuntimeError(f"structured freeze verification failed: {check['violations']}")
    return cfg, models, scenarios, rows


def _smoke_rows(rows: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, str] = {}
    for scenario in scenarios:
        by_class.setdefault(scenario["classification"]["difficulty_class"], scenario["scenario_id"])
    ids = {"scn_0004_trivial_invisible", "scn_0008_night_easy_visible", by_class["moderate"], by_class["ambiguous"]}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["order_index"]):
        if row["treatment_family"] != "neutral" or row["scenario_id"] not in ids:
            continue
        key = (row["scenario_id"], row["elicitation_mode"], row["treatment_id"])
        selected.setdefault(key, row)
    result = sorted(selected.values(), key=lambda item: item["order_index"])
    if len(result) != 16:
        raise RuntimeError(f"expected 16 neutral structured smoke rows, found {len(result)}")
    return result


def smoke(key: str) -> None:
    cfg, models, scenarios, rows = _verify(key)
    selected = _smoke_rows(rows, scenarios)
    fingerprint = experiment_fingerprint(cfg, models)
    run_manifest(cfg, selected, scenarios, models, SYSTEM_PROMPT, fingerprint=fingerprint, ran_dirty=False)
    scenario_by_id = {row["scenario_id"]: row for row in scenarios}
    records = [read_json(cfg.raw_dir / f"{row['trial_id']}.json") for row in selected]
    statuses = Counter((row.get("final_parse") or {}).get("status") for row in records)
    texts = [row["steps"][-1]["result"]["text"] for row in records]
    think = sum("<think" in text.lower() for text in texts)
    exact_prompt = all(
        row["steps"][-1]["result"]["raw"]["request_messages_sha256"]
        == sha256_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": compile_prompt(
                        scenario_by_id[row["scenario_id"]],
                        get_treatment(row["treatment_id"]),
                        row["evidence_mode"],
                        row["elicitation_mode"],
                    ).user_text,
                },
            ]
        )
        for row in records
    )
    values = {
        scenario_id: [
            row["final_parse"]["parsed"]["visible_probability"]
            for row in records
            if row["scenario_id"] == scenario_id and row["final_parse"]["status"] == "ok"
        ]
        for scenario_id in ("scn_0004_trivial_invisible", "scn_0008_night_easy_visible")
    }
    qwen = _model_spec(key)[0]["model_family"] == "Qwen3"
    report = {
        "purpose": "neutral-only structured engineering smoke; no treatment effects inspected",
        "n_trials": len(records),
        "parse_status_counts": dict(statuses),
        "parse_success_rate": statuses.get("ok", 0) / len(records),
        "think_marker_records": think,
        "exact_outbound_messages_hash_match": exact_prompt,
        "anchor_values": values,
        "anchor_order_sensible": bool(
            values["scn_0004_trivial_invisible"]
            and values["scn_0008_night_easy_visible"]
            and max(values["scn_0004_trivial_invisible"]) < 50
            and min(values["scn_0008_night_easy_visible"]) > 50
        ),
    }
    report["passed"] = all(
        [
            report["parse_success_rate"] >= 0.9,
            (not qwen) or think == 0,
            exact_prompt,
            report["anchor_order_sensible"],
        ]
    )
    write_json(cfg.manifest_dir / "neutral_smoke.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("neutral structured smoke failed; treatment run blocked")


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


def run(key: str) -> None:
    cfg, models, scenarios, rows = _verify(key)
    if not read_json(cfg.manifest_dir / "neutral_smoke.json").get("passed"):
        raise RuntimeError("passing structured neutral smoke required")
    started = datetime.now(UTC).isoformat()
    monitor = _gpu_monitor(cfg.manifest_dir / "full_gpu_dmon.txt")
    progress = {"n": 0}

    def report_progress(_row: dict[str, Any], _status: str | None) -> None:
        progress["n"] += 1
        if progress["n"] % 25 == 0:
            print(f"completed={progress['n']}", flush=True)

    try:
        summary = run_manifest(
            cfg,
            rows,
            scenarios,
            models,
            SYSTEM_PROMPT,
            fingerprint=experiment_fingerprint(cfg, models),
            ran_dirty=False,
            progress_cb=report_progress,
        )
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
    summary["started_at_utc"] = started
    write_json(cfg.raw_dir / "_run_summary.json", summary)
    print(json.dumps(summary, indent=2))


def analyze(key: str) -> None:
    cfg, models, scenarios, rows = _verify(key)
    summary = analyze_structured(cfg, scenarios)
    write_report(cfg, ROOT / "reports" / f"{cfg.experiment_name}_report.md")
    raw = [read_json(path) for path in sorted(cfg.raw_dir.glob("t_*.json"))]
    run_summary = read_json(cfg.raw_dir / "_run_summary.json")
    runtime = {
        "experiment_name": cfg.experiment_name,
        "experiment_fingerprint": summary["experiment_fingerprints_in_data"][0],
        "frozen_git_commit": read_json(cfg.freeze_path)["git"]["commit"],
        "run_started_at_utc": run_summary.get("started_at_utc"),
        "run_finished_at_utc": run_summary.get("finished_at_utc"),
        "integrity": {
            "manifest_trials": len(rows),
            "raw_records": len(raw),
            "unique_trial_ids": len({row["trial_id"] for row in raw}),
            "execution_errors": sum((row.get("final_parse") or {}).get("status") not in {"ok", "json_error", "validation_error", "empty"} for row in raw),
            "think_marker_records": sum("<think" in json.dumps(row).lower() for row in raw),
        },
        "model": models[0].runtime_metadata,
        "sampling": cfg.sampling.model_dump(),
        "parsing": summary["parse_stats"],
    }
    write_json(cfg.results_path / "runtime_audit.json", runtime)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "freeze", "smoke", "run", "analyze"))
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    commands = {
        "prepare": lambda: prepare(args.model_key),
        "freeze": lambda: freeze(args.model_key),
        "smoke": lambda: smoke(args.model_key),
        "run": lambda: run(args.model_key),
        "analyze": lambda: analyze(args.model_key),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
