#!/usr/bin/env python3
"""Build, freeze, score, and analyze the Qwen3 JSON-less chat experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ziva.chat_eval import (
    ChatPrompt,
    ContinuationPair,
    LocalVLLMClient,
    audit_chat_pairing,
    batched,
    build_chat_prompts,
    continuation_logprob,
    load_chat_config,
    render_chat_evidence,
    sha256_file,
    source_sha256,
)
from ziva.chat_eval_analysis import analyze_chat_scores, render_chat_report
from ziva.treatments import select_treatments
from ziva.util import sha256_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pilot_qwen3_14b_chat.yaml"


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    name = config["experiment_name"]
    return {
        "manifest_dir": ROOT / "data/manifests" / name,
        "results_dir": ROOT / "results" / name,
        "manifest": ROOT / "data/manifests" / name / "prompts.jsonl",
        "pairing": ROOT / "data/manifests" / name / "pairing_audit.json",
        "treatments": ROOT / "data/manifests" / name / "treatments.json",
        "tokenizer_audit": ROOT / "data/manifests" / name / "continuation_audit.json",
        "freeze": ROOT / "data/manifests" / name / "freeze.json",
        "anchors": ROOT / "data/manifests" / name / "anchor_sanity.json",
        "scores": ROOT / "results" / name / "scores.jsonl",
        "summary": ROOT / "results" / name / "summary.json",
        "scenario_effects": ROOT / "results" / name / "scenario_level_effects.csv",
        "report": ROOT / "reports" / f"{name}_report.md",
    }


def _read_prompts(path: Path) -> list[ChatPrompt]:
    return [ChatPrompt(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build(config: dict[str, Any], config_path: Path) -> None:
    paths = _paths(config)
    paths["manifest_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    scenarios_path = ROOT / config["scenario_file"]
    scenarios = json.loads(scenarios_path.read_text())
    prompts = build_chat_prompts(config, scenarios)
    audit = audit_chat_pairing(prompts)
    treatments = select_treatments(
        config["treatments"]["families"], config["treatments"]["variants_per_family"]
    )
    write_jsonl(paths["manifest"], [row.to_dict() for row in prompts])
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
                "prompts": len(prompts),
                "families": config["treatments"]["families"],
                "variants_per_family": config["treatments"]["variants_per_family"],
                "pairing_failures": audit["failures"],
                "scenario_source_sha256": sha256_file(scenarios_path),
                "config_sha256": sha256_file(config_path),
            },
            indent=2,
        )
    )


def tokenizer_audit(config: dict[str, Any]) -> None:
    paths = _paths(config)
    prompts = _read_prompts(paths["manifest"])
    audit_prompt = next(row for row in prompts if row.treatment_id == "neutral_v1")
    client = LocalVLLMClient(
        config["model"]["endpoint"],
        config["model"]["checkpoint"],
        config["model"]["enable_thinking"],
    )
    context_ids, context_strs = client.tokenize_context(audit_prompt)
    rows = []
    for pair in (ContinuationPair(**row) for row in config["scoring"]["continuation_pairs"]):
        visible_ids, visible_strs = client.tokenize_continuation(
            audit_prompt, pair.visible, context_ids
        )
        negative_ids, negative_strs = client.tokenize_continuation(
            audit_prompt, pair.not_visible, context_ids
        )
        rows.append(
            {
                **asdict(pair),
                "visible_token_ids": visible_ids[len(context_ids) :],
                "not_visible_token_ids": negative_ids[len(context_ids) :],
                "visible_token_strs": visible_strs,
                "not_visible_token_strs": negative_strs,
                "visible_token_count": len(visible_ids) - len(context_ids),
                "not_visible_token_count": len(negative_ids) - len(context_ids),
                "token_count_difference": (len(visible_ids) - len(context_ids))
                - (len(negative_ids) - len(context_ids)),
                "semantic_symmetry_audit": "matched affirmative and negative factual dispositions",
                "grammatical_fit": True,
            }
        )
    write_json(
        paths["tokenizer_audit"],
        {
            "performed_before_treatment_scoring": True,
            "audit_prompt_hash": audit_prompt.prompt_hash,
            "context_token_count": len(context_ids),
            "assistant_prefix_tail_token_strs": context_strs[-8:],
            "thinking_mode": "enable_thinking=false; continuation begins after the template's empty think wrapper",
            "primary_normalization_precommitted": config["scoring"]["primary"],
            "primary_normalization_reason": "per-token mean controls the observed unequal continuation token counts",
            "pairs": rows,
        },
    )
    print(json.dumps({"continuation_pairs": rows}, indent=2))


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = _paths(config)
    if not paths["tokenizer_audit"].exists():
        raise RuntimeError("run tokenizer-audit before freeze")
    source_paths = [
        config_path,
        ROOT / "src/ziva/chat_eval.py",
        ROOT / "src/ziva/chat_eval_analysis.py",
        ROOT / "scripts/run_qwen3_chat_eval.py",
        ROOT / "src/ziva/treatments.py",
        ROOT / config["scenario_file"],
        paths["manifest"],
        paths["pairing"],
        paths["treatments"],
        paths["tokenizer_audit"],
    ]
    tracked_dirty = _git("status", "--porcelain", "--untracked-files=no")
    if tracked_dirty:
        raise RuntimeError("tracked files are dirty; commit implementation before freezing")
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    freeze_record = {
        "experiment_name": config["experiment_name"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "system_prompt": config["conversation"]["system_prompt"],
        "question": config["conversation"]["question"],
        "renderer_source_sha256": source_sha256(render_chat_evidence),
        "treatment_snapshot_sha256": sha256_file(paths["treatments"]),
        "continuation_bank": config["scoring"]["continuation_pairs"],
        "continuation_scoring_rule": config["scoring"]["definition"],
        "primary_normalization": config["scoring"]["primary"],
        "robustness_normalization": config["scoring"]["robustness"],
        "model": config["model"],
        "scenario_ids": sorted({row.scenario_id for row in _read_prompts(paths["manifest"])}),
        "source_hashes": hashes,
    }
    freeze_record["experiment_fingerprint"] = sha256_json(freeze_record)
    write_json(paths["freeze"], freeze_record)
    print(json.dumps(freeze_record, indent=2))


def _verify_freeze(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    paths = _paths(config)
    frozen = json.loads(paths["freeze"].read_text())
    for relative, expected in frozen["source_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen source changed: {relative}: {actual} != {expected}")
    if sha256_file(config_path) != frozen["source_hashes"][str(config_path.relative_to(ROOT))]:
        raise RuntimeError("configuration changed after freeze")
    return frozen


def _score_prompts(
    config: dict[str, Any], prompts: list[ChatPrompt]
) -> list[dict[str, Any]]:
    client = LocalVLLMClient(
        config["model"]["endpoint"],
        config["model"]["checkpoint"],
        config["model"]["enable_thinking"],
    )
    pairs = [ContinuationPair(**row) for row in config["scoring"]["continuation_pairs"]]
    tasks: list[dict[str, Any]] = []
    contexts: dict[str, list[int]] = {}
    for prompt in prompts:
        context_ids, _ = client.tokenize_context(prompt)
        contexts[prompt.prompt_hash] = context_ids
        for pair in pairs:
            for side, continuation in (("visible", pair.visible), ("not_visible", pair.not_visible)):
                token_ids, token_strs = client.tokenize_continuation(
                    prompt, continuation, context_ids
                )
                tasks.append(
                    {
                        "prompt": prompt,
                        "pair": pair,
                        "side": side,
                        "continuation": continuation,
                        "token_ids": token_ids,
                        "continuation_token_strs": token_strs,
                        "continuation_start": len(context_ids),
                    }
                )
    for task_batch in batched(tasks, config["run"]["batch_size"]):
        logprob_rows = client.score_sequences(
            [task["token_ids"] for task in task_batch], config["scoring"]["prompt_logprobs"]
        )
        for task, token_logprobs in zip(task_batch, logprob_rows):
            total, mean, token_values = continuation_logprob(
                task["token_ids"], token_logprobs, task["continuation_start"]
            )
            task["total_logprob"] = total
            task["mean_logprob"] = mean
            task["continuation_token_logprobs"] = token_values

    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        prompt = task["prompt"]
        pair = task["pair"]
        key = (prompt.prompt_hash, pair.id)
        row = combined.setdefault(
            key,
            {
                **prompt.to_dict(),
                "continuation_pair_id": pair.id,
                "visible_continuation": pair.visible,
                "not_visible_continuation": pair.not_visible,
                "context_token_count": len(contexts[prompt.prompt_hash]),
            },
        )
        side = task["side"]
        row[f"{side}_token_count"] = len(task["token_ids"]) - task["continuation_start"]
        row[f"{side}_token_strs"] = task["continuation_token_strs"]
        row[f"{side}_token_logprobs"] = task["continuation_token_logprobs"]
        row[f"{side}_total_logprob"] = task["total_logprob"]
        row[f"{side}_mean_logprob"] = task["mean_logprob"]
    rows = []
    for row in combined.values():
        row["total_margin"] = row["visible_total_logprob"] - row["not_visible_total_logprob"]
        row["per_token_margin"] = row["visible_mean_logprob"] - row["not_visible_mean_logprob"]
        rows.append(row)
    return rows


def smoke(config: dict[str, Any], config_path: Path) -> None:
    _verify_freeze(config, config_path)
    paths = _paths(config)
    prompts = _read_prompts(paths["manifest"])
    anchor_cfg = config["anchors"]
    chosen = [
        row
        for row in prompts
        if row.treatment_id == anchor_cfg["treatment_id"]
        and row.scenario_id in {anchor_cfg["below_horizon"], anchor_cfg["bright_night"]}
    ]
    if len(chosen) != 2:
        raise RuntimeError(f"expected two anchor prompts, found {len(chosen)}")
    scores = _score_prompts(config, chosen)
    primary_column = (
        "per_token_margin"
        if config["scoring"]["primary"] == "per_token_mean_logprob_margin"
        else "total_margin"
    )
    by_scenario: dict[str, list[float]] = {}
    for row in scores:
        by_scenario.setdefault(row["scenario_id"], []).append(row[primary_column])
    below = sum(by_scenario[anchor_cfg["below_horizon"]]) / len(
        by_scenario[anchor_cfg["below_horizon"]]
    )
    bright = sum(by_scenario[anchor_cfg["bright_night"]]) / len(
        by_scenario[anchor_cfg["bright_night"]]
    )
    record = {
        "purpose": "engineering construct sanity only; no treatment effects inspected",
        "below_horizon_scenario": anchor_cfg["below_horizon"],
        "bright_night_scenario": anchor_cfg["bright_night"],
        "below_horizon_margin": below,
        "bright_night_margin": bright,
        "below_horizon_favors_not_visible": below < 0,
        "bright_night_favors_visible": bright > 0,
        "ordered_sensibly": bright > below,
        "passed": below < 0 and bright > 0 and bright > below,
        "scores": scores,
    }
    write_json(paths["anchors"], record)
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise RuntimeError("anchor sanity check failed; do not run treatment comparison")


def run(config: dict[str, Any], config_path: Path) -> None:
    frozen = _verify_freeze(config, config_path)
    paths = _paths(config)
    anchors = json.loads(paths["anchors"].read_text())
    if not anchors.get("passed"):
        raise RuntimeError("passing anchor smoke test is required before full scoring")
    prompts = _read_prompts(paths["manifest"])
    rows = _score_prompts(config, prompts)
    for row in rows:
        row["experiment_fingerprint"] = frozen["experiment_fingerprint"]
        row["model_revision"] = config["model"]["revision"]
        row["enable_thinking"] = config["model"]["enable_thinking"]
        row["scored_at_utc"] = datetime.now(UTC).isoformat()
    write_jsonl(paths["scores"], rows)
    print(json.dumps({"prompts": len(prompts), "score_rows": len(rows)}, indent=2))


def analyze(config: dict[str, Any], config_path: Path) -> None:
    _verify_freeze(config, config_path)
    paths = _paths(config)
    anchors = json.loads(paths["anchors"].read_text())
    summary, scenario = analyze_chat_scores(paths["scores"], config, anchors=anchors)
    write_json(paths["summary"], summary)
    paths["scenario_effects"].parent.mkdir(parents=True, exist_ok=True)
    scenario.to_csv(paths["scenario_effects"], index=False)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(render_chat_report(summary, config), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "tokenizer-audit", "freeze", "smoke", "run", "analyze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_chat_config(config_path)
    commands = {
        "build": lambda: build(config, config_path),
        "tokenizer-audit": lambda: tokenizer_audit(config),
        "freeze": lambda: freeze(config, config_path),
        "smoke": lambda: smoke(config, config_path),
        "run": lambda: run(config, config_path),
        "analyze": lambda: analyze(config, config_path),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
