#!/usr/bin/env python3
"""Run GPT-5.6 Sol through frozen Chat-v1 with a hard cost ledger."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai_gpt56_sol"
MODEL = "gpt-5.6-sol"
INPUT_USD_PER_MILLION = 5.0
OUTPUT_USD_PER_MILLION = 30.0
HARD_LIMIT_USD = 12.0
PROJECTION_GATE_USD = 10.5
LEDGER_PATH = ROOT / "data/manifests" / f"{MODEL_ID}_chat_v1" / "cost_ledger.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CostLedger:
    """Conservative response-level accounting with in-flight reservations."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reserved_usd = 0.0
        if LEDGER_PATH.exists():
            self.payload = json.loads(LEDGER_PATH.read_text())
        else:
            self.payload = {
                "experiment_name": f"{MODEL_ID}_chat_v1",
                "model": MODEL,
                "pricing_source": "https://developers.openai.com/api/docs/pricing",
                "pricing_basis": "Standard service tier; all prompt tokens charged conservatively as uncached",
                "input_usd_per_million_tokens": INPUT_USD_PER_MILLION,
                "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION,
                "hard_limit_usd": HARD_LIMIT_USD,
                "cumulative_conservative_cost_usd": 0.0,
                "responses": [],
            }

    @staticmethod
    def reservation(max_tokens: int) -> float:
        # Every frozen prompt is far below 8,192 tokens. Reserving that full
        # input allowance plus the requested output cap keeps concurrency safe.
        return (8192 * INPUT_USD_PER_MILLION + max_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000

    def reserve(self, max_tokens: int) -> float:
        amount = self.reservation(max_tokens)
        with self.lock:
            committed = float(self.payload["cumulative_conservative_cost_usd"])
            if committed + self.reserved_usd + amount > HARD_LIMIT_USD:
                raise RuntimeError(
                    f"Sol hard cost limit would be exceeded: committed={committed:.6f}, "
                    f"reserved={self.reserved_usd:.6f}, requested={amount:.6f}, limit={HARD_LIMIT_USD:.2f}"
                )
            self.reserved_usd += amount
        return amount

    def release(self, amount: float) -> None:
        with self.lock:
            self.reserved_usd -= amount

    def commit(self, amount: float, response: dict[str, Any]) -> float:
        input_tokens = int(response.get("input_tokens") or 0)
        output_tokens = int(response.get("output_tokens") or 0)
        cost = (
            input_tokens * INPUT_USD_PER_MILLION
            + output_tokens * OUTPUT_USD_PER_MILLION
        ) / 1_000_000
        with self.lock:
            self.reserved_usd -= amount
            self.payload["cumulative_conservative_cost_usd"] = round(
                float(self.payload["cumulative_conservative_cost_usd"]) + cost, 8
            )
            self.payload["responses"].append(
                {
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "response_id": response.get("response_id"),
                    "response_model": response.get("model"),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": response.get("reasoning_tokens"),
                    "conservative_cost_usd": round(cost, 8),
                }
            )
            self.payload["updated_at_utc"] = datetime.now(UTC).isoformat()
            write_json(LEDGER_PATH, self.payload)
        return cost


def _configure():
    api_runner = _load(ROOT / "scripts/run_openai_chat_v1.py", "ziva_openai_chat_v1_runner")
    api_runner.MODEL_ID = MODEL_ID
    api_runner.MODEL = MODEL
    original_config = api_runner._config

    def sol_config(frozen_runner):
        config = original_config(frozen_runner)
        config["model"]["model_family"] = "GPT-5.6 Sol"
        config["cost_controls"] = {
            "pricing_source": "https://developers.openai.com/api/docs/pricing",
            "service_tier": "standard (implicit API default)",
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION,
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION,
            "projection_gate_usd": PROJECTION_GATE_USD,
            "hard_limit_usd": HARD_LIMIT_USD,
            "accounting": "conservative: cached input is charged at the full uncached input rate",
        }
        return config

    api_runner._config = sol_config
    ledger = CostLedger()
    base_client = api_runner.LunaClient

    class SolClient(base_client):
        def complete(self, messages, seed, max_tokens):
            reservation = ledger.reserve(max_tokens)
            try:
                response = super().complete(messages, seed, max_tokens)
            except Exception:
                ledger.release(reservation)
                raise
            response["conservative_cost_usd"] = ledger.commit(reservation, response)
            return response

    api_runner.LunaClient = SolClient
    frozen_runner = api_runner._frozen_runner()
    frozen_runner.execute_concise_trial = api_runner.execute_trial
    return api_runner, frozen_runner, sol_config(frozen_runner), ledger


def _write_registry() -> None:
    import openai

    write_json(
        ROOT / "data/manifests/oss_model_registry" / f"{MODEL_ID}.json",
        {
            "status": "RESOLVED",
            "model_id": MODEL_ID,
            "checkpoint": MODEL,
            "revision": "API alias resolved per response model/system fingerprint",
            "tokenizer_revision": "provider managed",
            "provider": "openai",
            "endpoint": "v1/chat/completions",
            "openai_sdk_version": openai.__version__,
            "reasoning_effort": "none",
            "service_tier": "standard (implicit API default)",
            "pricing_source": "https://developers.openai.com/api/docs/pricing",
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION,
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION,
            "resolved_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _cost_smoke(frozen_runner, config: dict[str, Any], ledger: CostLedger) -> None:
    frozen = frozen_runner._verify_model(config)
    paths = frozen_runner._paths(config)
    if not json.loads(paths["smoke"].read_text()).get("passed"):
        raise RuntimeError("passing frozen neutral smoke required before cost smoke")
    trials = frozen_runner._read_trials(paths["manifest"])
    selected = []
    seen = set()
    for trial in trials:
        if trial.treatment_id not in seen:
            selected.append(trial)
            seen.add(trial.treatment_id)
    if len(selected) != 8:
        raise RuntimeError(f"expected one trial for each of 8 frozen templates, found {len(selected)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["run"]["concurrency"]) as pool:
        records = list(
            pool.map(
                lambda row: frozen_runner.execute_concise_trial(
                    row, config, frozen["experiment_fingerprint"]
                ),
                selected,
            )
        )
    if any(row["status"] != "ok" for row in records):
        raise RuntimeError("representative cost smoke contained execution errors")
    payload = {
        "purpose": "representative cost measurement only; treatment effects were not inspected",
        "n_trials": len(records),
        "selection": "first frozen-manifest trial for each of the eight treatment templates",
        "treatment_effects_inspected": False,
        "records": records,
        "cumulative_conservative_cost_usd": ledger.payload["cumulative_conservative_cost_usd"],
    }
    write_json(paths["manifest_dir"] / "cost_smoke.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


def _project(frozen_runner, config: dict[str, Any], ledger: CostLedger) -> None:
    paths = frozen_runner._paths(config)
    cost_smoke = json.loads((paths["manifest_dir"] / "cost_smoke.json").read_text())
    records = cost_smoke["records"]
    input_tokens = [row["turn_1"]["input_tokens"] + row["turn_2"]["input_tokens"] for row in records]
    output_tokens = [row["turn_1"]["output_tokens"] + row["turn_2"]["output_tokens"] for row in records]
    mean_input = sum(input_tokens) / len(input_tokens)
    mean_output = sum(output_tokens) / len(output_tokens)
    projected_run = 512 * (
        mean_input * INPUT_USD_PER_MILLION + mean_output * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    spent = float(ledger.payload["cumulative_conservative_cost_usd"])
    projected_total = spent + projected_run
    payload = {
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
        "pricing_basis": "Standard tier; conservative uncached-input accounting",
        "cost_smoke_trials": len(records),
        "mean_input_tokens_per_trial": round(mean_input, 4),
        "mean_output_tokens_per_trial": round(mean_output, 4),
        "engineering_spend_to_date_usd": round(spent, 6),
        "projected_512_trial_run_usd": round(projected_run, 6),
        "projected_cumulative_experiment_usd": round(projected_total, 6),
        "projection_gate_usd": PROJECTION_GATE_USD,
        "hard_limit_usd": HARD_LIMIT_USD,
        "comfortably_below_projection_gate": projected_total < PROJECTION_GATE_USD,
        "treatment_effects_inspected": False,
    }
    write_json(paths["manifest_dir"] / "cost_projection.json", payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("build", "freeze", "smoke", "cost-smoke", "project", "run", "analyze"),
    )
    args = parser.parse_args()
    _, frozen_runner, config, ledger = _configure()
    if args.command == "build":
        _write_registry()
        frozen_runner.build(config)
    elif args.command == "cost-smoke":
        _cost_smoke(frozen_runner, config, ledger)
    elif args.command == "project":
        _project(frozen_runner, config, ledger)
    elif args.command == "run":
        projection = json.loads(
            (frozen_runner._paths(config)["manifest_dir"] / "cost_projection.json").read_text()
        )
        if not projection["comfortably_below_projection_gate"]:
            raise RuntimeError("Sol projected cost did not pass the $10.50 launch gate")
        frozen_runner.run(config)
    else:
        {
            "freeze": frozen_runner.freeze_model,
            "smoke": frozen_runner.smoke,
            "analyze": frozen_runner.analyze,
        }[args.command](config)


if __name__ == "__main__":
    main()
