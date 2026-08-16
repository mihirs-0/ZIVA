#!/usr/bin/env python3
"""Run an OpenAI API model through the frozen ZIVA-Chat-v1 protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ziva.chat_percentage_concise import ConcisePercentageParse, parse_concise_percentage
from ziva.util import sha256_json, write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai_gpt56_luna"
MODEL = "gpt-5.6-luna"


def _frozen_runner():
    path = ROOT / "scripts/run_oss_chat_v1.py"
    spec = importlib.util.spec_from_file_location("ziva_frozen_chat_v1_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(runner) -> dict[str, Any]:
    config = runner._load_yaml(runner.PROTOCOL_PATH)
    return {
        **config,
        "experiment_name": f"{MODEL_ID}_chat_v1",
        # A common structured reference is retained solely because the frozen
        # analysis requires it. It is clearly labeled as Qwen, not Luna.
        "structured_scenario_effects": "results/pilot_qwen3_14b/scenario_level_diffs.csv",
        "structured_paired_table": "results/pilot_qwen3_14b/paired_table.csv",
        "model": {
            "id": MODEL_ID,
            "checkpoint": MODEL,
            "revision": "API alias resolved per response model/system fingerprint",
            "tokenizer_revision": "provider managed",
            "parameter_scale": "undisclosed",
            "model_family": "GPT-5.6 Luna",
            "provider": "openai",
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "sdk": "openai-python",
            "reasoning_effort": "none",
            "api_key_env": "OPENAI_API_KEY",
        },
        "provider_parameter_mapping": {
            "sent": ["temperature", "top_p", "presence_penalty", "seed", "max_completion_tokens"],
            "omitted_unsupported": ["top_k", "min_p"],
            "reasoning_effort": "none",
        },
    }


class LunaClient:
    def __init__(self, config: dict[str, Any]):
        load_dotenv(ROOT / ".env", override=False)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not loaded")
        self.client = OpenAI(api_key=key)
        self.config = config

    def complete(self, messages: list[dict[str, str]], seed: int, max_tokens: int) -> dict[str, Any]:
        sampling = self.config["sampling"]
        request = {
            "model": MODEL,
            "messages": messages,
            "reasoning_effort": "none",
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "presence_penalty": sampling["presence_penalty"],
            "seed": seed,
            "max_completion_tokens": max_tokens,
        }
        started = time.monotonic()
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "model": response.model,
            "response_id": response.id,
            "system_fingerprint": response.system_fingerprint,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
            "latency_s": round(time.monotonic() - started, 3),
            "seed": seed,
            "request_messages_sha256": sha256_json(messages),
            "generation_settings": {
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "top_k": None,
                "min_p": None,
                "presence_penalty": sampling["presence_penalty"],
                "max_completion_tokens": max_tokens,
                "reasoning_effort": "none",
            },
        }


def execute_trial(trial, config: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    client = LunaClient(config)
    messages_1 = [{"role": "system", "content": trial.system}, {"role": "user", "content": trial.turn_1_user}]
    last_error: Exception | None = None
    for attempt in range(1, config["run"]["retries"] + 1):
        try:
            turn_1 = client.complete(messages_1, trial.turn_1_seed, config["sampling"]["turn_1_max_tokens"])
            messages_2 = messages_1 + [
                {"role": "assistant", "content": turn_1["text"]},
                {"role": "user", "content": trial.turn_2_user},
            ]
            turn_2 = client.complete(messages_2, trial.turn_2_seed, config["sampling"]["turn_2_max_tokens"])
            parsed = parse_concise_percentage(turn_2["text"])
            return {
                **trial.to_dict(), "experiment": config["experiment_name"],
                "experiment_fingerprint": fingerprint, "attempts": attempt, "status": "ok",
                "transcript": [*messages_1, {"role": "assistant", "content": turn_1["text"]},
                               {"role": "user", "content": trial.turn_2_user},
                               {"role": "assistant", "content": turn_2["text"]}],
                "turn_1": turn_1, "turn_2": turn_2, "percentage_parse": parsed.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(attempt)
    return {
        **trial.to_dict(), "experiment": config["experiment_name"],
        "experiment_fingerprint": fingerprint, "attempts": config["run"]["retries"],
        "status": "error", "error": repr(last_error),
        "percentage_parse": ConcisePercentageParse("execution_error", None, None, None, [repr(last_error)]).to_dict(),
    }


def write_registry() -> None:
    import openai
    path = ROOT / "data/manifests/oss_model_registry" / f"{MODEL_ID}.json"
    write_json(path, {
        "status": "RESOLVED", "model_id": MODEL_ID, "checkpoint": MODEL,
        "revision": "API alias resolved per response model/system fingerprint",
        "tokenizer_revision": "provider managed", "provider": "openai",
        "endpoint": "v1/chat/completions", "openai_sdk_version": openai.__version__,
        "reasoning_effort": "none", "resolved_at_utc": datetime.now(UTC).isoformat(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "freeze", "smoke", "run", "analyze"))
    args = parser.parse_args()
    runner = _frozen_runner()
    config = _config(runner)
    runner.execute_concise_trial = execute_trial
    if args.command == "build":
        write_registry()
        runner.build(config)
    else:
        {"freeze": runner.freeze_model, "smoke": runner.smoke, "run": runner.run,
         "analyze": runner.analyze}[args.command](config)


if __name__ == "__main__":
    main()
