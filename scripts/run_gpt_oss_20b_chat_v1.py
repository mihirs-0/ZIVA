#!/usr/bin/env python3
"""Run OpenAI GPT-OSS-20B through the frozen ZIVA-Chat-v1 protocol."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "gpt_oss_20b"
CHECKPOINT = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"


def _frozen_runner():
    path = ROOT / "scripts/run_oss_chat_v1.py"
    spec = importlib.util.spec_from_file_location("ziva_frozen_chat_v1_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(runner) -> dict[str, Any]:
    protocol = runner._load_yaml(runner.PROTOCOL_PATH)
    return {
        **protocol,
        "experiment_name": f"oss_{MODEL_ID}_chat_v1",
        # GPT-OSS has no ZIVA-Structured run. Preserve the existing, explicitly
        # labeled common Qwen reference used by other Chat-v1-only models.
        "structured_scenario_effects": "results/pilot_qwen3_14b/scenario_level_diffs.csv",
        "structured_paired_table": "results/pilot_qwen3_14b/paired_table.csv",
        "model": {
            "id": MODEL_ID,
            "checkpoint": CHECKPOINT,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "parameter_scale": "20B (3.6B active)",
            "model_family": "GPT-OSS",
            "dtype": "bfloat16",
            "quantization": "checkpoint-native MXFP4 MoE weights",
            "framework": "vllm",
            "framework_version": "0.26.0",
            "endpoint": "http://127.0.0.1:8000",
            "tensor_parallel_size": 2,
            "gpu_configuration": "2x NVIDIA GeForce RTX 3090 24GB",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.90,
            # Empty kwargs retain the official Harmony template behavior and
            # add no model-specific text to the frozen benchmark prompts.
            "chat_template_kwargs": {},
            "reasoning_mode": "official checkpoint default; no prompt-level control added",
            "reasoning_parser": "openai_gptoss",
        },
    }


def write_registry() -> None:
    write_json(
        ROOT / "data/manifests/oss_model_registry" / f"{MODEL_ID}.json",
        {
            "status": "RESOLVED",
            "key": MODEL_ID,
            "checkpoint": CHECKPOINT,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "parameter_scale": "20B (3.6B active)",
            "model_family": "GPT-OSS",
            "resolved_at_utc": datetime.now(UTC).isoformat(),
            "resolution_source": "existing Hugging Face cache snapshot pinned before inference",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "freeze", "smoke", "run", "analyze"))
    args = parser.parse_args()
    runner = _frozen_runner()
    config = _config(runner)
    if args.command == "build":
        write_registry()
        runner.build(config)
    else:
        {
            "freeze": runner.freeze_model,
            "smoke": runner.smoke,
            "run": runner.run,
            "analyze": runner.analyze,
        }[args.command](config)


if __name__ == "__main__":
    main()
