#!/usr/bin/env python3
"""Run DeepSeek-R1-Distill-Qwen-32B through frozen ZIVA-Chat-v1."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek_r1_distill_qwen_32b"
CHECKPOINT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
REVISION = "711ad2ea6aa40cfca18895e8aca02ab92df1a746"


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
        "experiment_name": f"oss_{MODEL_ID}_chat_v1",
        # The frozen analyzer requires a Structured reference. DeepSeek has no
        # Structured run, so retain the same clearly labeled common Qwen
        # reference used for API-only Chat-v1 models.
        "structured_scenario_effects": "results/pilot_qwen3_14b/scenario_level_diffs.csv",
        "structured_paired_table": "results/pilot_qwen3_14b/paired_table.csv",
        "model": {
            "id": MODEL_ID,
            "checkpoint": CHECKPOINT,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "parameter_scale": "32B",
            "model_family": "DeepSeek-R1-Distill-Qwen",
            "dtype": "bfloat16",
            "quantization": "none",
            "framework": "vllm",
            "framework_version": "0.26.0",
            "vllm_model_runner": "V1 (VLLM_USE_V2_MODEL_RUNNER=0; required for UVA weight offload)",
            "endpoint": "http://127.0.0.1:8000",
            "tensor_parallel_size": 2,
            "gpu_configuration": "2x NVIDIA GeForce RTX 3090 24GB",
            "cpu_offload_gb_per_gpu": 16,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.90,
            # Empty kwargs means the official checkpoint template is applied
            # without any model-specific text or Qwen3 thinking control.
            "chat_template_kwargs": {},
            "thinking_mode": "official checkpoint default; no prompt-level control added",
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
            "parameter_scale": "32B",
            "model_family": "DeepSeek-R1-Distill-Qwen",
            "resolved_at_utc": datetime.now(UTC).isoformat(),
            "resolution_source": "Hugging Face model_info resolved and pinned before download",
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
