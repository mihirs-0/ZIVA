#!/usr/bin/env python3
"""Run Qwen3-30B-A3B through the globally frozen ZIVA-Chat-v1 protocol."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_30b_a3b"
CHECKPOINT = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def _runner():
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
        "structured_scenario_effects": f"results/oss_{MODEL_ID}_structured/scenario_level_diffs.csv",
        "structured_paired_table": f"results/oss_{MODEL_ID}_structured/paired_table.csv",
        "model": {
            "id": MODEL_ID, "checkpoint": CHECKPOINT, "revision": REVISION,
            "tokenizer_revision": REVISION, "parameter_scale": "30B-A3B",
            "model_family": "Qwen3", "dtype": "bfloat16", "quantization": "none",
            "framework": "vllm", "framework_version": "0.26.0",
            "vllm_model_runner": "V1 (VLLM_USE_V2_MODEL_RUNNER=0; UVA weight offload)",
            "endpoint": "http://127.0.0.1:8000", "tensor_parallel_size": 2,
            "gpu_configuration": "2x NVIDIA GeForce RTX 3090 24GB",
            "cpu_offload_gb_per_gpu": 10, "max_num_seqs": 8,
            "kv_cache_memory_bytes_per_gpu": 1610612736, "max_model_len": 8192,
            "gpu_memory_utilization": 0.90,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
            "thinking_mode": "disabled through the official Qwen chat-template argument",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "freeze", "smoke", "run", "analyze"))
    args = parser.parse_args()
    runner = _runner()
    config = _config(runner)
    {"build": runner.build, "freeze": runner.freeze_model, "smoke": runner.smoke,
     "run": runner.run, "analyze": runner.analyze}[args.command](config)


if __name__ == "__main__":
    main()
