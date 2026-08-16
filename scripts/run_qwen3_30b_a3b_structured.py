#!/usr/bin/env python3
"""Run Qwen3-30B-A3B through the unchanged frozen ZIVA-Structured path."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_30b_a3b"
CHECKPOINT = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def _runner():
    path = ROOT / "scripts/run_oss_structured.py"
    spec = importlib.util.spec_from_file_location("ziva_frozen_structured_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = {
        "key": MODEL_ID, "checkpoint": CHECKPOINT, "parameter_scale": "30B-A3B",
        "model_family": "Qwen3", "chat_template_kwargs": {"enable_thinking": False},
    }
    runtime = {
        "dtype": "bfloat16", "quantization": "none", "framework": "vllm",
        "framework_version": "0.26.0", "endpoint": "http://127.0.0.1:8000",
        "tensor_parallel_size": 2, "gpu_configuration": "2x NVIDIA GeForce RTX 3090 24GB",
        "max_model_len": 8192, "gpu_memory_utilization": 0.90,
        "generation_config_source": "vllm",
    }
    module._model_spec = lambda key: (model, runtime) if key == MODEL_ID else (_ for _ in ()).throw(RuntimeError(key))
    return module


def write_registry() -> None:
    write_json(ROOT / "data/manifests/oss_model_registry" / f"{MODEL_ID}.json", {
        "status": "RESOLVED", "key": MODEL_ID, "checkpoint": CHECKPOINT,
        "revision": REVISION, "tokenizer_revision": REVISION,
        "parameter_scale": "30B-A3B", "model_family": "Qwen3",
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "resolution_source": "Hugging Face model_info resolved and pinned before download",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "freeze", "smoke", "run", "analyze"))
    args = parser.parse_args()
    runner = _runner()
    if args.command == "prepare":
        write_registry()
    {"prepare": runner.prepare, "freeze": runner.freeze, "smoke": runner.smoke,
     "run": runner.run, "analyze": runner.analyze}[args.command](MODEL_ID)


if __name__ == "__main__":
    main()
