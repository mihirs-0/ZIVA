#!/usr/bin/env python3
"""Run GPT-5.6 Terra through the frozen OpenAI ZIVA-Chat-v1 pathway."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "scripts/run_openai_chat_v1.py"
    spec = importlib.util.spec_from_file_location("ziva_openai_chat_v1_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OpenAI Chat-v1 runner: {path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.MODEL_ID = "openai_gpt56_terra"
    runner.MODEL = "gpt-5.6-terra"
    original_config = runner._config

    def terra_config(frozen_runner):
        config = original_config(frozen_runner)
        config["model"]["model_family"] = "GPT-5.6 Terra"
        return config

    runner._config = terra_config
    runner.main()


if __name__ == "__main__":
    main()
