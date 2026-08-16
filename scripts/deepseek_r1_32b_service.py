#!/usr/bin/env python3
"""Serve the pinned DeepSeek 32B checkpoint on mihir's two RTX 3090s."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek_r1_distill_qwen_32b"
CHECKPOINT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
REVISION = "711ad2ea6aa40cfca18895e8aca02ab92df1a746"
REGISTRY_DIR = ROOT / "data/manifests/oss_model_registry"
SERVICE_PATH = REGISTRY_DIR / f"{MODEL_ID}_service.json"
LOG_PATH = REGISTRY_DIR / f"{MODEL_ID}_vllm.log"


def _shared():
    path = ROOT / "scripts/oss_model_service.py"
    spec = importlib.util.spec_from_file_location("ziva_oss_model_service", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load service helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status() -> None:
    shared = _shared()
    print(
        json.dumps(
            {
                "expected_model": CHECKPOINT,
                "endpoint_model": shared._endpoint_model(),
                "owned_vllm_pids": shared._owned_vllm_pids(),
                "gpus": shared._gpu_snapshot(),
            },
            indent=2,
        )
    )


def serve() -> None:
    shared = _shared()
    registry = json.loads((REGISTRY_DIR / f"{MODEL_ID}.json").read_text())
    if registry.get("revision") != REVISION:
        raise RuntimeError("resolved registry does not match pinned revision")
    current = shared._endpoint_model()
    if current == CHECKPOINT:
        print(json.dumps({"status": "already_serving", "model": current}, indent=2))
        return
    existing = shared._owned_vllm_pids()
    if existing:
        raise RuntimeError(f"another mihir-owned vLLM process is present: {existing}")

    executable = shared._vllm_executable()
    command = [
        str(executable),
        "serve",
        CHECKPOINT,
        "--revision",
        REVISION,
        "--served-model-name",
        CHECKPOINT,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.9",
        "--cpu-offload-gb",
        "12",
        "--generation-config",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(UTC).isoformat()}] command={json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env=environment,
        )
    record = {
        "status": "STARTING",
        "key": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "pid": process.pid,
        "command": command,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dtype": "bfloat16",
        "quantization": "none",
        "tensor_parallel_size": 2,
        "cpu_offload_gb_per_gpu": 12,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "versions": shared._versions(executable),
        "environment_overrides": {
            key: environment[key]
            for key in ("HF_HUB_DISABLE_XET", "HF_HUB_OFFLINE", "VLLM_USE_FLASHINFER_SAMPLER")
        },
        "chat_template": "official checkpoint tokenizer chat template",
        "log_path": str(LOG_PATH.relative_to(ROOT)),
    }
    write_json(SERVICE_PATH, record)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            record.update(
                status="FAILED_STARTUP",
                exit_code=process.returncode,
                finished_at_utc=datetime.now(UTC).isoformat(),
            )
            write_json(SERVICE_PATH, record)
            raise RuntimeError(f"vLLM exited with code {process.returncode}; see {LOG_PATH}")
        if shared._endpoint_model() == CHECKPOINT:
            record.update(
                status="READY",
                ready_at_utc=datetime.now(UTC).isoformat(),
                gpus=shared._gpu_snapshot(),
            )
            write_json(SERVICE_PATH, record)
            print(json.dumps(record, indent=2))
            return
        time.sleep(5)
    record["status"] = "STARTUP_TIMEOUT"
    write_json(SERVICE_PATH, record)
    raise RuntimeError(f"vLLM did not become ready within 30 minutes; see {LOG_PATH}")


def stop() -> None:
    shared = _shared()
    endpoint = shared._endpoint_model()
    if endpoint is not None and endpoint != CHECKPOINT:
        raise RuntimeError(f"refusing to stop unexpected endpoint {endpoint!r}")
    pids = shared._owned_vllm_pids()
    if not pids:
        print(json.dumps({"status": "already_stopped"}, indent=2))
        return
    if len(pids) != 1:
        raise RuntimeError(f"refusing to stop {len(pids)} mihir-owned vLLM processes: {pids}")
    os.kill(pids[0], signal.SIGTERM)
    print(json.dumps({"status": "stop_requested", "pid": pids[0]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve", "status", "stop"))
    args = parser.parse_args()
    {"serve": serve, "status": status, "stop": stop}[args.command]()


if __name__ == "__main__":
    main()
