#!/usr/bin/env python3
"""Resolve and sequentially serve the fixed OSS model queue on mihir's idli account."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "configs/oss_model_queue.yaml"
REGISTRY_DIR = ROOT / "data/manifests/oss_model_registry"


def _queue() -> dict[str, Any]:
    return yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))


def _spec(key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = _queue()
    matches = [row for row in queue["models"] if row["key"] == key]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate model key {key!r}")
    return matches[0], queue["runtime"]


def _registry_path(key: str) -> Path:
    return REGISTRY_DIR / f"{key}.json"


def resolve(key: str) -> None:
    spec, _ = _spec(key)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    required = spec.get("required_revision")
    if required:
        revision = required
        source = "fixed queue revision inherited from the banked Qwen baseline"
    else:
        from huggingface_hub import HfApi

        info = HfApi().model_info(spec["checkpoint"])
        revision = info.sha
        source = "Hugging Face model_info resolved from repository default revision"
    if not revision or len(revision) < 20:
        raise RuntimeError(f"did not resolve an immutable revision for {spec['checkpoint']}")
    record = {
        "status": "RESOLVED",
        "key": key,
        "checkpoint": spec["checkpoint"],
        "revision": revision,
        "tokenizer_revision": revision,
        "parameter_scale": spec["parameter_scale"],
        "model_family": spec["model_family"],
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "resolution_source": source,
    }
    write_json(_registry_path(key), record)
    print(json.dumps(record, indent=2))


def _endpoint_model() -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
            payload = json.loads(response.read())
        return payload["data"][0]["id"] if payload.get("data") else None
    except (OSError, KeyError, json.JSONDecodeError, urllib.error.URLError):
        return None


def _owned_vllm_pids() -> list[int]:
    result = subprocess.run(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,cmd="],
        capture_output=True,
        text=True,
        check=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if "vllm serve" in command:
            pids.append(int(pid_text))
    return pids


def _versions() -> dict[str, str]:
    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "vllm": version("vllm"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
    }


def _gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, name, uuid, driver, total, used = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            }
        )
    return rows


def status(key: str) -> None:
    spec, _ = _spec(key)
    print(
        json.dumps(
            {
                "expected_model": spec["checkpoint"],
                "endpoint_model": _endpoint_model(),
                "owned_vllm_pids": _owned_vllm_pids(),
                "gpus": _gpu_snapshot(),
            },
            indent=2,
        )
    )


def stop(key: str) -> None:
    spec, _ = _spec(key)
    endpoint = _endpoint_model()
    if endpoint is not None and endpoint != spec["checkpoint"]:
        raise RuntimeError(f"refusing to stop endpoint {endpoint!r}; expected {spec['checkpoint']!r}")
    pids = _owned_vllm_pids()
    if not pids:
        print(json.dumps({"status": "already_stopped", "key": key}, indent=2))
        return
    if len(pids) != 1:
        raise RuntimeError(f"refusing to stop {len(pids)} mihir-owned vLLM processes: {pids}")
    pid = pids[0]
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(json.dumps({"status": "stopped", "key": key, "pid": pid}, indent=2))
            return
        time.sleep(2)
    raise RuntimeError(f"vLLM pid {pid} did not stop after SIGTERM; manual review required")


def serve(key: str, max_model_len: int | None, gpu_memory_utilization: float | None) -> None:
    spec, runtime = _spec(key)
    registry = json.loads(_registry_path(key).read_text())
    if registry.get("status") != "RESOLVED":
        raise RuntimeError(f"model {key} has no immutable resolved revision")
    current = _endpoint_model()
    if current == spec["checkpoint"]:
        print(json.dumps({"status": "already_serving", "key": key, "model": current}, indent=2))
        return
    existing = _owned_vllm_pids()
    if existing:
        raise RuntimeError(f"another mihir-owned vLLM process is still present: {existing}")

    context = max_model_len or runtime["max_model_len"]
    memory = gpu_memory_utilization or runtime["gpu_memory_utilization"]
    log_path = REGISTRY_DIR / f"{key}_vllm.log"
    command = [
        "/usr/local/bin/vllm",
        "serve",
        spec["checkpoint"],
        "--revision",
        registry["revision"],
        "--served-model-name",
        spec["checkpoint"],
        "--dtype",
        runtime["dtype"],
        "--tensor-parallel-size",
        str(runtime["tensor_parallel_size"]),
        "--max-model-len",
        str(context),
        "--gpu-memory-utilization",
        str(memory),
        "--generation-config",
        runtime["generation_config_source"],
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(UTC).isoformat()}] command={json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    service = {
        "status": "STARTING",
        "key": key,
        "checkpoint": spec["checkpoint"],
        "revision": registry["revision"],
        "pid": process.pid,
        "command": command,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "max_model_len": context,
        "gpu_memory_utilization": memory,
        "versions": _versions(),
        "log_path": str(log_path.relative_to(ROOT)),
    }
    service_path = REGISTRY_DIR / f"{key}_service.json"
    write_json(service_path, service)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            service["status"] = "FAILED_STARTUP"
            service["exit_code"] = process.returncode
            service["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(service_path, service)
            raise RuntimeError(f"vLLM exited with code {process.returncode}; see {log_path}")
        if _endpoint_model() == spec["checkpoint"]:
            service["status"] = "READY"
            service["ready_at_utc"] = datetime.now(UTC).isoformat()
            service["gpus"] = _gpu_snapshot()
            write_json(service_path, service)
            print(json.dumps(service, indent=2))
            return
        time.sleep(5)
    service["status"] = "STARTUP_TIMEOUT"
    write_json(service_path, service)
    raise RuntimeError(f"vLLM did not become ready within 30 minutes; see {log_path}")


def failure(key: str, stage: str, message: str) -> None:
    spec, _ = _spec(key)
    record = {
        "status": "SKIPPED_ENGINEERING",
        "key": key,
        "checkpoint": spec["checkpoint"],
        "stage": stage,
        "message": message,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REGISTRY_DIR / f"{key}_failure.json", record)
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("resolve", "serve", "status", "stop", "failure"))
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--stage")
    parser.add_argument("--message")
    args = parser.parse_args()
    if args.command == "resolve":
        resolve(args.model_key)
    elif args.command == "serve":
        serve(args.model_key, args.max_model_len, args.gpu_memory_utilization)
    elif args.command == "status":
        status(args.model_key)
    elif args.command == "stop":
        stop(args.model_key)
    else:
        failure(args.model_key, args.stage or "unknown", args.message or "unspecified failure")


if __name__ == "__main__":
    main()
