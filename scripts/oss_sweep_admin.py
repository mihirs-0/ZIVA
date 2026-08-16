#!/usr/bin/env python3
"""Preservation and integrity utilities for the autonomous OSS sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ziva.chat_eval import sha256_file
from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/manifests/oss_overnight/preservation.json"
PREFIXES = (
    "configs/pilot_qwen3_14b*",
    "configs/models.qwen3_14b.yaml",
    "data/manifests/pilot_qwen3_14b*",
    "data/raw/pilot_qwen3_14b*",
    "data/scenarios/pilot_qwen3_14b*",
    "data/stimuli/pilot_qwen3_14b*",
    "results/pilot_qwen3_14b*",
    "reports/pilot_qwen3_14b*",
)


def _files() -> list[Path]:
    selected: set[Path] = set()
    for pattern in PREFIXES:
        for path in ROOT.glob(pattern):
            if path.is_file():
                selected.add(path)
            elif path.is_dir():
                selected.update(child for child in path.rglob("*") if child.is_file())
    return sorted(selected)


def record() -> None:
    files = _files()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).splitlines()
    payload = {
        "purpose": "Pre-sweep preservation hashes for every banked Qwen structured/chat/logprob artifact namespace.",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "base_commit": "4e36edd0c7f855b003c87ffcc862b1e108b15c63",
        "tracked_changes_observed_before_branch": [],
        "tracked_changes_at_record_time": status,
        "prefixes": list(PREFIXES),
        "file_count": len(files),
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
    }
    write_json(OUT, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, indent=2))


def verify() -> None:
    payload = json.loads(OUT.read_text())
    missing = []
    changed = []
    for relative, expected in payload["files"].items():
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
        elif sha256_file(path) != expected:
            changed.append(relative)
    current = {str(path.relative_to(ROOT)) for path in _files()}
    recorded = set(payload["files"])
    result = {
        "ok": not missing and not changed,
        "recorded_files": len(recorded),
        "current_files": len(current),
        "missing": missing,
        "changed": changed,
        "new_files_under_preserved_prefixes": sorted(current - recorded),
    }
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "verify"))
    args = parser.parse_args()
    {"record": record, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
