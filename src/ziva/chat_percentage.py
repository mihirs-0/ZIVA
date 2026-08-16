"""Two-turn, JSON-less conversational percentage experiment.

The existing structured benchmark and hidden-logprob experiment are not used
as execution paths here.  This module creates ordinary chats, requests one
rough percentage in a second turn, and parses that percentage conservatively.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .chat_eval import ChatPrompt, compile_chat_prompt
from .treatments import select_treatments
from .util import sha256_json, sha256_text

PROHIBITED_TURN1_CUES = (
    "json",
    "benchmark",
    "evaluation",
    "logging",
    "calibration",
    "probability",
    "confidence",
    "evidence sufficiency",
    "scoring",
    "experiment",
    "measurement",
)

_NUMBER = r"(?:100(?:\.0+)?|(?:\d{1,2})(?:\.\d+)?)"
_UNIT = r"(?:%|percent(?:age)?)"
_RANGE_RE = re.compile(
    rf"(?<![\d.])(?P<low>{_NUMBER})\s*(?:%\s*)?(?:-|–|—|to)\s*"
    rf"(?P<high>{_NUMBER})\s*{_UNIT}(?!\w)",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    rf"\bbetween\s+(?P<low>{_NUMBER})\s*(?:%\s*)?and\s+"
    rf"(?P<high>{_NUMBER})\s*{_UNIT}(?!\w)",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(rf"(?<![\d.])(?P<value>{_NUMBER})\s*{_UNIT}(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class PercentageParse:
    status: str
    value: float | None
    original_range: list[float] | None
    matched_text: str | None
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChatPercentageTrial:
    trial_id: str
    order_index: int
    repeat_index: int
    turn_1_seed: int
    turn_2_seed: int
    system: str
    turn_1_user: str
    turn_2_user: str
    scenario_id: str
    treatment_id: str
    treatment_family: str
    preferred_outcome: str | None
    expected_outcome: str | None
    treatment_text: str
    invariant_text: str
    invariant_hash: str
    prompt_hash: str
    difficulty_class: str
    difficulty_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_percentage_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_percentage(text: str) -> PercentageParse:
    """Extract one explicit percentage or one explicit range; never infer semantics."""
    if not text or not text.strip():
        return PercentageParse("empty", None, None, None, ["empty response"])

    ranges: list[tuple[float, float, str, tuple[int, int]]] = []
    for pattern in (_BETWEEN_RE, _RANGE_RE):
        for match in pattern.finditer(text):
            low, high = float(match.group("low")), float(match.group("high"))
            if low > high:
                low, high = high, low
            ranges.append((low, high, match.group(0), match.span()))
    # Deduplicate overlapping range patterns, retaining their first occurrence.
    unique_ranges: list[tuple[float, float, str, tuple[int, int]]] = []
    for item in sorted(ranges, key=lambda row: row[3]):
        if not any(item[3][0] < other[3][1] and other[3][0] < item[3][1] for other in unique_ranges):
            unique_ranges.append(item)

    covered = [row[3] for row in unique_ranges]
    singles = []
    for match in _SINGLE_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in covered):
            continue
        singles.append((float(match.group("value")), match.group(0)))

    if len(unique_ranges) == 1 and not singles:
        low, high, matched, _ = unique_ranges[0]
        return PercentageParse("ok_range", (low + high) / 2, [low, high], matched, [])
    if len(unique_ranges) > 1 or (unique_ranges and singles):
        return PercentageParse(
            "ambiguous", None, None, None, ["multiple incompatible percentage expressions"]
        )
    if not singles:
        return PercentageParse("no_percentage", None, None, None, ["no explicit percentage found"])

    values = {value for value, _ in singles}
    if len(values) == 1:
        value = next(iter(values))
        matched = singles[0][1]
        return PercentageParse("ok_single", value, None, matched, [])
    return PercentageParse("ambiguous", None, None, None, ["multiple incompatible percentage expressions"])


def assert_low_cue_conversation(prompt: ChatPrompt, followup: str) -> None:
    turn_1 = (prompt.system + "\n" + prompt.user_text).lower()
    found = [phrase for phrase in PROHIBITED_TURN1_CUES if phrase in turn_1]
    if found:
        raise AssertionError(f"Turn-1 prompt contains prohibited cue(s): {found}")
    combined = turn_1 + "\n" + followup.lower()
    prohibited_any_turn = (
        "json",
        "benchmark",
        "evaluation",
        "logging",
        "calibration",
        "scoring",
        "experiment",
    )
    found_any = [phrase for phrase in prohibited_any_turn if phrase in combined]
    if found_any:
        raise AssertionError(f"conversation contains prohibited cue(s): {found_any}")
    if any(marker in combined for marker in ("{", "}", "```")):
        raise AssertionError("conversation contains structured-data delimiters")


def _seed(trial_id: str, turn: int) -> int:
    return int(hashlib.sha256(f"{trial_id}:turn:{turn}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def build_percentage_trials(
    config: dict[str, Any], scenarios: list[dict[str, Any]]
) -> list[ChatPercentageTrial]:
    treatments = select_treatments(
        config["treatments"]["families"], config["treatments"]["variants_per_family"]
    )
    rows: list[ChatPercentageTrial] = []
    followup = config["conversation"]["turn_2_followup"]
    for scenario in scenarios:
        for treatment in treatments:
            prompt = compile_chat_prompt(
                scenario,
                treatment,
                config["conversation"]["system_prompt"],
                config["conversation"]["turn_1_question"],
            )
            assert_low_cue_conversation(prompt, followup)
            for repeat in range(config["sampling"]["repeats"]):
                trial_id = (
                    "tcp_" + sha256_text(f"{scenario['scenario_id']}\x00{treatment.id}\x00{repeat}")[:16]
                )
                rows.append(
                    ChatPercentageTrial(
                        trial_id=trial_id,
                        order_index=-1,
                        repeat_index=repeat,
                        turn_1_seed=_seed(trial_id, 1),
                        turn_2_seed=_seed(trial_id, 2),
                        system=prompt.system,
                        turn_1_user=prompt.user_text,
                        turn_2_user=followup,
                        scenario_id=prompt.scenario_id,
                        treatment_id=prompt.treatment_id,
                        treatment_family=prompt.treatment_family,
                        preferred_outcome=prompt.preferred_outcome,
                        expected_outcome=prompt.expected_outcome,
                        treatment_text=prompt.treatment_text,
                        invariant_text=prompt.invariant_text,
                        invariant_hash=prompt.invariant_hash,
                        prompt_hash=prompt.prompt_hash,
                        difficulty_class=prompt.difficulty_class,
                        difficulty_score=prompt.difficulty_score,
                    )
                )
    random.Random(config["run"]["shuffle_seed"]).shuffle(rows)
    return [ChatPercentageTrial(**{**row.to_dict(), "order_index": i}) for i, row in enumerate(rows)]


def audit_percentage_pairing(trials: list[ChatPercentageTrial]) -> dict[str, Any]:
    by_scenario: dict[str, list[ChatPercentageTrial]] = {}
    for row in trials:
        by_scenario.setdefault(row.scenario_id, []).append(row)
    audits = []
    for scenario_id, rows in sorted(by_scenario.items()):
        invariants = {row.invariant_hash for row in rows}
        followups = {row.turn_2_user for row in rows}
        systems = {row.system for row in rows}
        ok = len(invariants) == len(followups) == len(systems) == 1
        audits.append(
            {
                "scenario_id": scenario_id,
                "ok": ok,
                "invariant_hashes": sorted(invariants),
                "turn_2_followups": sorted(followups),
                "systems": sorted(systems),
            }
        )
    failures = [row for row in audits if not row["ok"]]
    if failures:
        raise AssertionError(f"percentage-chat pairing audit failed: {failures[:3]}")
    return {"n_scenarios": len(audits), "n_trials": len(trials), "failures": 0, "rows": audits}


class LocalVLLMChatClient:
    def __init__(self, config: dict[str, Any]):
        endpoint = config["model"]["endpoint"].rstrip("/")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("percentage chat runner permits localhost vLLM only")
        self.endpoint = endpoint
        self.config = config

    def complete(self, messages: list[dict[str, str]], seed: int, max_tokens: int) -> dict[str, Any]:
        body = {
            "model": self.config["model"]["checkpoint"],
            "messages": messages,
            "temperature": self.config["sampling"]["temperature"],
            "top_p": self.config["sampling"]["top_p"],
            "top_k": self.config["sampling"]["top_k"],
            "min_p": self.config["sampling"]["min_p"],
            "presence_penalty": self.config["sampling"]["presence_penalty"],
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": self.config["model"]["enable_thinking"]},
        }
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        return {
            "text": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "model": payload.get("model"),
            "response_id": payload.get("id"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "latency_s": round(time.monotonic() - started, 3),
            "seed": seed,
            "request_messages_sha256": sha256_json(messages),
            "generation_settings": {
                "temperature": body["temperature"],
                "top_p": body["top_p"],
                "top_k": body["top_k"],
                "min_p": body["min_p"],
                "presence_penalty": body["presence_penalty"],
                "max_tokens": body["max_tokens"],
                "chat_template_kwargs": body["chat_template_kwargs"],
            },
        }


def execute_percentage_trial(
    trial: ChatPercentageTrial, config: dict[str, Any], fingerprint: str
) -> dict[str, Any]:
    client = LocalVLLMChatClient(config)
    messages_1 = [
        {"role": "system", "content": trial.system},
        {"role": "user", "content": trial.turn_1_user},
    ]
    retries = config["run"]["retries"]
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            turn_1 = client.complete(messages_1, trial.turn_1_seed, config["sampling"]["turn_1_max_tokens"])
            messages_2 = messages_1 + [
                {"role": "assistant", "content": turn_1["text"]},
                {"role": "user", "content": trial.turn_2_user},
            ]
            turn_2 = client.complete(messages_2, trial.turn_2_seed, config["sampling"]["turn_2_max_tokens"])
            parsed = parse_percentage(turn_2["text"])
            return {
                **trial.to_dict(),
                "experiment": config["experiment_name"],
                "experiment_fingerprint": fingerprint,
                "attempts": attempt,
                "status": "ok",
                "transcript": [
                    *messages_1,
                    {"role": "assistant", "content": turn_1["text"]},
                    {"role": "user", "content": trial.turn_2_user},
                    {"role": "assistant", "content": turn_2["text"]},
                ],
                "turn_1": turn_1,
                "turn_2": turn_2,
                "percentage_parse": parsed.to_dict(),
            }
        # The trial boundary must preserve any client, decoding, or response-shape
        # failure as an execution record so a resumed run can retry it cleanly.
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(attempt)
    return {
        **trial.to_dict(),
        "experiment": config["experiment_name"],
        "experiment_fingerprint": fingerprint,
        "attempts": retries,
        "status": "error",
        "error": repr(last_error),
        "percentage_parse": PercentageParse(
            "execution_error", None, None, None, [repr(last_error)]
        ).to_dict(),
    }
