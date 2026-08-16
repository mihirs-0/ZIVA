"""Conservative endpoint parser and execution for the concise percentage rerun."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from .chat_percentage import ChatPercentageTrial, LocalVLLMChatClient

_NUMBER = r"(?:100(?:\.0+)?|(?:\d{1,2})(?:\.\d+)?)"
_UNIT = r"(?:%|percent(?:age)?)"
_RANGE_RE = re.compile(
    rf"(?<![\d.])(?P<low>{_NUMBER})\s*(?:%\s*)?(?:-|–|—|to)\s*"
    rf"(?P<high>{_NUMBER})\s*{_UNIT}(?!\w)",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(rf"(?<![\d.])(?P<value>{_NUMBER})\s*{_UNIT}(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class ConcisePercentageParse:
    status: str
    value: float | None
    original_range: list[float] | None
    matched_text: str | None
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_concise_percentage(text: str) -> ConcisePercentageParse:
    """Parse only explicit percent scalars or hyphen/``to`` percent ranges.

    Duplicate occurrences of the same scalar are compatible. Any combination
    containing distinct scalars, multiple ranges, or a range plus a scalar is
    rejected. No semantic or context-dependent inference is performed.
    """
    if not text or not text.strip():
        return ConcisePercentageParse("empty", None, None, None, ["empty response"])

    ranges: list[tuple[float, float, str, tuple[int, int]]] = []
    for match in _RANGE_RE.finditer(text):
        low, high = float(match.group("low")), float(match.group("high"))
        if low > high:
            low, high = high, low
        ranges.append((low, high, match.group(0), match.span()))

    covered = [row[3] for row in ranges]
    singles: list[tuple[float, str]] = []
    for match in _SINGLE_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in covered):
            continue
        singles.append((float(match.group("value")), match.group(0)))

    if len(ranges) == 1 and not singles:
        low, high, matched, _ = ranges[0]
        return ConcisePercentageParse("ok_range", (low + high) / 2, [low, high], matched, [])
    if len(ranges) > 1 or (ranges and singles):
        return ConcisePercentageParse(
            "ambiguous", None, None, None, ["multiple incompatible percentage expressions"]
        )
    if not singles:
        return ConcisePercentageParse("no_percentage", None, None, None, ["no explicit percentage found"])

    values = {value for value, _ in singles}
    if len(values) == 1:
        value = next(iter(values))
        return ConcisePercentageParse("ok_single", value, None, singles[0][1], [])
    return ConcisePercentageParse(
        "ambiguous", None, None, None, ["multiple incompatible percentage expressions"]
    )


def choose_token_budget(
    summaries: list[dict[str, Any]], minimum_parse_rate: float, maximum_truncation_rate: float
) -> int | None:
    """Choose the smallest candidate satisfying the preregistered neutral gates."""
    eligible = [
        row["max_tokens"]
        for row in summaries
        if row["parse_success_rate"] >= minimum_parse_rate
        and row["truncation_rate"] <= maximum_truncation_rate
        and row["anchor_order_sensible"]
        and row["no_think_content"]
    ]
    return min(eligible) if eligible else None


def execute_concise_trial(
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
            parsed = parse_concise_percentage(turn_2["text"])
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
        # Preserve any client or response-shape failure for resume-safe retrying.
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
        "percentage_parse": ConcisePercentageParse(
            "execution_error", None, None, None, [repr(last_error)]
        ).to_dict(),
    }
