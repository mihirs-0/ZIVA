"""Frozen ZIVA-Chat-v1 analysis and structured-protocol comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .chat_percentage_analysis import analyze_percentage_records, render_percentage_report
from .stats import paired_summary


def _paired(values: pd.Series, config: dict[str, Any], offset: int) -> dict[str, Any]:
    return paired_summary(
        values.dropna().to_numpy(dtype=float),
        n_boot=config["analysis"]["bootstrap_samples"],
        n_perm=config["analysis"]["permutation_samples"],
        seed=config["analysis"]["seed"] + offset,
    )


def _malformed_by(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, group in frame.groupby(column):
        malformed = int((~group["valid"]).sum())
        output[str(name)] = {
            "n": len(group),
            "malformed": malformed,
            "malformed_rate": round(malformed / len(group), 6),
        }
    return output


def _structured_comparisons(
    scenario: pd.DataFrame, paired_path: str | Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    structured = pd.read_csv(paired_path)
    by_scenario = (
        structured.groupby("scenario_id", as_index=False)[
            ["excited_minus_neutral", "skeptical_minus_neutral"]
        ]
        .mean()
        .rename(
            columns={
                "excited_minus_neutral": "structured_pooled_excited",
                "skeptical_minus_neutral": "structured_pooled_skeptical",
            }
        )
    )
    naturalistic = structured[structured["elicitation_mode"] == "naturalistic"][
        ["scenario_id", "excited_minus_neutral", "skeptical_minus_neutral"]
    ].rename(
        columns={
            "excited_minus_neutral": "structured_naturalistic_excited",
            "skeptical_minus_neutral": "structured_naturalistic_skeptical",
        }
    )
    merged = scenario.merge(by_scenario, on="scenario_id", how="inner").merge(
        naturalistic, on="scenario_id", how="inner"
    )
    definitions = {
        "excited_chat_minus_structured_pooled": (
            "excited_positive_minus_neutral",
            "structured_pooled_excited",
        ),
        "excited_chat_minus_structured_naturalistic": (
            "excited_positive_minus_neutral",
            "structured_naturalistic_excited",
        ),
        "skeptical_chat_minus_structured_pooled": (
            "skeptical_negative_minus_neutral",
            "structured_pooled_skeptical",
        ),
        "skeptical_chat_minus_structured_naturalistic": (
            "skeptical_negative_minus_neutral",
            "structured_naturalistic_skeptical",
        ),
    }
    summaries: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for offset, (name, (chat_col, structured_col)) in enumerate(definitions.items(), start=300):
        frame = merged[["scenario_id", chat_col, structured_col]].copy()
        frame["chat_minus_structured"] = frame[chat_col] - frame[structured_col]
        result = _paired(frame["chat_minus_structured"], config, offset)
        result["chat_mean_effect"] = round(float(frame[chat_col].mean()), 4)
        result["structured_mean_effect"] = round(float(frame[structured_col].mean()), 4)
        summaries[name] = result
        frames[name] = frame
    return summaries, frames


def analyze_chat_v1_records(
    raw_dir: str | Path, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None]:
    summary, scenario, _ = analyze_percentage_records(raw_dir, config)
    records = [json.loads(path.read_text()) for path in sorted(Path(raw_dir).glob("tcp_*.json"))]
    reliability = pd.DataFrame(
        [
            {
                "treatment_family": row["treatment_family"],
                "treatment_id": row["treatment_id"],
                "valid": row["percentage_parse"]["value"] is not None,
                "turn_1_truncated": row["turn_1"]["finish_reason"] == "length",
                "turn_2_truncated": row["turn_2"]["finish_reason"] == "length",
            }
            for row in records
        ]
    )
    summary["reliability"] = {
        "turn_1_truncations": int(reliability["turn_1_truncated"].sum()),
        "turn_1_truncation_rate": round(float(reliability["turn_1_truncated"].mean()), 6),
        "turn_2_truncations": int(reliability["turn_2_truncated"].sum()),
        "turn_2_truncation_rate": round(float(reliability["turn_2_truncated"].mean()), 6),
        "malformed_by_family": _malformed_by(reliability, "treatment_family"),
        "malformed_by_template": _malformed_by(reliability, "treatment_id"),
    }
    comparisons, comparison_frames = _structured_comparisons(
        scenario, config["structured_paired_table"], config
    )
    summary["structured_comparisons"] = comparisons

    previous_frame: pd.DataFrame | None = None
    previous_path = config.get("previous_chat_scenario_effects")
    if previous_path and Path(previous_path).exists():
        previous = pd.read_csv(previous_path)[
            ["scenario_id", "excited_positive_minus_neutral"]
        ].rename(columns={"excited_positive_minus_neutral": "previous_chat_excited_effect"})
        previous_frame = scenario[
            ["scenario_id", "excited_positive_minus_neutral"]
        ].merge(previous, on="scenario_id", how="inner")
        previous_frame["chat_v1_minus_previous_chat"] = (
            previous_frame["excited_positive_minus_neutral"]
            - previous_frame["previous_chat_excited_effect"]
        )
        result = _paired(previous_frame["chat_v1_minus_previous_chat"], config, 500)
        result["chat_v1_mean_effect"] = round(
            float(previous_frame["excited_positive_minus_neutral"].mean()), 4
        )
        result["previous_chat_mean_effect"] = round(
            float(previous_frame["previous_chat_excited_effect"].mean()), 4
        )
        summary["previous_chat_comparison"] = result
    return summary, scenario, comparison_frames, previous_frame


def _fmt(result: dict[str, Any]) -> str:
    lo, hi = result["ci95"]
    return (
        f"{result['mean']:+.2f} points (95% CI [{lo:+.2f}, {hi:+.2f}], "
        f"p={result['p_perm']}, d_z={result['cohens_dz']})"
    )


def render_chat_v1_report(summary: dict[str, Any]) -> str:
    base = render_percentage_report(summary)
    head, boundary = base.split("## Direct comparison with structured Qwen", maxsplit=1)
    _discard, boundary = boundary.split("## Interpretation boundary", maxsplit=1)
    reliability = summary["reliability"]
    comparisons = summary["structured_comparisons"]
    lines = [head.rstrip(), "", "## Reliability", ""]
    lines.extend(
        [
            (
                f"- Turn-1 truncation: {reliability['turn_1_truncations']}/{summary['n_trials']} "
                f"({reliability['turn_1_truncation_rate']:.2%})"
            ),
            (
                f"- Turn-2 truncation: {reliability['turn_2_truncations']}/{summary['n_trials']} "
                f"({reliability['turn_2_truncation_rate']:.2%})"
            ),
            f"- Malformed percentage rate: {summary['malformed_rate']:.2%}",
            "- Malformed by family: "
            + ", ".join(
                f"{name}={row['malformed_rate']:.2%}"
                for name, row in reliability["malformed_by_family"].items()
            ),
            "- Malformed by template: "
            + ", ".join(
                f"{name}={row['malformed_rate']:.2%}"
                for name, row in reliability["malformed_by_template"].items()
            ),
            "",
            "## Structured versus Chat-v1",
            "",
        ]
    )
    for name, result in comparisons.items():
        lines.append(f"- {name}: {_fmt(result)}")
    if "previous_chat_comparison" in summary:
        lines.extend(
            [
                "",
                "## Previous chat engineering-run comparison",
                "",
                (
                    "- Chat-v1 minus previous chat excited effect: "
                    f"{_fmt(summary['previous_chat_comparison'])}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary" + boundary,
        ]
    )
    return "\n".join(lines)
