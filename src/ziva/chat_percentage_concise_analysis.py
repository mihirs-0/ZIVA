"""Reliability additions for the concise conversational-percentage rerun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .chat_percentage_analysis import analyze_percentage_records, render_percentage_report
from .stats import paired_summary


def _paired(values: pd.Series, config: dict[str, Any], offset: int) -> dict[str, Any]:
    return paired_summary(
        values.to_numpy(dtype=float),
        n_boot=config["analysis"]["bootstrap_samples"],
        n_perm=config["analysis"]["permutation_samples"],
        seed=config["analysis"]["seed"] + offset,
    )


def analyze_concise_records(
    raw_dir: str | Path, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, scenario, structured_direct = analyze_percentage_records(raw_dir, config)
    records = [json.loads(path.read_text()) for path in sorted(Path(raw_dir).glob("tcp_*.json"))]
    reliability_rows = []
    for record in records:
        valid = record["percentage_parse"]["value"] is not None
        reliability_rows.append(
            {
                "treatment_family": record["treatment_family"],
                "treatment_id": record["treatment_id"],
                "valid": valid,
                "turn_1_truncated": record["turn_1"]["finish_reason"] == "length",
                "turn_2_truncated": record["turn_2"]["finish_reason"] == "length",
            }
        )
    reliability = pd.DataFrame(reliability_rows)

    def malformed_by(column: str) -> dict[str, Any]:
        output = {}
        for name, group in reliability.groupby(column):
            malformed = int((~group["valid"]).sum())
            output[str(name)] = {
                "n": len(group),
                "malformed": malformed,
                "malformed_rate": round(malformed / len(group), 6),
            }
        return output

    previous_rate = config["analysis"]["previous_malformed_rate"]
    summary["reliability"] = {
        "turn_1_truncations": int(reliability["turn_1_truncated"].sum()),
        "turn_1_truncation_rate": round(float(reliability["turn_1_truncated"].mean()), 6),
        "turn_2_truncations": int(reliability["turn_2_truncated"].sum()),
        "turn_2_truncation_rate": round(float(reliability["turn_2_truncated"].mean()), 6),
        "malformed_by_family": malformed_by("treatment_family"),
        "malformed_by_template": malformed_by("treatment_id"),
        "previous_chat_malformed_rate": previous_rate,
        "malformed_rate_change_points": round((summary["malformed_rate"] - previous_rate) * 100, 4),
        "relative_malformed_reduction": round((previous_rate - summary["malformed_rate"]) / previous_rate, 6),
    }

    previous = pd.read_csv(Path(config["previous_chat_scenario_effects"]))[
        ["scenario_id", "excited_positive_minus_neutral"]
    ].rename(columns={"excited_positive_minus_neutral": "previous_chat_excited_effect"})
    previous_direct = scenario[["scenario_id", "excited_positive_minus_neutral"]].merge(
        previous, on="scenario_id", how="inner"
    )
    previous_direct["concise_minus_previous_chat"] = (
        previous_direct["excited_positive_minus_neutral"] - previous_direct["previous_chat_excited_effect"]
    )
    comparison = _paired(previous_direct["concise_minus_previous_chat"], config, 400)
    comparison["concise_mean_effect"] = round(
        float(previous_direct["excited_positive_minus_neutral"].mean()), 4
    )
    comparison["previous_chat_mean_effect"] = round(
        float(previous_direct["previous_chat_excited_effect"].mean()), 4
    )
    summary["direct_concise_minus_previous_chat"] = comparison
    return summary, scenario, structured_direct, previous_direct


def render_concise_report(summary: dict[str, Any]) -> str:
    base = render_percentage_report(summary)
    head, boundary = base.split("## Interpretation boundary", maxsplit=1)
    reliability = summary["reliability"]
    comparison = summary["direct_concise_minus_previous_chat"]
    truncation_line = (
        f"- Turn-2 truncation: {reliability['turn_2_truncations']}/{summary['n_trials']} "
        f"({reliability['turn_2_truncation_rate']:.2%})"
    )
    malformed_line = (
        f"- Malformed percentage rate: {summary['malformed_rate']:.2%}; previous run: "
        f"{reliability['previous_chat_malformed_rate']:.2%}"
    )
    comparison_line = (
        f"- Concise minus previous: {comparison['mean']:+.2f} points "
        f"(95% CI [{comparison['ci95'][0]:+.2f}, {comparison['ci95'][1]:+.2f}], "
        f"p={comparison['p_perm']}, d_z={comparison['cohens_dz']})"
    )
    lines = [
        head.rstrip(),
        "",
        "## Reliability rerun",
        "",
        truncation_line,
        malformed_line,
        f"- Malformed-rate change: {reliability['malformed_rate_change_points']:+.2f} percentage points",
        "- Malformed by family: "
        + ", ".join(
            f"{name}={row['malformed_rate']:.2%}" for name, row in reliability["malformed_by_family"].items()
        ),
        "- Malformed by template: "
        + ", ".join(
            f"{name}={row['malformed_rate']:.2%}"
            for name, row in reliability["malformed_by_template"].items()
        ),
        "",
        "## Direct comparison with the previous chat run",
        "",
        f"- Concise excited effect: {comparison['concise_mean_effect']:+.2f} points",
        f"- Previous chat excited effect: {comparison['previous_chat_mean_effect']:+.2f} points",
        comparison_line,
        "",
        "## Interpretation boundary" + boundary,
    ]
    return "\n".join(lines)
