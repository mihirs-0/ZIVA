"""Scenario-level analysis for the two-turn conversational percentage experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .stats import paired_summary, pearson_r

FAMILIES = ("excited_positive", "skeptical_negative", "negative_preference")


def _paired(values: pd.Series, config: dict[str, Any], offset: int = 0) -> dict[str, Any]:
    return paired_summary(
        values.to_numpy(dtype=float),
        n_boot=config["analysis"]["bootstrap_samples"],
        n_perm=config["analysis"]["permutation_samples"],
        seed=config["analysis"]["seed"] + offset,
    )


def analyze_percentage_records(
    raw_dir: str | Path, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    records = [json.loads(path.read_text()) for path in sorted(Path(raw_dir).glob("tcp_*.json"))]
    rows = []
    for record in records:
        parsed = record["percentage_parse"]
        rows.append(
            {
                "trial_id": record["trial_id"],
                "scenario_id": record["scenario_id"],
                "treatment_id": record["treatment_id"],
                "treatment_family": record["treatment_family"],
                "template_index": int(record["treatment_id"].rsplit("_v", 1)[1]),
                "repeat_index": record["repeat_index"],
                "difficulty_class": record["difficulty_class"],
                "difficulty_score": record["difficulty_score"],
                "parse_status": parsed["status"],
                "visible_probability_chat": parsed["value"],
            }
        )
    trials = pd.DataFrame(rows)
    valid = trials.dropna(subset=["visible_probability_chat"]).copy()
    exact = valid.groupby(
        ["scenario_id", "treatment_id", "treatment_family", "difficulty_class", "difficulty_score"]
    )["visible_probability_chat"]
    sampling_sd = exact.std(ddof=1).dropna()
    template_means = exact.mean().reset_index()
    template_sd = (
        template_means.groupby(["scenario_id", "treatment_family", "difficulty_class", "difficulty_score"])[
            "visible_probability_chat"
        ]
        .std(ddof=1)
        .dropna()
    )

    family = valid.groupby(
        ["scenario_id", "treatment_family", "difficulty_class", "difficulty_score"],
        as_index=False,
    )["visible_probability_chat"].mean()
    wide = family.pivot(index="scenario_id", columns="treatment_family", values="visible_probability_chat")
    metadata = family.drop_duplicates("scenario_id").set_index("scenario_id")
    scenario = metadata[["difficulty_class", "difficulty_score"]].copy()
    for fam in FAMILIES:
        scenario[f"{fam}_minus_neutral"] = wide[fam] - wide["neutral"]
    scenario["negative_preference_direction_normalized"] = wide["neutral"] - wide["negative_preference"]

    contrasts: dict[str, Any] = {}
    for idx, fam in enumerate(FAMILIES):
        name = f"{fam}_minus_neutral"
        contrasts[name] = _paired(scenario[name], config, idx)
        contrasts[name]["excluding_case_000"] = _paired(
            scenario.loc[~scenario.index.str.startswith("case_000"), name], config, idx + 10
        )
    contrasts["negative_preference_direction_normalized"] = _paired(
        scenario["negative_preference_direction_normalized"], config, 20
    )

    neutral = family[family["treatment_family"] == "neutral"].set_index("scenario_id")[
        "visible_probability_chat"
    ]
    template_effects = {}
    for idx, (treatment_id, group) in enumerate(
        template_means[template_means["treatment_family"] != "neutral"].groupby("treatment_id")
    ):
        effect = group.set_index("scenario_id")["visible_probability_chat"] - neutral
        template_effects[treatment_id] = _paired(effect, config, 100 + idx)

    difficulty = {}
    excited = scenario["excited_positive_minus_neutral"]
    for idx, (label, indexes) in enumerate(scenario.groupby("difficulty_class").groups.items()):
        difficulty[str(label)] = _paired(excited.loc[indexes], config, 200 + idx)
    difficulty["pearson_r_with_difficulty_score"] = pearson_r(
        scenario["difficulty_score"].to_numpy(), excited.to_numpy()
    )

    paired_prompt = valid.pivot_table(
        index=["scenario_id", "template_index", "repeat_index"],
        columns="treatment_family",
        values="visible_probability_chat",
        aggfunc="first",
    )
    threshold = config["analysis"]["binary_threshold"]
    crossings = {}
    for fam in FAMILIES:
        pair = paired_prompt.dropna(subset=["neutral", fam])
        crossed = ((pair["neutral"] < threshold) & (pair[fam] >= threshold)) | (
            (pair["neutral"] >= threshold) & (pair[fam] < threshold)
        )
        crossings[f"{fam}_vs_neutral"] = {
            "n_pairs": len(pair),
            "crossing_rate": round(float(crossed.mean()), 4) if len(pair) else None,
        }

    structured = pd.read_csv(Path(config["structured_scenario_effects"]))
    direct = scenario.reset_index()[["scenario_id", "excited_positive_minus_neutral"]].merge(
        structured[["scenario_id", "excited_minus_neutral"]], on="scenario_id", how="inner"
    )
    direct["chat_minus_structured_effect"] = (
        direct["excited_positive_minus_neutral"] - direct["excited_minus_neutral"]
    )
    direct_summary = _paired(direct["chat_minus_structured_effect"], config, 300)
    direct_summary["chat_mean_effect"] = round(float(direct["excited_positive_minus_neutral"].mean()), 4)
    direct_summary["structured_mean_effect"] = round(float(direct["excited_minus_neutral"].mean()), 4)

    malformed = int(trials["visible_probability_chat"].isna().sum())
    summary = {
        "experiment_name": config["experiment_name"],
        "n_trials": len(trials),
        "n_scenarios": int(trials["scenario_id"].nunique()),
        "valid_percentages": int(len(trials) - malformed),
        "malformed_percentages": malformed,
        "malformed_rate": round(malformed / len(trials), 6),
        "parse_status_counts": trials["parse_status"].value_counts().to_dict(),
        "contrasts": contrasts,
        "difficulty": difficulty,
        "template_effects": template_effects,
        "binary_threshold_crossings": crossings,
        "variance": {
            "repeat_sampling_sd_mean_points": round(float(sampling_sd.mean()), 4),
            "repeat_sampling_sd_n_cells": len(sampling_sd),
            "template_sd_mean_points": round(float(template_sd.mean()), 4),
            "template_sd_n_cells": len(template_sd),
        },
        "direct_chat_minus_structured": direct_summary,
        "endpoint": "Turn-2 explicitly stated rough percentage; range midpoint when one clear range is given.",
    }
    return summary, scenario.reset_index(), direct


def _fmt(result: dict[str, Any]) -> str:
    lo, hi = result["ci95"]
    return (
        f"{result['mean']:+.2f} points (95% CI [{lo:+.2f}, {hi:+.2f}], "
        f"p={result['p_perm']}, d_z={result['cohens_dz']})"
    )


def render_percentage_report(summary: dict[str, Any]) -> str:
    contrasts = summary["contrasts"]
    lines = [
        f"# ZIVA conversational-percentage report: {summary['experiment_name']}",
        "",
        "## Design and completeness",
        "",
        f"- Trials: {summary['n_trials']}; scenarios: {summary['n_scenarios']}",
        f"- Valid percentages: {summary['valid_percentages']}; malformed: {summary['malformed_percentages']} ({summary['malformed_rate']:.2%})",
        "- Primary endpoint: the explicit rough percentage in the second ordinary conversational turn.",
        "",
        "## Results",
        "",
        f"- Excited-positive minus neutral: {_fmt(contrasts['excited_positive_minus_neutral'])}",
        f"- Skeptical-negative minus neutral: {_fmt(contrasts['skeptical_negative_minus_neutral'])}",
        f"- Negative-preference minus neutral: {_fmt(contrasts['negative_preference_minus_neutral'])}",
        f"- Negative-preference direction-normalized: {_fmt(contrasts['negative_preference_direction_normalized'])}",
        "",
        "## Sensitivity excluding case_000",
        "",
    ]
    for name in (
        "excited_positive_minus_neutral",
        "skeptical_negative_minus_neutral",
        "negative_preference_minus_neutral",
    ):
        lines.append(f"- {name}: {_fmt(contrasts[name]['excluding_case_000'])}")
    lines.extend(["", "## Effect by difficulty", ""])
    for label, result in summary["difficulty"].items():
        if label != "pearson_r_with_difficulty_score":
            lines.append(f"- {label}: {_fmt(result)}")
    lines.append(f"- Pearson r with difficulty: {summary['difficulty']['pearson_r_with_difficulty_score']}")
    lines.extend(["", "## Per-template effects", ""])
    for treatment_id, result in sorted(summary["template_effects"].items()):
        lines.append(f"- {treatment_id}: {_fmt(result)}")
    variance = summary["variance"]
    lines.extend(
        [
            "",
            "## Variance and binary crossings",
            "",
            f"- Repeat-sampling SD: {variance['repeat_sampling_sd_mean_points']:.2f} points",
            f"- Template/paraphrase SD: {variance['template_sd_mean_points']:.2f} points",
        ]
    )
    for name, record in summary["binary_threshold_crossings"].items():
        lines.append(f"- {name} 50% crossing rate: {record['crossing_rate']:.2%} (n={record['n_pairs']})")
    direct = summary["direct_chat_minus_structured"]
    lines.extend(
        [
            "",
            "## Direct comparison with structured Qwen",
            "",
            f"- Chat excited effect: {direct['chat_mean_effect']:+.2f} points",
            f"- Structured excited effect: {direct['structured_mean_effect']:+.2f} points",
            f"- Chat minus structured scenario-level effect: {_fmt(direct)}",
            "",
            "## Interpretation boundary",
            "",
            "This is a behavioral comparison of stated percentages under different conversational contexts. It does not establish internal beliefs, deception, conscious evaluation awareness, or scheming.",
            "",
        ]
    )
    return "\n".join(lines)
