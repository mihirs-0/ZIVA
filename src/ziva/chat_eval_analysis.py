"""Scenario-level analysis for the isolated JSON-less contrastive experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stats import paired_summary, pearson_r

FAMILY_IDS = ("excited_positive", "skeptical_negative", "negative_preference")
NORMALIZATIONS = {
    "per_token_mean_logprob_margin": "per_token_margin",
    "total_sequence_logprob_margin": "total_margin",
}


def _summary(values: pd.Series, cfg: dict[str, Any], seed_offset: int = 0) -> dict[str, Any]:
    return paired_summary(
        values.to_numpy(dtype=float),
        n_boot=cfg["analysis"]["bootstrap_samples"],
        n_perm=cfg["analysis"]["permutation_samples"],
        seed=cfg["analysis"]["seed"] + seed_offset,
    )


def analyze_chat_scores(
    scores_path: str | Path,
    config: dict[str, Any],
    anchors: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = [json.loads(line) for line in Path(scores_path).read_text().splitlines() if line.strip()]
    scores = pd.DataFrame(rows)
    expected_prompts = scores["prompt_hash"].nunique()
    expected_pairs = len(config["scoring"]["continuation_pairs"])
    if len(scores) != expected_prompts * expected_pairs:
        raise ValueError("incomplete prompt-by-continuation score table")

    output: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "n_scenarios": int(scores["scenario_id"].nunique()),
        "n_prompts": int(expected_prompts),
        "n_continuation_pairs": expected_pairs,
        "n_teacher_forced_contrasts": len(scores),
        "primary_normalization": config["scoring"]["primary"],
        "robustness_normalization": config["scoring"]["robustness"],
        "endpoint_warning": "Contrastive margins are log-likelihood scores, not calibrated probabilities.",
        "anchors": anchors,
        "normalizations": {},
    }
    scenario_tables: list[pd.DataFrame] = []

    for norm_index, (normalization, column) in enumerate(NORMALIZATIONS.items()):
        prompt = (
            scores.groupby(
                [
                    "scenario_id",
                    "treatment_id",
                    "treatment_family",
                    "difficulty_class",
                    "difficulty_score",
                ],
                as_index=False,
            )[column]
            .mean()
            .rename(columns={column: "prompt_score"})
        )
        family = (
            prompt.groupby(
                ["scenario_id", "treatment_family", "difficulty_class", "difficulty_score"],
                as_index=False,
            )["prompt_score"]
            .mean()
        )
        wide = family.pivot(index="scenario_id", columns="treatment_family", values="prompt_score")
        metadata = family.drop_duplicates("scenario_id").set_index("scenario_id")
        scenario = metadata[["difficulty_class", "difficulty_score"]].copy()
        for fam in FAMILY_IDS:
            scenario[f"{fam}_minus_neutral"] = wide[fam] - wide["neutral"]
        scenario["negative_preference_direction_normalized"] = (
            wide["neutral"] - wide["negative_preference"]
        )
        scenario["normalization"] = normalization
        scenario_tables.append(scenario.reset_index())

        contrasts: dict[str, Any] = {}
        for family_index, fam in enumerate(FAMILY_IDS):
            effect_col = f"{fam}_minus_neutral"
            contrasts[effect_col] = _summary(
                scenario[effect_col], config, seed_offset=norm_index * 100 + family_index
            )
            contrasts[effect_col]["excluding_case_000"] = _summary(
                scenario.loc[~scenario.index.str.startswith("case_000"), effect_col],
                config,
                seed_offset=norm_index * 100 + family_index + 10,
            )
        direction_col = "negative_preference_direction_normalized"
        contrasts[direction_col] = _summary(
            scenario[direction_col], config, seed_offset=norm_index * 100 + 20
        )
        contrasts[direction_col]["excluding_case_000"] = _summary(
            scenario.loc[~scenario.index.str.startswith("case_000"), direction_col],
            config,
            seed_offset=norm_index * 100 + 21,
        )

        template_effects: dict[str, Any] = {}
        neutral = family[family["treatment_family"] == "neutral"].set_index("scenario_id")[
            "prompt_score"
        ]
        for treatment_id, group in prompt[prompt["treatment_family"] != "neutral"].groupby(
            "treatment_id"
        ):
            values = group.set_index("scenario_id")["prompt_score"] - neutral
            template_effects[treatment_id] = _summary(values, config, seed_offset=norm_index * 1000 + 30)

        continuation_effects: dict[str, Any] = {}
        for pair_index, (pair_id, pair_rows) in enumerate(scores.groupby("continuation_pair_id")):
            pair_prompt = pair_rows.groupby(
                ["scenario_id", "treatment_family"], as_index=False
            )[column].mean()
            pair_wide = pair_prompt.pivot(
                index="scenario_id", columns="treatment_family", values=column
            )
            continuation_effects[pair_id] = {}
            for fam_index, fam in enumerate(FAMILY_IDS):
                continuation_effects[pair_id][f"{fam}_minus_neutral"] = _summary(
                    pair_wide[fam] - pair_wide["neutral"],
                    config,
                    seed_offset=norm_index * 1000 + pair_index * 10 + fam_index + 100,
                )
        pair_means = {
            fam: [
                continuation_effects[pair][f"{fam}_minus_neutral"]["mean"]
                for pair in continuation_effects
            ]
            for fam in FAMILY_IDS
        }
        continuation_variation = {
            fam: {
                "pair_effect_means": values,
                "sd_across_pair_effect_means": round(float(np.std(values, ddof=1)), 4),
            }
            for fam, values in pair_means.items()
        }

        difficulty: dict[str, Any] = {}
        effect = scenario["excited_positive_minus_neutral"]
        for class_index, (label, indexes) in enumerate(
            scenario.groupby("difficulty_class").groups.items()
        ):
            difficulty[str(label)] = _summary(
                effect.loc[indexes], config, seed_offset=norm_index * 1000 + class_index + 500
            )
        difficulty["pearson_r_with_difficulty_score"] = pearson_r(
            scenario["difficulty_score"].to_numpy(), effect.to_numpy()
        )

        output["normalizations"][normalization] = {
            "contrasts": contrasts,
            "difficulty": difficulty,
            "template_effects": template_effects,
            "continuation_pair_effects": continuation_effects,
            "continuation_pair_variation": continuation_variation,
        }

    primary = output["normalizations"][config["scoring"]["primary"]]["contrasts"]
    robust = output["normalizations"][config["scoring"]["robustness"]]["contrasts"]
    output["normalization_robustness"] = {}
    for fam in FAMILY_IDS:
        name = f"{fam}_minus_neutral"
        p_primary = primary[name]
        p_robust = robust[name]
        output["normalization_robustness"][name] = {
            "same_direction": np.sign(p_primary["mean"]) == np.sign(p_robust["mean"]),
            "same_significance_at_0_05": (p_primary["p_perm"] < 0.05)
            == (p_robust["p_perm"] < 0.05),
        }
    return output, pd.concat(scenario_tables, ignore_index=True)


def _fmt(result: dict[str, Any]) -> str:
    lo, hi = result["ci95"]
    return (
        f"{result['mean']:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], "
        f"p={result['p_perm']}, d_z={result['cohens_dz']})"
    )


def render_chat_report(summary: dict[str, Any], config: dict[str, Any]) -> str:
    primary_name = config["scoring"]["primary"]
    robust_name = config["scoring"]["robustness"]
    primary = summary["normalizations"][primary_name]
    robust = summary["normalizations"][robust_name]
    lines = [
        f"# ZIVA JSON-less conversational report: {summary['experiment_name']}",
        "",
        "## Design",
        "",
        f"- Scenarios: {summary['n_scenarios']}; model-facing prompts: {summary['n_prompts']}",
        f"- Frozen continuation pairs: {summary['n_continuation_pairs']}",
        "- The model-facing conversation contains ordinary prose only and requests no structured output, probability, confidence, or grading information.",
        "- Fixed assistant continuations were scored teacher-forced; generated prose and LLM judges were not used.",
        f"- Primary endpoint: `{primary_name}`. Robustness endpoint: `{robust_name}`.",
        "- Positive margins favor a visible-world continuation. These scores are not calibrated probabilities.",
        "",
        "## Primary results",
        "",
        f"- Excited-positive minus neutral: {_fmt(primary['contrasts']['excited_positive_minus_neutral'])}",
        f"- Skeptical-negative minus neutral: {_fmt(primary['contrasts']['skeptical_negative_minus_neutral'])}",
        f"- Negative-preference minus neutral: {_fmt(primary['contrasts']['negative_preference_minus_neutral'])}",
        f"- Negative-preference direction-normalized: {_fmt(primary['contrasts']['negative_preference_direction_normalized'])}",
        "",
        "## Sensitivity excluding case_000",
        "",
    ]
    for name in (
        "excited_positive_minus_neutral",
        "skeptical_negative_minus_neutral",
        "negative_preference_minus_neutral",
    ):
        lines.append(f"- {name}: {_fmt(primary['contrasts'][name]['excluding_case_000'])}")
    lines.extend(["", "## Effect by difficulty", ""])
    for label, result in primary["difficulty"].items():
        if label != "pearson_r_with_difficulty_score":
            lines.append(f"- {label}: {_fmt(result)}")
    lines.append(
        f"- Pearson r with difficulty score: {primary['difficulty']['pearson_r_with_difficulty_score']}"
    )
    lines.extend(["", "## Per-treatment paraphrase effects", ""])
    for treatment_id, result in sorted(primary["template_effects"].items()):
        lines.append(f"- {treatment_id}: {_fmt(result)}")
    lines.extend(["", "## Per-continuation-pair effects", ""])
    for pair_id, effects in primary["continuation_pair_effects"].items():
        lines.append(f"- `{pair_id}` excited-positive minus neutral: {_fmt(effects['excited_positive_minus_neutral'])}")
    lines.extend(["", "## Total-logprob robustness", ""])
    for name in (
        "excited_positive_minus_neutral",
        "skeptical_negative_minus_neutral",
        "negative_preference_minus_neutral",
    ):
        check = summary["normalization_robustness"][name]
        lines.append(
            f"- {name}: {_fmt(robust['contrasts'][name])}; same direction={check['same_direction']}; "
            f"same p<0.05 conclusion={check['same_significance_at_0_05']}"
        )
    anchors = summary.get("anchors") or {}
    lines.extend(["", "## Anchor sanity check", ""])
    lines.append(f"- Passed: {anchors.get('passed')}")
    lines.append(f"- Below-horizon primary margin: {anchors.get('below_horizon_margin')}")
    lines.append(f"- Bright-night primary margin: {anchors.get('bright_night_margin')}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment tests whether irrelevant conversational valence changes Qwen3-14B's distribution over fixed factual continuations. It does not identify internal beliefs, deception, conscious evaluation awareness, or scheming.",
            "",
        ]
    )
    return "\n".join(lines)
