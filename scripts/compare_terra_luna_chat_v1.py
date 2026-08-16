#!/usr/bin/env python3
"""Direct paired GPT-5.6 Terra-minus-Luna Chat-v1 treatment comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ziva.stats import paired_summary
from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
TERRA = ROOT / "results/openai_gpt56_terra_chat_v1/scenario_level_effects.csv"
LUNA = ROOT / "results/openai_gpt56_luna_chat_v1/scenario_level_effects.csv"
OUT = ROOT / "results/openai_gpt56_terra_vs_luna_chat_v1"
REPORT = ROOT / "reports/openai_gpt56_terra_vs_luna_chat_v1_report.md"
CONTRASTS = (
    "excited_positive_minus_neutral",
    "skeptical_negative_minus_neutral",
    "negative_preference_minus_neutral",
)


def fmt(result: dict) -> str:
    lo, hi = result["ci95"]
    return (
        f"{result['mean']:+.2f} points (95% CI [{lo:+.2f}, {hi:+.2f}], "
        f"p={result['p_perm']}, d_z={result['cohens_dz']})"
    )


def main() -> None:
    terra = pd.read_csv(TERRA)[["scenario_id", *CONTRASTS]]
    luna = pd.read_csv(LUNA)[["scenario_id", *CONTRASTS]]
    merged = terra.merge(luna, on="scenario_id", suffixes=("_terra", "_luna"), validate="one_to_one")
    if len(merged) != 32:
        raise RuntimeError(f"expected 32 paired scenarios, found {len(merged)}")
    summary = {
        "comparison": "GPT-5.6 Terra minus GPT-5.6 Luna under frozen ZIVA-Chat-v1",
        "n_scenarios": len(merged),
        "estimand": "D_s = treatment contrast Terra_s - treatment contrast Luna_s",
        "bootstrap_samples": 5000,
        "permutation_samples": 5000,
        "seed": 0,
        "contrasts": {},
    }
    table = merged[["scenario_id"]].copy()
    for offset, contrast in enumerate(CONTRASTS):
        difference = merged[f"{contrast}_terra"] - merged[f"{contrast}_luna"]
        table[f"{contrast}_terra"] = merged[f"{contrast}_terra"]
        table[f"{contrast}_luna"] = merged[f"{contrast}_luna"]
        table[f"{contrast}_terra_minus_luna"] = difference
        result = paired_summary(
            difference.to_numpy(dtype=float), n_boot=5000, n_perm=5000, seed=offset
        )
        result["terra_mean_effect"] = round(float(merged[f"{contrast}_terra"].mean()), 4)
        result["luna_mean_effect"] = round(float(merged[f"{contrast}_luna"].mean()), 4)
        summary["contrasts"][contrast] = result
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "scenario_level_difference_of_differences.csv", index=False)
    write_json(OUT / "summary.json", summary)
    lines = [
        "# GPT-5.6 Terra versus Luna: frozen ZIVA-Chat-v1",
        "",
        "Direct scenario-level model-by-treatment comparison over the same 32 scenarios.",
        "For each contrast, `D_s = delta_Terra_s - delta_Luna_s`.",
        "",
        "## Difference-of-differences",
        "",
    ]
    for contrast, result in summary["contrasts"].items():
        lines.extend(
            [
                f"### {contrast}",
                "",
                f"- Terra effect: {result['terra_mean_effect']:+.2f} points",
                f"- Luna effect: {result['luna_mean_effect']:+.2f} points",
                f"- Terra minus Luna: {fmt(result)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These are direct paired interaction contrasts. They do not infer heterogeneity from "
            "different within-model significance labels and do not establish internal beliefs, motives, "
            "deception, or evaluation awareness.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
