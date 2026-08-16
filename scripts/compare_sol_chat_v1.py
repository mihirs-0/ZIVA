#!/usr/bin/env python3
"""Direct paired GPT-5.6 Sol-minus-Luna/Terra Chat-v1 comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ziva.stats import paired_summary
from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "results/openai_gpt56_sol_chat_v1/scenario_level_effects.csv"
COMPARATORS = {
    "luna": ROOT / "results/openai_gpt56_luna_chat_v1/scenario_level_effects.csv",
    "terra": ROOT / "results/openai_gpt56_terra_chat_v1/scenario_level_effects.csv",
}
OUT = ROOT / "results/openai_gpt56_sol_chat_v1_comparisons"
REPORT = ROOT / "reports/openai_gpt56_sol_chat_v1_comparisons.md"
CONTRASTS = (
    "excited_positive_minus_neutral",
    "skeptical_negative_minus_neutral",
    "negative_preference_minus_neutral",
)


def _fmt(result: dict) -> str:
    lo, hi = result["ci95"]
    return (
        f"{result['mean']:+.2f} points (95% CI [{lo:+.2f}, {hi:+.2f}], "
        f"p={result['p_perm']}, d_z={result['cohens_dz']})"
    )


def main() -> None:
    sol = pd.read_csv(SOL)[["scenario_id", *CONTRASTS]]
    summary = {
        "estimand": "D_s = treatment contrast Sol_s - treatment contrast comparator_s",
        "n_scenarios": 32,
        "bootstrap_samples": 5000,
        "permutation_samples": 5000,
        "comparisons": {},
    }
    lines = ["# GPT-5.6 Sol paired Chat-v1 interactions", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    for model_index, (name, path) in enumerate(COMPARATORS.items()):
        other = pd.read_csv(path)[["scenario_id", *CONTRASTS]]
        merged = sol.merge(other, on="scenario_id", suffixes=("_sol", f"_{name}"), validate="one_to_one")
        if len(merged) != 32:
            raise RuntimeError(f"{name}: expected 32 paired scenarios, found {len(merged)}")
        table = merged[["scenario_id"]].copy()
        comparison = {"contrasts": {}}
        lines.extend([f"## Sol minus {name.title()}", ""])
        for contrast_index, contrast in enumerate(CONTRASTS):
            sol_values = merged[f"{contrast}_sol"]
            other_values = merged[f"{contrast}_{name}"]
            difference = sol_values - other_values
            table[f"{contrast}_sol"] = sol_values
            table[f"{contrast}_{name}"] = other_values
            table[f"{contrast}_sol_minus_{name}"] = difference
            result = paired_summary(
                difference.to_numpy(dtype=float),
                n_boot=5000,
                n_perm=5000,
                seed=model_index * 10 + contrast_index,
            )
            result["sol_mean_effect"] = round(float(sol_values.mean()), 4)
            result["comparator_mean_effect"] = round(float(other_values.mean()), 4)
            comparison["contrasts"][contrast] = result
            lines.extend(
                [
                    f"### {contrast}",
                    "",
                    f"- Sol effect: {result['sol_mean_effect']:+.2f} points",
                    f"- {name.title()} effect: {result['comparator_mean_effect']:+.2f} points",
                    f"- Paired interaction: {_fmt(result)}",
                    "",
                ]
            )
        table.to_csv(OUT / f"sol_minus_{name}_scenario_level.csv", index=False)
        summary["comparisons"][name] = comparison
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These are direct paired model-by-treatment interactions, not comparisons of individual p-values.",
            "",
        ]
    )
    write_json(OUT / "summary.json", summary)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
