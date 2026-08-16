#!/usr/bin/env python3
"""Paired DeepSeek-minus-prior-model Chat-v1 interaction comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ziva.stats import paired_summary
from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK = ROOT / "results/oss_deepseek_r1_distill_qwen_32b_chat_v1/scenario_level_effects.csv"
OUT = ROOT / "results/oss_deepseek_r1_distill_qwen_32b_chat_v1_comparisons"
REPORT = ROOT / "reports/oss_deepseek_r1_distill_qwen_32b_chat_v1_comparisons.md"
MODELS = {
    "qwen3_14b": ROOT / "results/oss_qwen3_14b_chat_v1/scenario_level_effects.csv",
    "qwen3_8b": ROOT / "results/oss_qwen3_8b_chat_v1/scenario_level_effects.csv",
    "gemma4_12b": ROOT / "results/oss_gemma4_12b_chat_v1/scenario_level_effects.csv",
    "gpt56_luna": ROOT / "results/openai_gpt56_luna_chat_v1/scenario_level_effects.csv",
    "gpt56_terra": ROOT / "results/openai_gpt56_terra_chat_v1/scenario_level_effects.csv",
}
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
    deepseek = pd.read_csv(DEEPSEEK)[["scenario_id", *CONTRASTS]]
    summary = {
        "comparison": "DeepSeek-R1-Distill-Qwen-32B minus prior model under frozen ZIVA-Chat-v1",
        "n_scenarios": 32,
        "estimand": "D_s = treatment contrast DeepSeek_s - treatment contrast comparator_s",
        "bootstrap_samples": 5000,
        "permutation_samples": 5000,
        "seed_rule": "comparison_index * 10 + contrast_index",
        "comparisons": {},
    }
    lines = [
        "# DeepSeek-R1-Distill-Qwen-32B cross-model Chat-v1 comparisons",
        "",
        "Direct paired model-by-treatment comparisons over the same 32 frozen scenarios.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for model_index, (name, path) in enumerate(MODELS.items()):
        comparator = pd.read_csv(path)[["scenario_id", *CONTRASTS]]
        merged = deepseek.merge(
            comparator, on="scenario_id", suffixes=("_deepseek", "_comparator"), validate="one_to_one"
        )
        if len(merged) != 32:
            raise RuntimeError(f"{name}: expected 32 paired scenarios, found {len(merged)}")
        table = merged[["scenario_id"]].copy()
        model_summary = {"contrasts": {}}
        lines.extend([f"## DeepSeek minus {name}", ""])
        for contrast_index, contrast in enumerate(CONTRASTS):
            deepseek_values = merged[f"{contrast}_deepseek"]
            comparator_values = merged[f"{contrast}_comparator"]
            difference = deepseek_values - comparator_values
            table[f"{contrast}_deepseek"] = deepseek_values
            table[f"{contrast}_{name}"] = comparator_values
            table[f"{contrast}_deepseek_minus_{name}"] = difference
            result = paired_summary(
                difference.to_numpy(dtype=float),
                n_boot=5000,
                n_perm=5000,
                seed=model_index * 10 + contrast_index,
            )
            result["deepseek_mean_effect"] = round(float(deepseek_values.mean()), 4)
            result["comparator_mean_effect"] = round(float(comparator_values.mean()), 4)
            model_summary["contrasts"][contrast] = result
            lines.extend(
                [
                    f"### {contrast}",
                    "",
                    f"- DeepSeek effect: {result['deepseek_mean_effect']:+.2f} points",
                    f"- {name} effect: {result['comparator_mean_effect']:+.2f} points",
                    f"- Paired interaction: {_fmt(result)}",
                    "",
                ]
            )
        table.to_csv(OUT / f"deepseek_minus_{name}_scenario_level.csv", index=False)
        summary["comparisons"][name] = model_summary
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These are direct paired interactions, not comparisons of individual-model significance labels.",
            "",
        ]
    )
    write_json(OUT / "summary.json", summary)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
