#!/usr/bin/env python3
"""Frozen scenario-paired cross-model interactions for Qwen3-30B-A3B."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ziva.stats import paired_summary
from ziva.util import write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_30b_a3b"
STRUCTURED = ROOT / f"results/oss_{MODEL_ID}_structured/paired_table.csv"
CHAT = ROOT / f"results/oss_{MODEL_ID}_chat_v1/scenario_level_effects.csv"
OUT = ROOT / f"results/oss_{MODEL_ID}_cross_model_comparisons"
REPORT = ROOT / "reports" / f"oss_{MODEL_ID}_cross_model_comparisons.md"
STRUCTURED_MODELS = {
    "qwen3_8b": ROOT / "results/oss_qwen3_8b_structured/paired_table.csv",
    "qwen3_14b": ROOT / "results/pilot_qwen3_14b/paired_table.csv",
    "gemma4_12b": ROOT / "results/oss_gemma4_12b_structured/paired_table.csv",
}
CHAT_MODELS = {
    "qwen3_8b": ROOT / "results/oss_qwen3_8b_chat_v1/scenario_level_effects.csv",
    "qwen3_14b": ROOT / "results/oss_qwen3_14b_chat_v1/scenario_level_effects.csv",
    "gemma4_12b": ROOT / "results/oss_gemma4_12b_chat_v1/scenario_level_effects.csv",
    "gpt56_luna": ROOT / "results/openai_gpt56_luna_chat_v1/scenario_level_effects.csv",
    "gpt56_terra": ROOT / "results/openai_gpt56_terra_chat_v1/scenario_level_effects.csv",
    "gpt56_sol": ROOT / "results/openai_gpt56_sol_chat_v1/scenario_level_effects.csv",
}
STRUCTURED_CONTRASTS = ("excited_minus_neutral", "skeptical_minus_neutral")
CHAT_CONTRASTS = (
    "excited_positive_minus_neutral",
    "skeptical_negative_minus_neutral",
    "negative_preference_minus_neutral",
)


def _scenario_structured(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).groupby("scenario_id", as_index=False)[list(STRUCTURED_CONTRASTS)].mean()


def _fmt(result: dict) -> str:
    lo, hi = result["ci95"]
    return (f"{result['mean']:+.2f} points (95% CI [{lo:+.2f}, {hi:+.2f}], "
            f"p={result['p_perm']}, d_z={result['cohens_dz']})")


def _compare(
    protocol: str,
    focal: pd.DataFrame,
    comparators: dict[str, Path],
    contrasts: tuple[str, ...],
    seed_base: int,
) -> tuple[dict, list[str]]:
    output: dict = {}
    lines: list[str] = []
    for model_index, (name, path) in enumerate(comparators.items()):
        comparator = _scenario_structured(path) if protocol == "structured" else pd.read_csv(path)
        merged = focal[["scenario_id", *contrasts]].merge(
            comparator[["scenario_id", *contrasts]], on="scenario_id",
            suffixes=("_qwen30", "_comparator"), validate="one_to_one",
        )
        if len(merged) != 32:
            raise RuntimeError(f"{protocol}/{name}: expected 32 paired scenarios, found {len(merged)}")
        table = merged[["scenario_id"]].copy()
        model_summary = {"contrasts": {}}
        lines.extend([f"## {protocol.title()}: Qwen3-30B-A3B minus {name}", ""])
        for contrast_index, contrast in enumerate(contrasts):
            focal_values = merged[f"{contrast}_qwen30"]
            comparator_values = merged[f"{contrast}_comparator"]
            difference = focal_values - comparator_values
            table[f"{contrast}_qwen30"] = focal_values
            table[f"{contrast}_{name}"] = comparator_values
            table[f"{contrast}_interaction"] = difference
            result = paired_summary(
                difference.to_numpy(dtype=float), n_boot=5000, n_perm=5000,
                seed=seed_base + model_index * 10 + contrast_index,
            )
            result["qwen3_30b_a3b_mean_effect"] = round(float(focal_values.mean()), 4)
            result["comparator_mean_effect"] = round(float(comparator_values.mean()), 4)
            model_summary["contrasts"][contrast] = result
            lines.extend([
                f"### {contrast}", "",
                f"- Qwen3-30B-A3B effect: {result['qwen3_30b_a3b_mean_effect']:+.2f} points",
                f"- {name} effect: {result['comparator_mean_effect']:+.2f} points",
                f"- Paired interaction: {_fmt(result)}", "",
            ])
        table.to_csv(OUT / f"{protocol}_qwen30_minus_{name}_scenario_level.csv", index=False)
        output[name] = model_summary
    return output, lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    structured, structured_lines = _compare(
        "structured", _scenario_structured(STRUCTURED), STRUCTURED_MODELS,
        STRUCTURED_CONTRASTS, 1000,
    )
    chat, chat_lines = _compare(
        "chat_v1", pd.read_csv(CHAT), CHAT_MODELS, CHAT_CONTRASTS, 2000,
    )
    summary = {
        "comparison": "Qwen3-30B-A3B minus prior model within frozen protocol",
        "n_scenarios": 32,
        "estimand": "D_s = focal treatment contrast_s - comparator treatment contrast_s",
        "bootstrap_samples": 5000, "permutation_samples": 5000,
        "structured": structured, "chat_v1": chat,
    }
    write_json(OUT / "summary.json", summary)
    REPORT.write_text("\n".join([
        "# Qwen3-30B-A3B cross-model comparisons", "",
        "Direct model-by-treatment interactions paired over the same 32 frozen scenarios.", "",
        *structured_lines, *chat_lines,
        "## Interpretation boundary", "",
        "These are direct paired interactions, not comparisons of individual-model p-values.", "",
    ]), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
