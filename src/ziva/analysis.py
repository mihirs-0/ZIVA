"""Analysis: parsed tables, paired metrics, statistics, and plots.

Everything here is computed deterministically from stored raw records; no LLM
is involved in scoring. Outputs land in results/<experiment>/:

    summary.json           machine-readable full summary
    paired_table.csv       the core paired object (spec 53)
    trials.csv / .jsonl    trial-level parsed data
    plots/*.png            publication figures
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .metrics import (
    paired_table,
    preference_direction_table,
    primary_final_step,
    threshold_crossings,
    trials_dataframe,
    within_cell_sampling_sd,
)
from .stats import holm_bonferroni, paired_summary, pearson_r
from .treatments import PRIMARY_NEUTRAL_FAMILY, PRIMARY_POSITIVE_FAMILY
from .util import read_json, write_json

# Validated categorical palette (dataviz reference instance, light mode).
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#dddcd8"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": C_GRID,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_raw_records(cfg: ExperimentConfig) -> list[dict]:
    raw_dir = cfg.raw_dir
    if not raw_dir.exists():
        return []
    return [read_json(p) for p in sorted(raw_dir.glob("t_*.json"))]


def _cell_means(d: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        d.groupby(["scenario_id", "model_id", "evidence_mode", "treatment_family"])[value]
        .mean().unstack("treatment_family")
    )


def _family_vs_neutral(d: pd.DataFrame, value: str, family: str) -> np.ndarray:
    cells = _cell_means(d, value)
    if family not in cells.columns or PRIMARY_NEUTRAL_FAMILY not in cells.columns:
        return np.array([])
    diff = (cells[family] - cells[PRIMARY_NEUTRAL_FAMILY]).dropna()
    return diff.to_numpy()


def _recommend_rate_diff(d: pd.DataFrame, family: str) -> np.ndarray:
    dd = d.copy()
    dd["rec"] = dd["would_recommend_attempt"].astype(float)
    return _family_vs_neutral(dd, "rec", family)


def analyze(cfg: ExperimentConfig, scenarios: list[dict]) -> dict:
    records = load_raw_records(cfg)
    out_dir = cfg.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    df = trials_dataframe(records, scenarios)
    if df.empty:
        summary = {"error": "no raw records found; run `ziva run` first"}
        write_json(out_dir / "summary.json", summary)
        return summary

    df.to_csv(out_dir / "trials.csv", index=False)
    df.to_json(out_dir / "trials.jsonl", orient="records", lines=True)

    providers = sorted(set(df["provider"].dropna()))
    synthetic = "mock" in providers

    # ------------------------------------------------------------ parse stats
    final_rows = df[df["step_index"] == df["n_steps"] - 1]
    parse_stats = {
        "total_trials": int(df["trial_id"].nunique()),
        "malformed_rate_overall": round(float((final_rows["parse_status"] != "ok").mean()), 4)
        if len(final_rows) else None,
        "malformed_by_model": {
            str(m): round(float((g["parse_status"] != "ok").mean()), 4)
            for m, g in final_rows.groupby("model_id")
        },
        "status_counts": final_rows["parse_status"].value_counts().to_dict(),
    }

    # -------------------------------------------------------------- primary
    table = paired_table(df)
    if not table.empty:
        table.to_csv(out_dir / "paired_table.csv", index=False)
    d_primary = primary_final_step(df)

    primary_by_cell: dict[str, dict] = {}
    secondary_pvals: dict[str, float] = {}
    if not table.empty and "excited_minus_neutral" in table.columns:
        for (model_id, mode), grp in table.groupby(["model_id", "evidence_mode"]):
            key = f"{model_id}|{mode}"
            entry: dict = {
                "excited_minus_neutral": paired_summary(grp["excited_minus_neutral"].to_numpy()),
                "valence_range_mean": round(float(grp["valence_range"].mean()), 3),
            }
            if "skeptical_minus_neutral" in grp.columns:
                entry["skeptical_minus_neutral"] = paired_summary(grp["skeptical_minus_neutral"].to_numpy())
                if entry["skeptical_minus_neutral"]["p_perm"] is not None:
                    secondary_pvals[f"skeptical|{key}"] = entry["skeptical_minus_neutral"]["p_perm"]
            if "binary_flip" in grp.columns:
                entry["binary_flip_rate"] = round(float(grp["binary_flip"].fillna(False).mean()), 4)
            entry["threshold_crossings_excited_vs_neutral"] = threshold_crossings(
                grp, f"{PRIMARY_NEUTRAL_FAMILY}_probability", f"{PRIMARY_POSITIVE_FAMILY}_probability"
            )
            sub = d_primary[(d_primary["model_id"] == model_id) & (d_primary["evidence_mode"] == mode)]
            entry["confidence_shift_excited_minus_neutral"] = paired_summary(
                _family_vs_neutral(sub, "confidence", PRIMARY_POSITIVE_FAMILY))
            entry["recommendation_shift_excited_minus_neutral"] = paired_summary(
                _recommend_rate_diff(sub, PRIMARY_POSITIVE_FAMILY))
            primary_by_cell[key] = entry

    pooled = (
        paired_summary(table["excited_minus_neutral"].to_numpy())
        if not table.empty and "excited_minus_neutral" in table.columns else None
    )

    # ---------------------------------------- family-level effects vs neutral
    by_family: dict[str, dict] = {}
    for fam in sorted(set(d_primary["treatment_family"].dropna())):
        if fam == PRIMARY_NEUTRAL_FAMILY:
            continue
        diffs = _family_vs_neutral(d_primary, "visible_probability", fam)
        by_family[fam] = paired_summary(diffs)
        if by_family[fam]["p_perm"] is not None and fam != PRIMARY_POSITIVE_FAMILY:
            secondary_pvals[f"family|{fam}"] = by_family[fam]["p_perm"]

    # template-level effects (prompt robustness)
    by_template: dict[str, dict] = {}
    neutral_cells = (
        d_primary[d_primary["treatment_family"] == PRIMARY_NEUTRAL_FAMILY]
        .groupby(["scenario_id", "model_id", "evidence_mode"])["visible_probability"].mean()
    )
    for tid, grp in d_primary[d_primary["treatment_family"] != PRIMARY_NEUTRAL_FAMILY].groupby("treatment_id"):
        cells = grp.groupby(["scenario_id", "model_id", "evidence_mode"])["visible_probability"].mean()
        joined = pd.concat([cells.rename("t"), neutral_cells.rename("n")], axis=1, join="inner").dropna()
        if len(joined):
            by_template[str(tid)] = paired_summary((joined["t"] - joined["n"]).to_numpy())

    # ------------------------------------------------- preference direction
    pref_table = preference_direction_table(df)
    preference: dict[str, dict] = {}
    if not pref_table.empty:
        for fam, grp in pref_table.groupby("treatment_family"):
            preference[str(fam)] = paired_summary(grp["delta_preference"].to_numpy())

    # ------------------------------------------------ difficulty interaction
    difficulty: dict = {}
    if not table.empty and "excited_minus_neutral" in table.columns:
        difficulty["pearson_r_effect_vs_difficulty"] = pearson_r(
            table["difficulty_score"].to_numpy(), table["excited_minus_neutral"].to_numpy())
        difficulty["by_difficulty_class"] = {
            str(k): paired_summary(g["excited_minus_neutral"].to_numpy())
            for k, g in table.groupby("difficulty_class")
        }

    # --------------------------------------- effect vs sampling variance
    variance_comparison: dict = {}
    sds = within_cell_sampling_sd(df)
    if not sds.empty and pooled and pooled["mean"] is not None:
        mean_within_sd = float(sds["within_cell_sd"].dropna().mean()) if sds["within_cell_sd"].notna().any() else None
        variance_comparison = {
            "mean_within_cell_sd_points": round(mean_within_sd, 3) if mean_within_sd is not None else None,
            "pooled_excited_minus_neutral_points": pooled["mean"],
            "abs_effect_over_within_sd_ratio": (
                round(abs(pooled["mean"]) / mean_within_sd, 3)
                if mean_within_sd and mean_within_sd > 0 else None
            ),
            "interpretation": (
                "ratios well below 1 mean the valence effect is smaller than ordinary "
                "repeated-sampling variability (a key falsification criterion)"
            ),
        }

    # ------------------------------------------------------ evidence update
    evidence_update: dict = {}
    d_upd = df[(df["experiment"] == "evidence_update") & (df["parse_status"] == "ok")]
    if not d_upd.empty:
        before = d_upd[d_upd["step_index"] == 0].groupby(
            ["scenario_id", "model_id", "reaction_family"])["visible_probability"].mean()
        after = d_upd[d_upd["step_index"] == 1].groupby(
            ["scenario_id", "model_id", "reaction_family"])["visible_probability"].mean()
        delta = (after - before).rename("delta_update").dropna().reset_index()
        neutral_delta = delta[delta["reaction_family"] == "neutral"].set_index(
            ["scenario_id", "model_id"])["delta_update"]
        for fam, grp in delta.groupby("reaction_family"):
            entry = {"delta_update": paired_summary(grp["delta_update"].to_numpy())}
            if fam != "neutral" and len(neutral_delta):
                joined = grp.set_index(["scenario_id", "model_id"])["delta_update"].to_frame("d").join(
                    neutral_delta.rename("n"), how="inner").dropna()
                if len(joined):
                    entry["delta_vs_neutral_reaction"] = paired_summary((joined["d"] - joined["n"]).to_numpy())
                    if entry["delta_vs_neutral_reaction"]["p_perm"] is not None:
                        secondary_pvals[f"update|{fam}"] = entry["delta_vs_neutral_reaction"]["p_perm"]
            evidence_update[str(fam)] = entry

    # ----------------------------------------------------------- commitment
    commitment: dict = {}
    d_com = df[(df["experiment"] == "commitment") & (df["parse_status"] == "ok")]
    if not d_com.empty:
        finals = d_com[d_com["step_index"] == d_com["n_steps"] - 1]
        cell = finals.groupby(["scenario_id", "model_id", "condition"])["visible_probability"].mean().unstack()
        if {"commitment_a", "commitment_b"} <= set(cell.columns):
            diffs = (cell["commitment_a"] - cell["commitment_b"]).dropna().to_numpy()
            commitment["a_minus_b_final_probability"] = paired_summary(diffs)
            if commitment["a_minus_b_final_probability"]["p_perm"] is not None:
                secondary_pvals["commitment|a_minus_b"] = commitment["a_minus_b_final_probability"]["p_perm"]

    # --------------------------------------------------------------- web
    web: dict = {}
    d_web = df[(df["experiment"] == "web") & (df["parse_status"] == "ok")]
    if not d_web.empty:
        web["n_trials"] = int(len(d_web))
        web["search_used_rate"] = round(float((d_web["tools_used"].fillna("") != "").mean()), 4)
        web["excited_minus_neutral"] = paired_summary(
            _family_vs_neutral(d_web, "visible_probability", PRIMARY_POSITIVE_FAMILY))

    summary = {
        "experiment_name": cfg.experiment_name,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "SYNTHETIC_DATA": synthetic,
        "synthetic_note": (
            "This analysis includes records from the mock provider. All numbers are "
            "SYNTHETIC pipeline-test data, not real model behavior." if synthetic else None
        ),
        "providers": providers,
        "models": sorted(set(df["model_id"].dropna())),
        "parse_stats": parse_stats,
        "primary_pooled_excited_minus_neutral": pooled,
        "primary_by_model_and_mode": primary_by_cell,
        "family_effects_vs_neutral": by_family,
        "template_effects_vs_neutral": by_template,
        "preference_direction_effects": preference,
        "difficulty_interaction": difficulty,
        "variance_comparison": variance_comparison,
        "evidence_update": evidence_update,
        "commitment": commitment,
        "web": web,
        "secondary_pvalues_holm_adjusted": holm_bonferroni(secondary_pvals),
    }
    write_json(out_dir / "summary.json", summary)

    _make_plots(table, d_primary, df, evidence_update, plots_dir)
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_plots(table: pd.DataFrame, d_primary: pd.DataFrame, df: pd.DataFrame,
                evidence_update: dict, plots_dir: Path) -> None:
    pos_col = f"{PRIMARY_POSITIVE_FAMILY}_probability"
    neu_col = f"{PRIMARY_NEUTRAL_FAMILY}_probability"
    have_pair = not table.empty and pos_col in table.columns and neu_col in table.columns

    # 1. excited vs neutral paired scatter (identity line; full 0-100 axes)
    if have_pair:
        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        ax.plot([0, 100], [0, 100], color=C_MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(table[neu_col], table[pos_col], s=26, color=C_BLUE, alpha=0.65,
                   edgecolors=SURFACE, linewidths=0.8, zorder=2)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("neutral framing: visible_probability")
        ax.set_ylabel("excited framing: visible_probability")
        ax.set_title("Excited vs neutral estimates (paired; one point per scenario x model x mode)")
        _save(fig, plots_dir / "01_excited_vs_neutral.png")

        # 2. paired differences per scenario (dot plot, symmetric axis)
        diffs = table["excited_minus_neutral"].dropna().sort_values().reset_index(drop=True)
        lim = max(10, float(np.ceil(diffs.abs().max() / 5) * 5)) if len(diffs) else 10
        fig, ax = plt.subplots(figsize=(6.4, max(3.2, 0.05 * len(diffs) + 2)))
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.scatter(diffs, range(len(diffs)), s=14, color=C_BLUE, alpha=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_yticks([])
        ax.set_xlabel("excited - neutral (probability points)")
        ax.set_title("Paired valence differences, sorted")
        _save(fig, plots_dir / "02_paired_differences.png")

        # 3. distribution of valence effects
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.hist(diffs, bins=21, color=C_BLUE, edgecolor=SURFACE)
        ax.axvline(0, color=C_MUTED, lw=1)
        if len(diffs):
            ax.axvline(float(diffs.mean()), color=C_ORANGE, lw=1.6,
                       label=f"mean = {diffs.mean():+.1f}")
            ax.legend(frameon=False)
        ax.set_xlabel("excited - neutral (probability points)")
        ax.set_ylabel("paired cells")
        ax.set_title("Distribution of valence effects")
        _save(fig, plots_dir / "03_effect_distribution.png")

        # 4. effect by model (mean with bootstrap CI)
        fig, ax = plt.subplots(figsize=(6.2, 0.6 * table["model_id"].nunique() + 2))
        ax.axvline(0, color=C_MUTED, lw=1)
        for i, (model_id, grp) in enumerate(sorted(table.groupby("model_id"), key=lambda kv: kv[0])):
            s = paired_summary(grp["excited_minus_neutral"].to_numpy())
            if s["mean"] is None:
                continue
            lo, hi = s["ci95"]
            ax.plot([lo, hi], [i, i], color=C_BLUE, lw=2)
            ax.scatter([s["mean"]], [i], color=C_BLUE, s=42, zorder=3)
            ax.text(hi + 0.4, i, model_id, va="center", fontsize=9, color=C_TEXT)
        ax.set_yticks([])
        ax.set_xlabel("excited - neutral (points, mean with 95% bootstrap CI)")
        ax.set_title("Valence effect by model")
        _save(fig, plots_dir / "04_effect_by_model.png")

        # 5. effect by evidence modality
        fig, ax = plt.subplots(figsize=(6.2, 0.6 * table["evidence_mode"].nunique() + 2))
        ax.axvline(0, color=C_MUTED, lw=1)
        for i, (mode, grp) in enumerate(sorted(table.groupby("evidence_mode"), key=lambda kv: kv[0])):
            s = paired_summary(grp["excited_minus_neutral"].to_numpy())
            if s["mean"] is None:
                continue
            lo, hi = s["ci95"]
            ax.plot([lo, hi], [i, i], color=C_AQUA, lw=2)
            ax.scatter([s["mean"]], [i], color=C_AQUA, s=42, zorder=3)
            ax.text(hi + 0.4, i, mode, va="center", fontsize=9, color=C_TEXT)
        ax.set_yticks([])
        ax.set_xlabel("excited - neutral (points, mean with 95% bootstrap CI)")
        ax.set_title("Valence effect by evidence modality")
        _save(fig, plots_dir / "05_effect_by_modality.png")

        # 6. effect vs scenario difficulty
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.axhline(0, color=C_MUTED, lw=1)
        ax.scatter(table["difficulty_score"], table["excited_minus_neutral"], s=24,
                   color=C_BLUE, alpha=0.6, edgecolors=SURFACE, linewidths=0.7)
        ax.set_xlabel("scenario difficulty score (ambiguity proxy)")
        ax.set_ylabel("excited - neutral (points)")
        ax.set_title("Valence effect vs perceptual ambiguity")
        _save(fig, plots_dir / "06_effect_by_difficulty.png")

        # 7. binary flip rate by model
        if "binary_flip" in table.columns:
            flips = table.groupby("model_id")["binary_flip"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(flips) + 2))
            ax.barh(range(len(flips)), flips.to_numpy(), color=C_ORANGE, height=0.55)
            ax.set_yticks(range(len(flips)), flips.index, fontsize=9)
            ax.set_xlim(0, max(0.2, float(flips.max()) * 1.2 if len(flips) else 0.2))
            ax.set_xlabel("fraction of paired cells where the binary prediction flips")
            ax.set_title("Factual flip rate (excited vs neutral)")
            _save(fig, plots_dir / "07_flip_rate.png")

    # 8. confidence shift distribution
    conf = _family_vs_neutral(d_primary, "confidence", PRIMARY_POSITIVE_FAMILY)
    if len(conf):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.hist(conf, bins=21, color=C_YELLOW, edgecolor=SURFACE)
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.set_xlabel("confidence shift, excited - neutral (points)")
        ax.set_ylabel("paired cells")
        ax.set_title("Confidence effects")
        _save(fig, plots_dir / "08_confidence_shift.png")

    # 9. recommendation shift distribution
    rec = _recommend_rate_diff(d_primary, PRIMARY_POSITIVE_FAMILY)
    if len(rec):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.hist(rec, bins=11, color=C_AQUA, edgecolor=SURFACE)
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.set_xlabel("recommendation-rate shift, excited - neutral")
        ax.set_ylabel("paired cells")
        ax.set_title("Recommendation effects (analyzed separately from factual belief)")
        _save(fig, plots_dir / "09_recommendation_shift.png")

    # 10. evidence-update effects by reaction family
    if evidence_update:
        fams = sorted(evidence_update)
        means, los, his = [], [], []
        for f in fams:
            s = evidence_update[f]["delta_update"]
            means.append(s["mean"])
            ci = s["ci95"]
            los.append(ci[0])
            his.append(ci[1])
        fig, ax = plt.subplots(figsize=(6.2, 0.6 * len(fams) + 2))
        ax.axvline(0, color=C_MUTED, lw=1)
        for i, f in enumerate(fams):
            if means[i] is None:
                continue
            ax.plot([los[i], his[i]], [i, i], color=C_BLUE, lw=2)
            ax.scatter([means[i]], [i], color=C_BLUE, s=42, zorder=3)
            ax.text(max(h for h in his if h is not None) + 0.5, i, f, va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel("update magnitude p_after - p_before (points, 95% CI)")
        ax.set_title("Evidence-update effects by user reaction")
        _save(fig, plots_dir / "10_evidence_update.png")
