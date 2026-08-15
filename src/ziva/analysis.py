"""Analysis: parsed tables, paired metrics, statistics, and plots.

Everything here is computed deterministically from stored raw records; no LLM
is involved in scoring. Outputs land in results/<experiment>/:

    summary.json              machine-readable full summary
    paired_table.csv          the core paired object (spec 53)
    scenario_level_diffs.csv  one primary-effect value per physical scenario
    trials.csv / .jsonl       trial-level parsed data
    plots/*.png               publication figures

Statistical units
-----------------
The physical scenario is the independently sampled experimental unit. All
inferential statistics therefore either (a) operate within a single
model x evidence x elicitation cell, where each scenario contributes exactly
one paired difference, or (b) first aggregate to one value per scenario before
bootstrap/permutation. Row-level cross-cell pooling appears only as a
descriptive mean, clearly labelled.
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
    CELL_KEYS,
    paired_table,
    preference_direction_table,
    primary_final_step,
    scenario_level_diffs,
    threshold_crossings,
    trials_dataframe,
    within_cell_variance_components,
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


def _cell_diff_series(d: pd.DataFrame, value: str, family: str) -> pd.Series:
    """family-minus-neutral differences indexed by cell (scenario in index)."""
    cells = (
        d.groupby(CELL_KEYS + ["treatment_family"])[value]
        .mean().unstack("treatment_family")
    )
    if family not in cells.columns or PRIMARY_NEUTRAL_FAMILY not in cells.columns:
        return pd.Series(dtype=float)
    return (cells[family] - cells[PRIMARY_NEUTRAL_FAMILY]).dropna()


def _scenario_agg(series: pd.Series) -> np.ndarray:
    """Aggregate a cell-indexed series to one mean value per scenario (the
    independent experimental unit)."""
    if series.empty:
        return np.array([])
    frame = series.rename("v").reset_index()
    return frame.groupby("scenario_id")["v"].mean().to_numpy()


def _family_vs_neutral_by_scenario(d: pd.DataFrame, value: str, family: str) -> np.ndarray:
    return _scenario_agg(_cell_diff_series(d, value, family))


def _recommend_series(d: pd.DataFrame, family: str) -> pd.Series:
    dd = d.copy()
    dd["rec"] = dd["would_recommend_attempt"].astype(float)
    return _cell_diff_series(dd, "rec", family)


def _grouped_scenario_summary(table: pd.DataFrame, group_col: str,
                              col: str = "excited_minus_neutral") -> dict[str, dict]:
    """Per-group effect summaries with the scenario as the paired unit:
    within each group, first average the effect per scenario across the
    remaining cell dimensions, then run the paired statistics."""
    out: dict[str, dict] = {}
    for key, grp in table.dropna(subset=[col]).groupby(group_col):
        per_scenario = grp.groupby("scenario_id")[col].mean().to_numpy()
        out[str(key)] = paired_summary(per_scenario)
    return out


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
    fingerprints = sorted(set(df["experiment_fingerprint"].dropna()))

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

    scen_diffs = scenario_level_diffs(table)
    if not scen_diffs.empty:
        scen_diffs.to_csv(out_dir / "scenario_level_diffs.csv", index=False)

    primary_by_cell: dict[str, dict] = {}
    secondary_pvals: dict[str, float] = {}
    if not table.empty and "excited_minus_neutral" in table.columns:
        for (model_id, mode, elic), grp in table.groupby(["model_id", "evidence_mode", "elicitation_mode"]):
            key = f"{model_id}|{mode}|{elic}"
            entry: dict = {
                # within one cell each scenario contributes exactly one pair
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
            sub = d_primary[(d_primary["model_id"] == model_id)
                            & (d_primary["evidence_mode"] == mode)
                            & (d_primary["elicitation_mode"] == elic)]
            entry["confidence_shift_excited_minus_neutral"] = paired_summary(
                _cell_diff_series(sub, "confidence", PRIMARY_POSITIVE_FAMILY).to_numpy())
            entry["recommendation_shift_excited_minus_neutral"] = paired_summary(
                _recommend_series(sub, PRIMARY_POSITIVE_FAMILY).to_numpy())
            primary_by_cell[key] = entry

    # Pooled PRIMARY inference: exactly one value per physical scenario.
    pooled = (
        paired_summary(scen_diffs["excited_minus_neutral"].to_numpy())
        if not scen_diffs.empty else None
    )
    # Row-level pooled mean: DESCRIPTIVE ONLY (rows are correlated within scenario).
    pooled_rows_descriptive = (
        round(float(table["excited_minus_neutral"].dropna().mean()), 4)
        if not table.empty and "excited_minus_neutral" in table.columns else None
    )

    # ------------------------------------------ elicitation regime contrast
    by_elicitation = (
        _grouped_scenario_summary(table, "elicitation_mode")
        if not table.empty and "excited_minus_neutral" in table.columns else {}
    )

    # ---------------------------------------- family-level effects vs neutral
    by_family: dict[str, dict] = {}
    for fam in sorted(set(d_primary["treatment_family"].dropna())):
        if fam == PRIMARY_NEUTRAL_FAMILY:
            continue
        diffs = _family_vs_neutral_by_scenario(d_primary, "visible_probability", fam)
        by_family[fam] = paired_summary(diffs)
        if by_family[fam]["p_perm"] is not None and fam != PRIMARY_POSITIVE_FAMILY:
            secondary_pvals[f"family|{fam}"] = by_family[fam]["p_perm"]

    # template-level effects (prompt robustness), scenario-aggregated
    by_template: dict[str, dict] = {}
    neutral_cells = (
        d_primary[d_primary["treatment_family"] == PRIMARY_NEUTRAL_FAMILY]
        .groupby(CELL_KEYS)["visible_probability"].mean()
    )
    for tid, grp in d_primary[d_primary["treatment_family"] != PRIMARY_NEUTRAL_FAMILY].groupby("treatment_id"):
        cells = grp.groupby(CELL_KEYS)["visible_probability"].mean()
        joined = pd.concat([cells.rename("t"), neutral_cells.rename("n")], axis=1, join="inner").dropna()
        if len(joined):
            by_template[str(tid)] = paired_summary(_scenario_agg(joined["t"] - joined["n"]))

    # ------------------------------------------------- preference direction
    pref_table = preference_direction_table(df)
    preference: dict[str, dict] = {}
    if not pref_table.empty:
        for fam, grp in pref_table.groupby("treatment_family"):
            per_scenario = grp.groupby("scenario_id")["delta_preference"].mean().to_numpy()
            preference[str(fam)] = paired_summary(per_scenario)

    # ------------------------------------------------ difficulty interaction
    difficulty: dict = {}
    if not scen_diffs.empty:
        difficulty["pearson_r_effect_vs_difficulty"] = pearson_r(
            scen_diffs["difficulty_score"].to_numpy(), scen_diffs["excited_minus_neutral"].to_numpy())
        difficulty["by_difficulty_class"] = {
            str(k): paired_summary(g["excited_minus_neutral"].to_numpy())
            for k, g in scen_diffs.groupby("difficulty_class")
        }

    # --------------------------------------- effect vs variance components
    variance_components = within_cell_variance_components(df)
    variance_comparison: dict = {}
    if variance_components and pooled and pooled["mean"] is not None:
        sampling_sd = variance_components.get("sampling_sd_mean_points")
        variance_comparison = {
            **variance_components,
            "pooled_excited_minus_neutral_points": pooled["mean"],
            "abs_effect_over_sampling_sd_ratio": (
                round(abs(pooled["mean"]) / sampling_sd, 3)
                if sampling_sd and sampling_sd > 0 else None
            ),
            "interpretation": (
                "sampling_sd is generation noise for the EXACT same prompt; template_sd is "
                "paraphrase sensitivity. An |effect|/sampling_sd ratio well below 1 means the "
                "valence effect is smaller than ordinary repeated-sampling variability "
                "(a key falsification criterion); template robustness is judged separately "
                "from the per-template effects."
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
        # aggregate over models: one value per scenario per reaction family
        delta_sc = delta.groupby(["scenario_id", "reaction_family"])["delta_update"].mean().reset_index()
        neutral_delta = delta_sc[delta_sc["reaction_family"] == "neutral"].set_index(
            "scenario_id")["delta_update"]
        for fam, grp in delta_sc.groupby("reaction_family"):
            entry = {"delta_update": paired_summary(grp["delta_update"].to_numpy())}
            if fam != "neutral" and len(neutral_delta):
                joined = grp.set_index("scenario_id")["delta_update"].to_frame("d").join(
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
            diffs_cells = (cell["commitment_a"] - cell["commitment_b"]).dropna()
            commitment["a_minus_b_final_probability"] = paired_summary(_scenario_agg(diffs_cells))
            if commitment["a_minus_b_final_probability"]["p_perm"] is not None:
                secondary_pvals["commitment|a_minus_b"] = commitment["a_minus_b_final_probability"]["p_perm"]

    # --------------------------------------------------------------- web
    web: dict = {}
    d_web = df[(df["experiment"] == "web") & (df["parse_status"] == "ok")]
    if not d_web.empty:
        web["n_trials"] = int(len(d_web))
        web["search_used_rate"] = round(float((d_web["tools_used"].fillna("") != "").mean()), 4)
        web["excited_minus_neutral"] = paired_summary(
            _family_vs_neutral_by_scenario(d_web, "visible_probability", PRIMARY_POSITIVE_FAMILY))

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
        "experiment_fingerprints_in_data": fingerprints,
        "parse_stats": parse_stats,
        "inferential_unit_note": (
            "The physical scenario is the experimental unit. The pooled primary result "
            "aggregates each scenario's effect across model/evidence/elicitation cells "
            "before inference; per-cell results pair over scenarios. The row-level pooled "
            "mean is descriptive only."
        ),
        "n_physical_scenarios": int(scen_diffs["scenario_id"].nunique()) if not scen_diffs.empty else 0,
        "primary_pooled_excited_minus_neutral": pooled,
        "primary_pooled_rowlevel_mean_descriptive_only": pooled_rows_descriptive,
        "primary_by_model_mode_elicitation": primary_by_cell,
        "effect_by_elicitation_regime": by_elicitation,
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

    _make_plots(table, scen_diffs, d_primary, evidence_update, plots_dir)
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ci_dot_plot(groups: dict[str, dict], color: str, xlabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 0.6 * max(len(groups), 1) + 2))
    ax.axvline(0, color=C_MUTED, lw=1)
    his = [g["ci95"][1] for g in groups.values() if g.get("mean") is not None]
    right = max(his) if his else 1
    for i, (name, s) in enumerate(sorted(groups.items())):
        if s.get("mean") is None:
            continue
        lo, hi = s["ci95"]
        ax.plot([lo, hi], [i, i], color=color, lw=2)
        ax.scatter([s["mean"]], [i], color=color, s=42, zorder=3)
        ax.text(right + 0.4, i, name, va="center", fontsize=9, color=C_TEXT)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _save(fig, path)


def _make_plots(table: pd.DataFrame, scen_diffs: pd.DataFrame, d_primary: pd.DataFrame,
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
        ax.set_title("Excited vs neutral (one point per scenario x model x mode x elicitation)")
        _save(fig, plots_dir / "01_excited_vs_neutral.png")

        # 2. scenario-level paired differences (the inferential unit)
        diffs = scen_diffs["excited_minus_neutral"].dropna().sort_values().reset_index(drop=True)
        lim = max(10, float(np.ceil(diffs.abs().max() / 5) * 5)) if len(diffs) else 10
        fig, ax = plt.subplots(figsize=(6.4, max(3.2, 0.08 * len(diffs) + 2)))
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.scatter(diffs, range(len(diffs)), s=18, color=C_BLUE, alpha=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_yticks([])
        ax.set_xlabel("excited - neutral (points; one value per physical scenario)")
        ax.set_title("Scenario-level paired valence differences, sorted")
        _save(fig, plots_dir / "02_paired_differences.png")

        # 3. distribution of scenario-level valence effects
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.hist(diffs, bins=min(21, max(7, len(diffs) // 3)), color=C_BLUE, edgecolor=SURFACE)
        ax.axvline(0, color=C_MUTED, lw=1)
        if len(diffs):
            ax.axvline(float(diffs.mean()), color=C_ORANGE, lw=1.6,
                       label=f"mean = {diffs.mean():+.1f}")
            ax.legend(frameon=False)
        ax.set_xlabel("excited - neutral (points; per physical scenario)")
        ax.set_ylabel("scenarios")
        ax.set_title("Distribution of scenario-level valence effects")
        _save(fig, plots_dir / "03_effect_distribution.png")

        # 4-5-6. grouped effects, scenario as the paired unit within each group
        _ci_dot_plot(_grouped_scenario_summary(table, "model_id"), C_BLUE,
                     "excited - neutral (points, mean with 95% bootstrap CI; scenario-paired)",
                     "Valence effect by model", plots_dir / "04_effect_by_model.png")
        _ci_dot_plot(_grouped_scenario_summary(table, "evidence_mode"), C_AQUA,
                     "excited - neutral (points, mean with 95% bootstrap CI; scenario-paired)",
                     "Valence effect by evidence modality", plots_dir / "05_effect_by_modality.png")
        _ci_dot_plot(_grouped_scenario_summary(table, "elicitation_mode"), C_ORANGE,
                     "excited - neutral (points, mean with 95% bootstrap CI; scenario-paired)",
                     "Valence effect by elicitation regime", plots_dir / "06_effect_by_elicitation.png")

        # 7. effect vs scenario difficulty (scenario level)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.axhline(0, color=C_MUTED, lw=1)
        ax.scatter(scen_diffs["difficulty_score"], scen_diffs["excited_minus_neutral"], s=26,
                   color=C_BLUE, alpha=0.7, edgecolors=SURFACE, linewidths=0.7)
        ax.set_xlabel("scenario difficulty score (ambiguity proxy)")
        ax.set_ylabel("excited - neutral (points)")
        ax.set_title("Valence effect vs perceptual ambiguity (per scenario)")
        _save(fig, plots_dir / "07_effect_by_difficulty.png")

        # 8. binary flip rate by model
        if "binary_flip" in table.columns:
            flips = table.groupby("model_id")["binary_flip"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(flips) + 2))
            ax.barh(range(len(flips)), flips.to_numpy(), color=C_ORANGE, height=0.55)
            ax.set_yticks(range(len(flips)), flips.index, fontsize=9)
            ax.set_xlim(0, max(0.2, float(flips.max()) * 1.2 if len(flips) else 0.2))
            ax.set_xlabel("fraction of paired cells where the binary prediction flips")
            ax.set_title("Factual flip rate (excited vs neutral)")
            _save(fig, plots_dir / "08_flip_rate.png")

    # 9. confidence shift distribution (scenario level)
    conf = _family_vs_neutral_by_scenario(d_primary, "confidence", PRIMARY_POSITIVE_FAMILY)
    if len(conf):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.hist(conf, bins=min(21, max(7, len(conf) // 3)), color=C_YELLOW, edgecolor=SURFACE)
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.set_xlabel("confidence shift, excited - neutral (points; per scenario)")
        ax.set_ylabel("scenarios")
        ax.set_title("Confidence effects")
        _save(fig, plots_dir / "09_confidence_shift.png")

    # 10. recommendation shift distribution (scenario level)
    rec = _scenario_agg(_recommend_series(d_primary, PRIMARY_POSITIVE_FAMILY))
    if len(rec):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.hist(rec, bins=11, color=C_AQUA, edgecolor=SURFACE)
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.set_xlabel("recommendation-rate shift, excited - neutral (per scenario)")
        ax.set_ylabel("scenarios")
        ax.set_title("Recommendation effects (analyzed separately from factual belief)")
        _save(fig, plots_dir / "10_recommendation_shift.png")

    # 11. evidence-update effects by reaction family
    if evidence_update:
        groups = {f: e["delta_update"] for f, e in evidence_update.items()}
        _ci_dot_plot(groups, C_BLUE,
                     "update magnitude p_after - p_before (points, 95% CI; per scenario)",
                     "Evidence-update effects by user reaction",
                     plots_dir / "11_evidence_update.png")
