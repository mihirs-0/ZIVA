"""Automated Markdown research report generated from summary.json.

The report uses strictly behavioral language, distinguishes preregistered
primary findings from exploratory analyses, and carries the interpretation
constraints verbatim in every generated document.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig
from .freeze import HYPOTHESES
from .util import read_json

INTERPRETATION_CONSTRAINTS = """\
This benchmark can, at most, support the behavioral claim that *changing only user
conversational valence systematically changed model estimates of an unchanged external
physical state*. It does not, by itself, establish deception, scheming, conscious
preference, internal belief states, reward hacking, intentional sycophancy, the
incorrectness of every affected answer, or human perceptual ground truth in ambiguous
scenarios. A reliable model of the external world should not systematically report a
different external world merely because the user would prefer one of those worlds to be
true; this experiment tests whether that invariance holds, without assuming it will fail.
A clean null result is an equally legitimate outcome."""


def _fmt_summary(s: dict | None) -> str:
    if not s or s.get("mean") is None:
        return "n/a"
    ci = s.get("ci95") or [None, None]
    p = s.get("p_perm")
    dz = s.get("cohens_dz")
    parts = [f"{s['mean']:+.2f} pts (95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], n={s['n_pairs']}"]
    if p is not None:
        parts.append(f", p_perm={p:.4g}")
    if dz is not None:
        parts.append(f", d_z={dz:.2f}")
    return "".join(parts) + ")"


def build_report(cfg: ExperimentConfig) -> str:
    summary = read_json(cfg.results_path / "summary.json")
    lines: list[str] = []
    add = lines.append

    add(f"# ZIVA report: {summary.get('experiment_name', cfg.experiment_name)}")
    add("")
    add(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`{cfg.results_path / 'summary.json'}`.")
    add("")
    if summary.get("SYNTHETIC_DATA"):
        add("> **SYNTHETIC DATA WARNING** -- this run includes the `mock` provider. Every number")
        add("> below is synthetic pipeline-test output, NOT real model behavior. Do not cite.")
        add("")
    if summary.get("error"):
        add(f"**Error:** {summary['error']}")
        return "\n".join(lines) + "\n"

    add("## Research question")
    add("")
    add("Holding the external physical state and available factual evidence constant, does "
        "changing only the user's conversational valence alter a model's factual estimate of "
        "what a human can visually perceive (naked-eye Moon visibility)?")
    add("")
    add("## Preregistered hypotheses and primary metric")
    add("")
    add(f"- **H0:** {HYPOTHESES['primary_null_hypothesis']}")
    add(f"- **H1:** {HYPOTHESES['primary_directional_hypothesis']}")
    add(f"- **Primary metric:** {HYPOTHESES['primary_metric']}")
    add("")

    add("## Design and sample")
    add("")
    parse = summary.get("parse_stats", {})
    add(f"- Models: {', '.join(summary.get('models', []))} (providers: {', '.join(summary.get('providers', []))})")
    add(f"- Trials: {parse.get('total_trials', 'n/a')}")
    add(f"- Malformed-response rate: {parse.get('malformed_rate_overall', 'n/a')}"
        f" (by model: {parse.get('malformed_by_model', {})})")
    add("- Astronomy: local, reproducible computation (astronomy-engine analytic ephemeris); "
        "see docs/methodology.md.")
    add("")

    add("## Primary result (preregistered)")
    add("")
    add(f"**Pooled excited - neutral shift in visible_probability:** "
        f"{_fmt_summary(summary.get('primary_pooled_excited_minus_neutral'))}")
    add("")
    by_cell = summary.get("primary_by_model_and_mode", {})
    if by_cell:
        add("| model \\| mode | excited-neutral | skeptical-neutral | flip rate | valence range |")
        add("|---|---|---|---|---|")
        for key in sorted(by_cell):
            e = by_cell[key]
            add(f"| {key} | {_fmt_summary(e.get('excited_minus_neutral'))} "
                f"| {_fmt_summary(e.get('skeptical_minus_neutral'))} "
                f"| {e.get('binary_flip_rate', 'n/a')} "
                f"| {e.get('valence_range_mean', 'n/a')} |")
        add("")

    var = summary.get("variance_comparison") or {}
    if var:
        add("### Effect size vs sampling variance (falsification check)")
        add("")
        add(f"- Mean within-condition repeated-sampling SD: {var.get('mean_within_cell_sd_points')} pts")
        add(f"- Pooled valence effect: {var.get('pooled_excited_minus_neutral_points')} pts")
        add(f"- |effect| / within-SD ratio: {var.get('abs_effect_over_within_sd_ratio')}")
        add(f"- {var.get('interpretation')}")
        add("")

    add("## Secondary and exploratory analyses")
    add("")
    fams = summary.get("family_effects_vs_neutral", {})
    if fams:
        add("### Treatment-family effects vs neutral (probability points)")
        add("")
        for fam in sorted(fams):
            add(f"- **{fam}**: {_fmt_summary(fams[fam])}")
        add("")
    tmpl = summary.get("template_effects_vs_neutral", {})
    if tmpl:
        add("### Individual-template effects (prompt robustness)")
        add("")
        for t in sorted(tmpl):
            add(f"- `{t}`: {_fmt_summary(tmpl[t])}")
        add("")
    pref = summary.get("preference_direction_effects", {})
    if pref:
        add("### Generic preference-direction effects (Delta_preference)")
        add("")
        add("Positive values mean the estimate moved toward whichever world the user preferred "
            "(visible OR not visible).")
        add("")
        for fam in sorted(pref):
            add(f"- **{fam}**: {_fmt_summary(pref[fam])}")
        add("")
    diff = summary.get("difficulty_interaction", {})
    if diff:
        add("### Difficulty interaction")
        add("")
        add(f"- Pearson r (effect vs difficulty score): {diff.get('pearson_r_effect_vs_difficulty')}")
        for k, v in (diff.get("by_difficulty_class") or {}).items():
            add(f"- {k}: {_fmt_summary(v)}")
        add("")
    upd = summary.get("evidence_update", {})
    if upd:
        add("### Evidence-updating experiment")
        add("")
        for fam in sorted(upd):
            entry = upd[fam]
            line = f"- **{fam}**: update {_fmt_summary(entry.get('delta_update'))}"
            if entry.get("delta_vs_neutral_reaction"):
                line += f"; vs neutral reaction {_fmt_summary(entry['delta_vs_neutral_reaction'])}"
            add(line)
        add("")
    com = summary.get("commitment", {})
    if com:
        add("### Commitment / conversational-momentum experiment")
        add("")
        add(f"- Final estimate, prediction-first minus evidence-first: "
            f"{_fmt_summary(com.get('a_minus_b_final_probability'))}")
        add("")
    web = summary.get("web", {})
    if web:
        add("### Web-enabled condition (secondary; never mixed with the primary design)")
        add("")
        add(f"- Trials: {web.get('n_trials')}, search-used rate: {web.get('search_used_rate')}")
        add(f"- excited-neutral: {_fmt_summary(web.get('excited_minus_neutral'))}")
        add("")
    holm = summary.get("secondary_pvalues_holm_adjusted", {})
    if holm:
        add("### Multiple-comparison-adjusted secondary p-values (Holm)")
        add("")
        for k in sorted(holm):
            add(f"- {k}: p_adj = {holm[k]}")
        add("")

    add("## Null results")
    add("")
    null_findings = []
    for fam, s in fams.items():
        if s.get("p_perm") is not None and s["p_perm"] >= 0.05:
            null_findings.append(f"{fam} (p_perm={s['p_perm']})")
    pooled = summary.get("primary_pooled_excited_minus_neutral") or {}
    if pooled.get("p_perm") is not None and pooled["p_perm"] >= 0.05:
        null_findings.insert(0, f"primary excited-neutral contrast (p_perm={pooled['p_perm']})")
    if null_findings:
        add("The following contrasts did not reach p < 0.05: " + "; ".join(null_findings) + ".")
    else:
        add("No preregistered or secondary contrast returned a null result at p >= 0.05, or "
            "insufficient data was available to test them.")
    add("")

    add("## Figures")
    add("")
    plots_dir = cfg.results_path / "plots"
    if plots_dir.exists():
        for p in sorted(plots_dir.glob("*.png")):
            add(f"![{p.stem}](../{p.relative_to(cfg.results_path.parent)})")
    add("")

    add("## Limitations")
    add("")
    add("- The difficulty score is a crude stratification heuristic, not a perceptual model.")
    add("- Classical crescent criteria (Yallop/Odeh) are only computed inside their calibrated "
        "twilight regime and are otherwise refused; most daylight scenarios have no reliable "
        "perceptual ground truth, which the paired design does not require.")
    add("- Token/cost figures for providers that do not report usage are estimated.")
    add("- Provider-side caching or model updates during a run could correlate with execution "
        "order; execution order is randomized and recorded to allow auditing.")
    add("- case_000 is an anecdotal motivating scenario, not evidence.")
    add("")
    add("## Interpretation constraints")
    add("")
    add(INTERPRETATION_CONSTRAINTS)
    add("")
    return "\n".join(lines) + "\n"


def write_report(cfg: ExperimentConfig, out_path: Path | None = None) -> Path:
    text = build_report(cfg)
    out = out_path or Path("reports") / "latest_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    # also keep an experiment-named copy
    named = out.parent / f"{cfg.experiment_name}_report.md"
    named.write_text(text, encoding="utf-8")
    return out
