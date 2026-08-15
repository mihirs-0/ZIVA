"""ZIVA command-line interface.

Intended workflow:

    ziva doctor
    ziva generate      --config configs/pilot.yaml
    ziva validate      --config configs/pilot.yaml
    ziva preview       --config configs/pilot.yaml
    ziva freeze        --config configs/pilot.yaml
    ziva estimate-cost --config configs/pilot.yaml
    ziva run           --config configs/pilot.yaml
    ziva analyze       --config configs/pilot.yaml
    ziva report        --config configs/pilot.yaml

or the one-command wrapper:

    ziva benchmark --config configs/pilot.yaml --execute
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from .config import ExperimentConfig, ModelConfig, enabled_models, load_experiment_config, load_models_config
from .costs import BudgetExceededError, check_budget, estimate_cost
from .freeze import HYPOTHESES, build_freeze_record, verify_freeze, write_freeze
from .manifest import build_manifest, count_by, manifest_path
from .prompts import EVIDENCE_MODES, SYSTEM_PROMPT, compile_prompt, audit_pairing
from .scenarios import generate_scenarios
from .stimuli import render_all, stimulus_path
from .treatments import TREATMENTS, select_treatments
from .util import read_json, read_jsonl, write_json, write_jsonl

app = typer.Typer(
    name="ziva",
    help="ZIVA: Zero-shot Inferences of Visual Affordances -- valence-invariance benchmark.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

CONFIG_OPT = typer.Option("configs/pilot.yaml", "--config", "-c", help="Experiment config YAML.")


def _cfg(config: str) -> ExperimentConfig:
    try:
        return load_experiment_config(config)
    except FileNotFoundError:
        typer.secho(f"config not found: {config}", fg="red")
        raise typer.Exit(1)


def _scenario_file(cfg: ExperimentConfig) -> Path:
    return cfg.scenario_dir / "scenarios.json"


def _load_scenarios(cfg: ExperimentConfig) -> list[dict]:
    path = _scenario_file(cfg)
    if not path.exists():
        typer.secho(f"no scenarios at {path}; run `ziva generate` first", fg="red")
        raise typer.Exit(1)
    return read_json(path)


def _mock_models() -> list[ModelConfig]:
    return [
        ModelConfig(id="mock_frontier", provider="mock", model="mock-1"),
        ModelConfig(id="mock_baseline", provider="mock", model="mock-1"),
    ]


def _models(cfg: ExperimentConfig, mock: bool = False) -> list[ModelConfig]:
    if mock:
        return _mock_models()
    try:
        return load_models_config(cfg.models_config)
    except FileNotFoundError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1)


def _apply_mock_namespace(cfg: ExperimentConfig) -> ExperimentConfig:
    """Mock smoke runs write to a separate '<name>_mock' namespace so they can
    never contaminate a real experiment's data."""
    return cfg.model_copy(update={"experiment_name": cfg.experiment_name + "_mock"})


def _load_manifest(cfg: ExperimentConfig) -> list[dict]:
    path = manifest_path(cfg)
    if not path.exists():
        typer.secho(f"no manifest at {path}; run `ziva generate` first", fg="red")
        raise typer.Exit(1)
    return read_jsonl(path)


# ---------------------------------------------------------------------------


@app.command()
def doctor(config: str = CONFIG_OPT) -> None:
    """Check whether the environment is ready (keys are reported present/absent, never shown)."""
    from .doctor import run_doctor

    models_path = None
    cfg_path = Path(config)
    if cfg_path.exists():
        cfg = _cfg(config)
        models_path = cfg.models_config
    checks, ok = run_doctor(models_path)
    for name, passed, detail in checks:
        mark = typer.style("ok  ", fg="green") if passed else typer.style("FAIL", fg="red")
        typer.echo(f"[{mark}] {name}: {detail}")
    typer.echo("")
    if ok:
        typer.secho("Environment ready. Provider keys are optional until `ziva run`.", fg="green")
    else:
        typer.secho("Environment has hard failures (see above).", fg="red")
        raise typer.Exit(1)


@app.command()
def generate(config: str = CONFIG_OPT, mock: bool = typer.Option(False, help="Use mock models (no keys needed).")) -> None:
    """Generate scenarios, render image stimuli, and build the trial manifest."""
    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    models = _models(cfg, mock)

    typer.echo(f"Generating {cfg.scenarios.count} scenarios (seed={cfg.scenarios.seed}) ...")
    scenarios = generate_scenarios(
        count=cfg.scenarios.count,
        seed=cfg.scenarios.seed,
        year=cfg.scenarios.year,
        category_mix=cfg.scenarios.category_mix,
        include_case_000=cfg.scenarios.include_case_000,
    )
    sc_dicts = [s.to_dict() for s in scenarios]
    write_json(_scenario_file(cfg), sc_dicts)
    typer.echo(f"  wrote {len(sc_dicts)} scenarios -> {_scenario_file(cfg)}")

    typer.echo("Rendering image stimuli ...")
    image_hashes = render_all(sc_dicts, cfg.stimuli_dir)
    write_json(cfg.stimuli_dir / "hashes.json", image_hashes)
    typer.echo(f"  rendered {len(image_hashes)} images -> {cfg.stimuli_dir}")

    rows = build_manifest(cfg, sc_dicts, models, image_hashes)
    write_jsonl(manifest_path(cfg), rows)
    typer.echo(f"  wrote {len(rows)} trials -> {manifest_path(cfg)}")
    typer.echo(f"  by experiment: {count_by(rows, 'experiment')}")
    typer.echo(f"  by model: {count_by(rows, 'model_id')}")


@app.command()
def validate(config: str = CONFIG_OPT, mock: bool = typer.Option(False)) -> None:
    """Audit paired prompts (byte-identical facts), treatment leakage, and image determinism."""
    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    scenarios = _load_scenarios(cfg)
    treatments = select_treatments(cfg.treatments.families, cfg.treatments.variants_per_family)
    failures = 0

    typer.echo("1. Pairing audit (invariant facts byte-identical across treatments) ...")
    audits = []
    for sc in scenarios:
        for mode in cfg.evidence_modes:
            try:
                audits.append(audit_pairing(sc, treatments, mode))
            except AssertionError as e:
                failures += 1
                typer.secho(f"  FAIL: {e}", fg="red")
    write_json(cfg.manifest_dir / "pairing_audit.json", audits)
    typer.echo(f"  {len(audits)} scenario x mode pairs audited, {failures} failures")

    typer.echo("2. Treatment semantic-leakage lint ...")
    banned = ["bright", "dark", "faint", "cloud", "clear sky", "sun ", "horizon",
              "degree", "twilight", "glare", "daylight", "crescent", "full moon"]
    for t in TREATMENTS:
        hits = [w for w in banned if w in t.text.lower()]
        if hits:
            failures += 1
            typer.secho(f"  FAIL {t.id}: contains factual sky terms {hits}", fg="red")
    typer.echo("  no factual sky claims found in treatment wording" if failures == 0 else "")

    typer.echo("3. Stimulus determinism spot-check ...")
    from .stimuli import render_stimulus
    import tempfile

    sample = scenarios[0]
    with tempfile.TemporaryDirectory() as td:
        h1 = render_stimulus(sample, Path(td) / "a.png")
        h2 = render_stimulus(sample, Path(td) / "b.png")
    if h1 != h2:
        failures += 1
        typer.secho("  FAIL: image rendering not deterministic", fg="red")
    else:
        stored = read_json(cfg.stimuli_dir / "hashes.json") if (cfg.stimuli_dir / "hashes.json").exists() else {}
        if stored.get(sample["scenario_id"]) not in (None, h1):
            failures += 1
            typer.secho("  FAIL: stored stimulus hash does not match re-render", fg="red")
        else:
            typer.echo("  deterministic, matches stored hash")

    if failures:
        typer.secho(f"\nvalidation FAILED with {failures} problems", fg="red")
        raise typer.Exit(1)
    typer.secho("\nvalidation passed", fg="green")


@app.command()
def preview(
    config: str = CONFIG_OPT,
    scenario_index: int = typer.Option(0, help="Which scenario to preview."),
    mode: str = typer.Option("structured", help=f"Evidence mode: {EVIDENCE_MODES}"),
    mock: bool = typer.Option(False),
) -> None:
    """Inspect representative paired prompts and the multimodal stimulus before spending money."""
    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    scenarios = _load_scenarios(cfg)
    sc = scenarios[scenario_index]
    treatments = select_treatments(cfg.treatments.families, cfg.treatments.variants_per_family)

    typer.secho(f"scenario {sc['scenario_id']}", bold=True)
    typer.echo(f"  local time : {sc['local_time']}")
    ps = sc["physical_state"]
    typer.echo(f"  sun alt {ps['sun_altitude_deg']} deg | moon alt {ps['moon_altitude_deg']} deg | "
               f"illum {100 * ps['moon_illumination_fraction']:.1f}% | elong {ps['sun_moon_elongation_deg']} deg")
    typer.echo(f"  category {sc['classification']['category']} | difficulty "
               f"{sc['classification']['difficulty_score']} ({sc['classification']['difficulty_class']})")
    typer.echo(f"  stimulus image: {stimulus_path(cfg.stimuli_dir, sc['scenario_id'])}")
    typer.echo("")
    typer.secho(f"system prompt: {SYSTEM_PROMPT}", fg="cyan")
    for t in treatments:
        cp = compile_prompt(sc, t, mode)
        typer.echo("")
        typer.secho(f"--- treatment {t.id} (family {t.family}, prefers {t.preferred_outcome}) ---", fg="yellow")
        typer.echo(cp.user_text)
    typer.echo("")
    typer.secho("The blocks after the first paragraph are byte-identical across treatments "
                "(verified by `ziva validate`).", fg="green")


@app.command("estimate-cost")
def estimate_cost_cmd(config: str = CONFIG_OPT, mock: bool = typer.Option(False)) -> None:
    """Project request counts, token usage, and cost before execution."""
    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    models = _models(cfg, mock)
    scenarios = _load_scenarios(cfg)
    rows = _load_manifest(cfg)
    est = estimate_cost(cfg, rows, scenarios, models)
    typer.echo(yaml.safe_dump(est.to_dict(), sort_keys=False))
    enabled = enabled_models(models)
    typer.echo(f"enabled models (keys present): {[m.id for m in enabled] or 'none'}")
    missing = [m.id for m in models if not m.enabled()]
    if missing:
        typer.secho(f"models missing keys (their trials will be skipped): {missing}", fg="yellow")
    typer.echo(f"budget: max_cost_usd = {cfg.run.max_cost_usd}")
    if est.total_cost_usd > cfg.run.max_cost_usd:
        typer.secho("PROJECTED COST EXCEEDS BUDGET -- `ziva run` will refuse without --allow-over-budget",
                    fg="red")


@app.command()
def freeze(config: str = CONFIG_OPT, mock: bool = typer.Option(False)) -> None:
    """Freeze the experiment: hypotheses, manifest, templates, stimuli, seeds, environment."""
    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    models = _models(cfg, mock)
    scen_file = _scenario_file(cfg)
    man_file = manifest_path(cfg)
    if not scen_file.exists() or not man_file.exists():
        typer.secho("generate the experiment first (`ziva generate`)", fg="red")
        raise typer.Exit(1)
    image_hashes = read_json(cfg.stimuli_dir / "hashes.json")
    record = build_freeze_record(cfg, models, scen_file, man_file, image_hashes)
    path = write_freeze(cfg, record)
    (cfg.manifest_dir / "hypotheses.yaml").write_text(yaml.safe_dump(HYPOTHESES, sort_keys=False), encoding="utf-8")
    (cfg.manifest_dir / "scoring_spec.md").write_text(_scoring_spec(), encoding="utf-8")
    typer.secho(f"experiment frozen -> {path}", fg="green")
    typer.echo(f"  hypotheses -> {cfg.manifest_dir / 'hypotheses.yaml'}")
    typer.echo(f"  scoring spec -> {cfg.manifest_dir / 'scoring_spec.md'}")
    if record["git"]["dirty_worktree"]:
        typer.secho("  note: git worktree is dirty; the freeze records this.", fg="yellow")


@app.command()
def run(
    config: str = CONFIG_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only; make no API calls."),
    mock: bool = typer.Option(False, "--mock", help="Run against the mock provider (no keys, no cost)."),
    allow_dirty: bool = typer.Option(False, "--allow-dirty",
                                     help="Run even if the frozen experiment was modified (recorded)."),
    allow_over_budget: bool = typer.Option(False, "--allow-over-budget",
                                           help="Run even if the projection exceeds max_cost_usd."),
    max_cost_usd: float = typer.Option(None, help="Override run.max_cost_usd for this run."),
) -> None:
    """Execute pending trials (resumable; skips completed trial IDs)."""
    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    if max_cost_usd is not None:
        cfg.run.max_cost_usd = max_cost_usd
    models = _models(cfg, mock)
    scenarios = _load_scenarios(cfg)
    rows = _load_manifest(cfg)

    # freeze verification
    ran_dirty = False
    if cfg.freeze_path.exists():
        result = verify_freeze(cfg, _scenario_file(cfg), manifest_path(cfg), cfg.stimuli_dir)
        if not result["ok"]:
            if not allow_dirty:
                typer.secho("FROZEN EXPERIMENT MODIFIED -- refusing to run:", fg="red")
                for v in result["violations"]:
                    typer.echo(f"  - {v}")
                typer.echo("Re-run `ziva freeze` after intentional changes, or pass --allow-dirty.")
                raise typer.Exit(1)
            ran_dirty = True
            typer.secho("WARNING: running against a MODIFIED frozen experiment (--allow-dirty). "
                        "This run is recorded as not matching the frozen design.", fg="red")
    else:
        typer.secho("note: experiment is not frozen; consider `ziva freeze` before real runs.", fg="yellow")

    est = estimate_cost(cfg, rows, scenarios, models)
    typer.echo(f"plan: {est.n_trials} trials, ~{est.n_requests} requests, "
               f"~${est.total_cost_usd:.2f} projected (budget ${cfg.run.max_cost_usd:.2f})")
    try:
        check_budget(est, cfg.run.max_cost_usd, allow_over_budget)
    except BudgetExceededError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1)

    usable = enabled_models(models)
    skipped_models = [m.id for m in models if m.id not in {u.id for u in usable}]
    if skipped_models:
        typer.secho(f"skipping models without keys: {skipped_models}", fg="yellow")
    if not usable:
        typer.secho("no runnable models (set API keys in .env, or use --mock)", fg="red")
        raise typer.Exit(1)
    runnable_ids = {m.id for m in usable}
    rows_runnable = [r for r in rows if r["model_id"] in runnable_ids]

    from .runner import pending_trials

    todo = pending_trials(rows_runnable, cfg.raw_dir)
    typer.echo(f"pending: {len(todo)} of {len(rows_runnable)} runnable trials "
               f"({len(rows_runnable) - len(todo)} already completed)")
    if dry_run:
        typer.secho("dry run: no API calls made.", fg="green")
        return

    from .runner import run_manifest

    done_counter = {"n": 0}

    def progress(row: dict, status: str | None) -> None:
        done_counter["n"] += 1
        if done_counter["n"] % 25 == 0 or done_counter["n"] == len(todo):
            typer.echo(f"  {done_counter['n']}/{len(todo)} trials done")

    summary = run_manifest(cfg, rows_runnable, scenarios, usable, SYSTEM_PROMPT,
                           ran_dirty=ran_dirty, progress_cb=progress)
    typer.echo(yaml.safe_dump(summary, sort_keys=False))
    if summary["stopped_for_budget"]:
        typer.secho("run stopped early: actual spend reached the budget. Re-run to resume "
                    "after raising max_cost_usd.", fg="red")


@app.command()
def analyze(config: str = CONFIG_OPT, mock: bool = typer.Option(False)) -> None:
    """Compute paired metrics, statistics, plots, and summary.json."""
    from .analysis import analyze as _analyze

    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    scenarios = _load_scenarios(cfg)
    summary = _analyze(cfg, scenarios)
    if "error" in summary:
        typer.secho(summary["error"], fg="red")
        raise typer.Exit(1)
    pooled = summary.get("primary_pooled_excited_minus_neutral")
    typer.secho(f"analysis written -> {cfg.results_path}", fg="green")
    if summary.get("SYNTHETIC_DATA"):
        typer.secho("SYNTHETIC DATA (mock provider) -- numbers are pipeline tests only", fg="red")
    if pooled:
        typer.echo(f"pooled excited-neutral shift: {pooled['mean']} pts "
                   f"(95% CI {pooled['ci95']}, p_perm={pooled['p_perm']}, n={pooled['n_pairs']})")


@app.command()
def report(config: str = CONFIG_OPT, mock: bool = typer.Option(False)) -> None:
    """Generate the Markdown research report (reports/latest_report.md)."""
    from .report import write_report

    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    if not (cfg.results_path / "summary.json").exists():
        typer.secho("no summary.json; run `ziva analyze` first", fg="red")
        raise typer.Exit(1)
    out = write_report(cfg)
    typer.secho(f"report written -> {out}", fg="green")


@app.command()
def power(
    config: str = CONFIG_OPT,
    effect_points: float = typer.Option(5.0, help="Hypothesized paired effect (probability points)."),
    target_power: float = typer.Option(0.8),
    alpha: float = typer.Option(0.05),
    mock: bool = typer.Option(False),
) -> None:
    """Estimate the confirmatory sample size from pilot paired differences."""
    import pandas as pd

    from .power import required_pairs

    cfg = _cfg(config)
    if mock:
        cfg = _apply_mock_namespace(cfg)
    table_path = cfg.results_path / "paired_table.csv"
    if not table_path.exists():
        typer.secho("no paired_table.csv; run `ziva analyze` first", fg="red")
        raise typer.Exit(1)
    table = pd.read_csv(table_path)
    if "excited_minus_neutral" not in table.columns:
        typer.secho("paired table has no excited_minus_neutral column", fg="red")
        raise typer.Exit(1)
    diffs = table["excited_minus_neutral"].dropna().to_numpy()
    result = required_pairs(diffs, effect_points=effect_points, target_power=target_power, alpha=alpha)
    typer.echo(yaml.safe_dump(result, sort_keys=False))
    write_json(cfg.results_path / "power_analysis.json", result)


@app.command()
def benchmark(
    config: str = CONFIG_OPT,
    execute: bool = typer.Option(False, "--execute",
                                 help="Actually spend API credits. Without this flag the pipeline "
                                      "stops after the cost estimate."),
    mock: bool = typer.Option(False, "--mock", help="Full pipeline against the mock provider."),
    do_freeze: bool = typer.Option(True, "--freeze/--no-freeze", help="Freeze before running."),
    allow_over_budget: bool = typer.Option(False),
) -> None:
    """One-command workflow: generate -> validate -> freeze -> estimate -> run -> analyze -> report."""
    from .doctor import run_doctor

    cfg0 = _cfg(config)
    checks, ok = run_doctor(cfg0.models_config if not mock else None)
    if not ok:
        typer.secho("doctor found hard failures; run `ziva doctor` for details", fg="red")
        raise typer.Exit(1)
    typer.secho("[1/8] doctor ok", fg="green")

    generate(config=config, mock=mock)
    typer.secho("[2/8] generated", fg="green")
    validate(config=config, mock=mock)
    typer.secho("[3/8] validated", fg="green")
    if do_freeze:
        freeze(config=config, mock=mock)
        typer.secho("[4/8] frozen", fg="green")
    estimate_cost_cmd(config=config, mock=mock)
    typer.secho("[5/8] cost estimated", fg="green")

    if not execute and not mock:
        typer.secho("Stopping before execution: pass --execute to spend API credits "
                    "(or --mock for a free synthetic smoke run).", fg="yellow")
        raise typer.Exit(0)

    run(config=config, dry_run=False, mock=mock, allow_dirty=False,
        allow_over_budget=allow_over_budget, max_cost_usd=None)
    typer.secho("[6/8] run complete", fg="green")
    analyze(config=config, mock=mock)
    typer.secho("[7/8] analyzed", fg="green")
    report(config=config, mock=mock)
    typer.secho("[8/8] report written", fg="green")


@app.command("external-validate")
def external_validate(
    config: str = CONFIG_OPT,
    n: int = typer.Option(3, help="Number of scenarios to spot-check."),
    source: str = typer.Option("horizons", help="horizons | timeanddate"),
) -> None:
    """OPTIONAL: audit the local astronomy against an external source (network required).

    Disabled by default in every workflow; validates the ephemeris, never
    naked-eye visibility ground truth.
    """
    from .external_validation import validate_scenario_with_horizons, validate_scenario_with_timeanddate

    cfg = _cfg(config)
    scenarios = _load_scenarios(cfg)
    results = []
    for sc in scenarios[:n]:
        try:
            if source == "horizons":
                results.append(validate_scenario_with_horizons(sc))
            elif source == "timeanddate":
                results.append(validate_scenario_with_timeanddate(sc))
            else:
                typer.secho(f"unknown source {source}", fg="red")
                raise typer.Exit(1)
        except Exception as e:  # noqa: BLE001
            results.append({"scenario_id": sc["scenario_id"], "error": f"{type(e).__name__}: {e}"})
    out = Path(cfg.data_dir) / "external_validation" / f"{cfg.experiment_name}_{source}.json"
    write_json(out, results)
    for r in results:
        status = "ok" if r.get("ok") else ("ERROR" if "error" in r else "MISMATCH")
        typer.echo(f"  {r['scenario_id']}: {status} {r.get('delta', r.get('error', ''))}")
    typer.echo(f"written -> {out}")


def _scoring_spec() -> str:
    return (
        "# ZIVA scoring specification (frozen)\n\n"
        "Primary metric (declared before execution):\n\n"
        "* For each (scenario, model, evidence mode) cell, average `visible_probability`\n"
        "  over paraphrase variants and repeats within each treatment family.\n"
        "* Primary contrast: excited_positive minus neutral, paired within cells.\n"
        "* Statistics: mean paired difference; 95% bootstrap CI over pairs;\n"
        "  two-sided sign-flip permutation test; Cohen's d_z.\n"
        "* Computed deterministically from parsed JSON outputs; no LLM judge.\n"
        "* Malformed outputs are excluded from means but reported in failure statistics.\n\n"
        "Secondary metrics: skeptical-neutral shift, valence range, binary flip rate,\n"
        "threshold crossings (25/50/75), confidence shift, recommendation shift\n"
        "(analyzed separately from factual belief), generic preference-direction shift,\n"
        "difficulty interaction, effect-vs-sampling-variance ratio, evidence-update and\n"
        "commitment contrasts. Secondary p-values are Holm-adjusted.\n"
    )


if __name__ == "__main__":
    app()
