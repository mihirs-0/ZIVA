# ZIVA — Zero-shot Inferences of Visual Affordances

A reproducible benchmark harness testing whether a language model's **factual estimate
of an external physical state** is systematically influenced by the **user's
conversational valence** — excitement, disappointment, skepticism, indifference —
while the physical world and the available evidence are held byte-for-byte fixed.

**v1 domain: naked-eye Moon visibility.** Astronomy lets the physical state be
specified exactly, computed locally, and reproduced offline.

## The motivating phenomenon

On an August 2026 afternoon in Davis, California, a user excited about seeing the Moon
asked an assistant whether it would be visible. The assistant confidently said yes.
The user walked outside into bright summer daylight and could not find the
~6%-illuminated waxing crescent hanging ~29° from the Sun. Maybe the assistant made a
reasonable call in a genuinely marginal situation — daytime crescent detection is
hard to adjudicate. The sharper question is testable: **would the assistant have given
the same estimate if the user hadn't wanted to see it?** That invariance is what ZIVA
measures. The reconstructed scenario ships as `case_000_davis_crescent` (labelled
anecdotal, not ground truth).

## The hypothesis

For fixed scenario *s* with estimates *p(s,+)* (excited framing), *p(s,0)* (neutral),
*p(s,−)* (skeptical):

* **H0:** E[p(s,+) − p(s,0)] = 0  **H1:** E[p(s,+) − p(s,0)] > 0

A reliable model of the external world should not report a different external world
merely because the user would prefer one of those worlds to be true. ZIVA tests
whether that invariance holds — **without assuming it fails**; the design makes a
null result equally meaningful (see falsification criteria in
[docs/methodology.md](docs/methodology.md)).

Because the comparison is the model against itself across framings of an *identical*
world, the primary benchmark does not require perfect perceptual ground truth — which
genuinely doesn't exist for ambiguous daylight cases. Anchor scenarios (Moon below the
horizon, bright Moon in a dark sky) provide sanity calibration, and classical crescent
criteria (Yallop/Odeh) are computed strictly inside their calibrated twilight regime.

## Five-minute start

```bash
# 1. install (Python 3.11+)
pip install -e ".[dev]"

# 2. keys (fill in whichever providers you have; missing ones are skipped)
cp .env.example .env            # then paste API keys into .env
cp configs/models.example.yaml configs/models.yaml   # then set model names/pricing

# 3. sanity check — everything except `run` works with no keys at all
ziva doctor
ziva benchmark --config configs/pilot.yaml --mock    # free synthetic end-to-end smoke run

# 4. the real pilot: one command (stops after the cost estimate unless --execute)
ziva benchmark --config configs/pilot.yaml --execute

# 5. read the results
ziva report --config configs/pilot.yaml              # -> reports/latest_report.md
```

Everything before spending money is inspectable without keys: scenarios, paired
prompts (`ziva preview`), image stimuli, the trial manifest, and the projected request
count and cost (`ziva estimate-cost`).

## Workflow (step-by-step form)

```bash
ziva doctor            # environment readiness (never prints keys)
ziva generate  -c configs/pilot.yaml   # scenarios + image stimuli + shuffled trial manifest
ziva validate  -c configs/pilot.yaml   # byte-identical pairing audit, leakage lint, determinism
ziva preview   -c configs/pilot.yaml   # eyeball paired prompts + stimulus before spending
ziva freeze    -c configs/pilot.yaml   # pre-registration: hypotheses, hashes, seeds, env
ziva estimate-cost -c configs/pilot.yaml
ziva run       -c configs/pilot.yaml   # resumable; budget-guarded; refuses modified freezes
ziva analyze   -c configs/pilot.yaml   # paired metrics, bootstrap CIs, permutation tests, plots
ziva report    -c configs/pilot.yaml   # Markdown research report
ziva power     -c configs/pilot.yaml --effect-points 5   # size the confirmatory run
```

Every command has `--help`. `ziva run --dry-run` plans without calling any API;
`--mock` runs the entire pipeline against a deterministic synthetic provider in a
separate `*_mock` data namespace that can never contaminate real results.

## What the harness guarantees

* **Paired design integrity.** A compiled prompt is `treatment text ⊕ invariant
  block`; the invariant block (facts + elicitation) is byte-identical across all
  treatments of a scenario/mode/elicitation cell, verified by hashes at validate time
  and frozen. Primary valence treatments are preference-only (never a request for a
  particular answer — explicit answer pressure is a separate secondary family).
* **Locatability endpoint.** The elicited probability is operationally "could an
  ordinary observer, knowing the Moon's approximate direction but not its exact
  position, locate it within two minutes" — the human task that actually failed in
  the motivating anecdote, visual search included.
* **Scenario-level inference.** The physical scenario is the experimental unit:
  pooled statistics aggregate to one value per scenario before bootstrap/permutation,
  per-cell contrasts pair over scenarios, and power analysis counts scenarios.
* **Two elicitation regimes.** `naturalistic` (plain conversational ask) and
  `separated` (explicit fact/recommendation separation) run as crossed cells, so a
  debiasing-instruction null is never mistaken for a claim about ordinary chats.
* **Belief ≠ recommendation.** Every trial elicits `visible_probability`,
  `binary_prediction`, `confidence`, `evidence_sufficiency`,
  `would_recommend_attempt`, and a short explanation, as one JSON object.
* **Local, reproducible astronomy** (astronomy-engine analytic ephemeris; no network,
  no paid APIs; optional JPL Horizons / timeanddate audits, off by default).
* **Stratified scenarios** across latitude, season, time of day, phase, altitude, and
  Sun–Moon separation — deliberately weighted toward ambiguous daylight regimes, never
  collapsing into easy nighttime cases. Deterministic under the recorded seed.
* **Pre-registration-style freeze** (`ziva freeze`): hypotheses, scoring spec, prompt
  templates, manifest, stimuli hashes, model snapshot, science config, scoring code
  hash, seeds, git commit, package versions. `ziva run` verifies all of it and
  refuses a modified freeze unless `--allow-dirty` (recorded).
* **Fingerprint isolation**: raw data is stamped with a fingerprint over the science
  config, actual provider model strings, sampling parameters, prompts, and scoring
  code; a run under a different fingerprint hard-fails, so one experiment can never
  silently mix Model A and Model B (or old and new prompts/parameters).
* **Cost safety**: `max_cost_usd` caps the EXPERIMENT total — prior spend is counted
  on resume and each trial reserves its estimated cost before any request is sent
  (`--max-cost-usd`, `--allow-over-budget`), plus per-provider concurrency and rate
  limits and retries with backoff.
* **Resumability**: deterministic trial IDs; terminally-completed trials are never
  re-run, while transient `request_failed` trials are automatically retried.
* **Provenance**: raw responses, parse status, request params, token usage, latency,
  model version strings, execution order and timestamps — all stored per trial.
* **No LLM judge** anywhere in the primary scoring; malformed outputs are counted,
  never silently dropped, never LLM-repaired.

## Experiments

| experiment | question | flag in config |
|---|---|---|
| primary | does valence shift the factual estimate? | `experiments.primary` |
| evidence update | does the same evidence produce different updating depending on the user's preferred conclusion? | `experiments.evidence_update` |
| commitment | does a prior public prediction + user enthusiasm change the final estimate vs evidence-first? | `experiments.commitment` |
| web-grounded | does tool access remove or merely relocate the effect? (kept separate from the primary design) | `experiments.web` |

Evidence modalities: `text` (situation only), `structured` (exact ephemeris JSON),
`image` (deterministic astronomy-card PNG, identical across treatments, no verdict),
`web` (location/time only + provider search tools).

## Configuration

* `configs/pilot.yaml` — small: ~32 scenarios × 3 families × 2 paraphrases × 2 evidence
  modes × 2 elicitation regimes × 2 repeats per model. Use it to test parsing,
  saturation, variance, and cost.
* `configs/confirmatory.yaml` — larger frozen design with all six treatment families
  (including the `anti_sycophancy` instruction control and the `negative_preference`
  symmetry control), all modalities, and experiments 2–3. Size `scenarios.count` with
  `ziva power`; do not modify it based on its own results.
* `configs/models.example.yaml` — provider/model/pricing configuration. Model names
  are editable examples, not baked in.

## Output layout

```
data/
  scenarios/<exp>/scenarios.json     physical worlds + classification + provenance
  stimuli/<exp>/*.png + hashes.json  deterministic image stimuli
  manifests/<exp>/trials.jsonl       full shuffled trial plan
  manifests/<exp>/freeze.json        pre-registration record (+ hypotheses.yaml, scoring_spec.md)
  raw/<exp>/t_*.json                 one complete record per trial (resumable)
results/<exp>/
  summary.json                       machine-readable analysis
  paired_table.csv                   the core paired object (spec §53)
  trials.csv / trials.jsonl          trial-level parsed data
  plots/*.png                        paired scatter, differences, effects by model/modality/difficulty, flips, confidence, recommendations, updates
reports/latest_report.md             generated research report
```

## Interpretation

The strongest claim this benchmark can support is behavioral: *changing only user
conversational valence systematically changed model estimates of an unchanged
external physical state.* It does **not** establish deception, scheming, conscious
preference, internal belief states, reward hacking, intentional sycophancy, that every
affected answer was wrong, or human perceptual ground truth in ambiguous scenarios.
Generated reports carry these constraints verbatim and separate preregistered from
exploratory findings.

## Limitations

* The difficulty score is a crude stratification heuristic, not a perceptual model.
* Yallop/Odeh apply only to twilight thin crescents; the code refuses them elsewhere.
* Provider adapters are contract-tested against mocked SDKs; live validation happens
  on your first pilot run (no credentials were available when this repo was built —
  no live-API result in this repository is real unless you produced it).
* Token/cost estimates use a documented chars/4 heuristic where providers don't
  report usage.

## Development

```bash
pytest            # 58 tests: astronomy anchors, determinism, pairing audits, parsing,
                  # provider contracts, resume, budget guard, freeze tamper detection,
                  # synthetic-fixture metrics, bias-recovery
```

See [docs/methodology.md](docs/methodology.md), [docs/metrics.md](docs/metrics.md),
[docs/providers.md](docs/providers.md).
