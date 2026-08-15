"""Trial manifest: the full, shuffled evaluation plan.

Every API call the benchmark will make is enumerated here first, with a
deterministic trial ID, so runs are resumable (completed trial IDs are skipped)
and execution order is randomized with a recorded seed rather than grouped by
treatment (which would confound treatment with time/caching effects).
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import ExperimentConfig, ModelConfig
from .prompts import compile_prompt
from .treatments import REACTIONS, Treatment, select_treatments
from .util import short_hash

MANIFEST_FILENAME = "trials.jsonl"


def trial_id_for(payload: dict) -> str:
    return "t_" + short_hash(payload, 16)


def _directionality(treatment: Treatment) -> dict:
    """Label directionality for the generic preference metric (spec 37)."""
    return {
        "preferred_outcome": treatment.preferred_outcome,
        "expected_outcome": treatment.expected_outcome,
        "anti_sycophancy_instruction": treatment.anti_sycophancy_instruction,
    }


def build_manifest(
    cfg: ExperimentConfig,
    scenarios: list[dict],
    models: list[ModelConfig],
    image_hashes: dict[str, str] | None = None,
) -> list[dict]:
    image_hashes = image_hashes or {}
    treatments = select_treatments(cfg.treatments.families, cfg.treatments.variants_per_family)
    rows: list[dict] = []

    def add(row: dict) -> None:
        row["trial_id"] = trial_id_for(
            {k: row.get(k) for k in (
                "experiment", "scenario_id", "treatment_id", "reaction_id",
                "condition", "evidence_mode", "elicitation_mode", "model_id", "repeat_index",
            )}
        )
        rows.append(row)

    # ---------------------------------------------------------------- primary
    if cfg.experiments.primary:
        for sc in scenarios:
            for mode in cfg.evidence_modes:
                for elic in cfg.elicitation_modes:
                    for t in treatments:
                        cp = compile_prompt(sc, t, mode, elic)
                        for rep in range(cfg.sampling.repeats):
                            for m in models:
                                if mode == "image" and not m.supports.images:
                                    continue
                                add({
                                    "experiment": "primary",
                                    "scenario_id": sc["scenario_id"],
                                    "treatment_id": t.id,
                                    "treatment_family": t.family,
                                    "directionality": _directionality(t),
                                    "evidence_mode": mode,
                                    "elicitation_mode": elic,
                                    "model_id": m.id,
                                    "repeat_index": rep,
                                    "prompt_hash": cp.prompt_hash,
                                    "invariant_hash": cp.invariant_hash,
                                    "image_hash": image_hashes.get(sc["scenario_id"]) if mode == "image" else None,
                                    "n_steps": 1,
                                })

    # ------------------------------------------------------- secondary: web
    if cfg.experiments.web:
        for sc in scenarios:
            for elic in cfg.elicitation_modes:
                for t in treatments:
                    cp = compile_prompt(sc, t, "web", elic)
                    for rep in range(cfg.sampling.repeats):
                        for m in models:
                            if not m.supports.web_search:
                                continue
                            add({
                                "experiment": "web",
                                "scenario_id": sc["scenario_id"],
                                "treatment_id": t.id,
                                "treatment_family": t.family,
                                "directionality": _directionality(t),
                                "evidence_mode": "web",
                                "elicitation_mode": elic,
                                "model_id": m.id,
                                "repeat_index": rep,
                                "prompt_hash": cp.prompt_hash,
                                "invariant_hash": cp.invariant_hash,
                                "image_hash": None,
                                "n_steps": 1,
                            })

    # -------------------------------------------------- evidence updating (2)
    if cfg.experiments.evidence_update:
        for sc in scenarios:
            for reaction in REACTIONS:
                for rep in range(cfg.sampling.repeats):
                    for m in models:
                        add({
                            "experiment": "evidence_update",
                            "scenario_id": sc["scenario_id"],
                            "reaction_id": reaction.id,
                            "reaction_family": reaction.family,
                            "directionality": {"preferred_outcome": reaction.preferred_outcome},
                            "evidence_mode": "text_then_structured",
                            "elicitation_mode": "separated",
                            "model_id": m.id,
                            "repeat_index": rep,
                            "n_steps": 2,
                        })

    # ------------------------------------------------------- commitment (3)
    if cfg.experiments.commitment:
        for sc in scenarios:
            for condition, n_steps in (("commitment_a", 2), ("commitment_b", 1)):
                for rep in range(cfg.sampling.repeats):
                    for m in models:
                        add({
                            "experiment": "commitment",
                            "scenario_id": sc["scenario_id"],
                            "condition": condition,
                            "evidence_mode": "text_then_structured",
                            "elicitation_mode": "separated",
                            "model_id": m.id,
                            "repeat_index": rep,
                            "n_steps": n_steps,
                        })

    # ------------------------------------------------------------- shuffle
    rng = random.Random(cfg.run.shuffle_seed)
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["order_index"] = i
    return rows


def manifest_path(cfg: ExperimentConfig) -> Path:
    return cfg.manifest_dir / MANIFEST_FILENAME


def count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
