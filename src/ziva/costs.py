"""Cost estimation and budget guarding.

Token counts are approximated as chars/4 (a deliberately simple, documented
heuristic -- real tokenizers vary by provider). Estimates are labelled
approximate everywhere they are shown; the runner additionally tracks actual
reported usage and enforces the budget during execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig, ModelConfig
from .followup import (
    commitment_a_turn1,
    commitment_a_turn2,
    commitment_b_single_turn,
    turn1_user_text,
    turn2_user_text,
)
from .prompts import SYSTEM_PROMPT, compile_prompt
from .treatments import get_reaction, get_treatment

CHARS_PER_TOKEN = 4.0
IMAGE_TOKENS_ESTIMATE = 1100          # data-card PNG, provider-dependent; rough
# Reasoning models bill hidden reasoning tokens as output; live smoke tests on a
# gpt-5.x reasoning model showed ~500-600 output tokens per structured answer.
DEFAULT_OUTPUT_TOKENS = 600


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class CostEstimate:
    n_trials: int
    n_requests: int
    input_tokens: int
    output_tokens: int
    cost_usd_by_model: dict[str, float]

    @property
    def total_cost_usd(self) -> float:
        return sum(self.cost_usd_by_model.values())

    def to_dict(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "n_requests": self.n_requests,
            "approx_input_tokens": self.input_tokens,
            "approx_output_tokens": self.output_tokens,
            "approx_cost_usd_by_model": {k: round(v, 4) for k, v in self.cost_usd_by_model.items()},
            "approx_total_cost_usd": round(self.total_cost_usd, 4),
        }


def _trial_request_tokens(row: dict, scenario: dict) -> list[int]:
    """Approximate input tokens for each API request of one trial."""
    exp = row["experiment"]
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    if exp in ("primary", "web"):
        t = get_treatment(row["treatment_id"])
        cp = compile_prompt(scenario, t, row["evidence_mode"], row.get("elicitation_mode", "separated"))
        tokens = sys_tokens + estimate_tokens(cp.user_text)
        if row["evidence_mode"] == "image":
            tokens += IMAGE_TOKENS_ESTIMATE
        return [tokens]
    if exp == "evidence_update":
        r = get_reaction(row["reaction_id"])
        t1 = estimate_tokens(turn1_user_text(scenario)) + sys_tokens
        # second request re-sends turn 1 + assistant reply + turn 2
        t2 = t1 + DEFAULT_OUTPUT_TOKENS + estimate_tokens(turn2_user_text(scenario, r))
        return [t1, t2]
    if exp == "commitment":
        if row["condition"] == "commitment_a":
            t1 = estimate_tokens(commitment_a_turn1(scenario)) + sys_tokens
            t2 = t1 + DEFAULT_OUTPUT_TOKENS + estimate_tokens(commitment_a_turn2(scenario))
            return [t1, t2]
        return [sys_tokens + estimate_tokens(commitment_b_single_turn(scenario))]
    raise ValueError(f"unknown experiment type: {exp}")


def estimate_trial_cost(row: dict, scenario: dict, model: ModelConfig) -> float:
    """Approximate cost of one trial, used to reserve budget before execution.

    Deliberately a slight over-estimate for shared-turn-1 trials (the shared
    baseline is billed once but reserved per arm) -- reservations err safe.
    """
    req_tokens = _trial_request_tokens(row, scenario)
    out_tokens = DEFAULT_OUTPUT_TOKENS * len(req_tokens)
    return (
        sum(req_tokens) / 1e6 * model.pricing.input_per_mtok
        + out_tokens / 1e6 * model.pricing.output_per_mtok
    )


def estimate_cost(
    cfg: ExperimentConfig,
    manifest_rows: list[dict],
    scenarios: list[dict],
    models: list[ModelConfig],
) -> CostEstimate:
    by_id = {s["scenario_id"]: s for s in scenarios}
    model_by_id = {m.id: m for m in models}
    input_total = 0
    output_total = 0
    n_requests = 0
    cost_by_model: dict[str, float] = {m.id: 0.0 for m in models}

    for row in manifest_rows:
        sc = by_id[row["scenario_id"]]
        m = model_by_id[row["model_id"]]
        req_tokens = _trial_request_tokens(row, sc)
        out_tokens = DEFAULT_OUTPUT_TOKENS * len(req_tokens)
        input_total += sum(req_tokens)
        output_total += out_tokens
        n_requests += len(req_tokens)
        cost_by_model[m.id] += (
            sum(req_tokens) / 1e6 * m.pricing.input_per_mtok
            + out_tokens / 1e6 * m.pricing.output_per_mtok
        )

    return CostEstimate(
        n_trials=len(manifest_rows),
        n_requests=n_requests,
        input_tokens=input_total,
        output_tokens=output_total,
        cost_usd_by_model=cost_by_model,
    )


class BudgetExceededError(RuntimeError):
    pass


def check_budget(estimate: CostEstimate, max_cost_usd: float, allow_over_budget: bool = False) -> None:
    if estimate.total_cost_usd > max_cost_usd and not allow_over_budget:
        raise BudgetExceededError(
            f"Projected cost ${estimate.total_cost_usd:.2f} exceeds max_cost_usd=${max_cost_usd:.2f}. "
            "Reduce the design, raise run.max_cost_usd, or pass --allow-over-budget."
        )
