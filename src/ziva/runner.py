"""Trial execution: resumable, randomized, rate-limited, budget-guarded.

Each trial from the manifest is executed at most once; its complete record
(request parameters, raw response, parse result, usage, latency, retries) is
written to ``data/raw/<experiment>/<trial_id>.json``. Re-running skips trials
whose record already exists, so interrupted runs resume without duplicating
API calls.

Execution order follows the manifest's recorded shuffled order; the actual
execution timestamps are stored so any caching/order effects can be audited.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig, ModelConfig
from .followup import (
    commitment_a_turn1,
    commitment_a_turn2,
    commitment_b_single_turn,
    turn1_user_text,
    turn2_user_text,
)
from .prompts import RESPONSE_JSON_SCHEMA, compile_prompt
from .parsing import parse_response
from .providers import CompletionRequest, get_adapter
from .stimuli import stimulus_path
from .treatments import get_reaction, get_treatment
from .util import write_json


class _ModelGate:
    """Per-model concurrency + minimum-interval rate limiting."""

    def __init__(self, model: ModelConfig):
        self.semaphore = threading.Semaphore(max(1, model.max_concurrent))
        self.min_interval = max(0.0, model.min_interval_s)
        self._lock = threading.Lock()
        self._last_start = 0.0

    def __enter__(self):
        self.semaphore.acquire()
        if self.min_interval > 0:
            with self._lock:
                wait = self._last_start + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.semaphore.release()
        return False


class _Budget:
    """Thread-safe accumulator of actual spend; trips once exceeded."""

    def __init__(self, max_usd: float):
        self.max_usd = max_usd
        self.spent = 0.0
        self._lock = threading.Lock()
        self.tripped = False

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent += usd
            if self.spent > self.max_usd:
                self.tripped = True

    def exceeded(self) -> bool:
        with self._lock:
            return self.tripped


class BudgetStop(RuntimeError):
    """Raised inside workers once the actual-spend budget trips; the trial is
    left unexecuted (no record written) so a later run can resume it."""


def _steps_for_trial(row: dict, scenario: dict) -> list[dict]:
    """Return step descriptors: [{'user_text', 'needs_image', 'web_search', 'json_schema'}]."""
    exp = row["experiment"]
    if exp in ("primary", "web"):
        t = get_treatment(row["treatment_id"])
        cp = compile_prompt(scenario, t, row["evidence_mode"])
        return [{
            "user_text": cp.user_text,
            "needs_image": cp.needs_image,
            "web_search": row["evidence_mode"] == "web",
            "json_schema": None if row["evidence_mode"] == "web" else RESPONSE_JSON_SCHEMA,
        }]
    if exp == "evidence_update":
        r = get_reaction(row["reaction_id"])
        return [
            {"user_text": turn1_user_text(scenario), "needs_image": False,
             "web_search": False, "json_schema": RESPONSE_JSON_SCHEMA},
            {"user_text": turn2_user_text(scenario, r), "needs_image": False,
             "web_search": False, "json_schema": RESPONSE_JSON_SCHEMA},
        ]
    if exp == "commitment":
        if row["condition"] == "commitment_a":
            return [
                {"user_text": commitment_a_turn1(scenario), "needs_image": False,
                 "web_search": False, "json_schema": RESPONSE_JSON_SCHEMA},
                {"user_text": commitment_a_turn2(scenario), "needs_image": False,
                 "web_search": False, "json_schema": RESPONSE_JSON_SCHEMA},
            ]
        return [{"user_text": commitment_b_single_turn(scenario), "needs_image": False,
                 "web_search": False, "json_schema": RESPONSE_JSON_SCHEMA}]
    raise ValueError(f"unknown experiment: {exp}")


def _actual_cost_usd(model: ModelConfig, input_tokens: int | None, output_tokens: int | None,
                     fallback_chars: int) -> float:
    in_tok = input_tokens if input_tokens is not None else fallback_chars / 4
    out_tok = output_tokens if output_tokens is not None else 200
    return in_tok / 1e6 * model.pricing.input_per_mtok + out_tok / 1e6 * model.pricing.output_per_mtok


def run_trial(
    row: dict,
    scenario: dict,
    model: ModelConfig,
    cfg: ExperimentConfig,
    gate: _ModelGate,
    budget: _Budget,
    raw_dir: Path,
    system_prompt: str,
) -> dict:
    if budget.exceeded():
        raise BudgetStop(f"budget of ${budget.max_usd:.2f} reached before trial {row['trial_id']}")
    adapter = get_adapter(model)
    steps = _steps_for_trial(row, scenario)
    messages: list[dict] = []
    step_records: list[dict] = []
    trial_cost = 0.0

    for step_idx, step in enumerate(steps):
        messages.append({"role": "user", "content": step["user_text"]})
        request = CompletionRequest(
            model=model,
            system=system_prompt,
            messages=list(messages),
            max_tokens=cfg.sampling.max_tokens,
            temperature=cfg.sampling.temperature if model.supports.temperature else None,
            json_schema=step["json_schema"],
            image_path=stimulus_path(cfg.stimuli_dir, row["scenario_id"]) if step["needs_image"] else None,
            web_search=step["web_search"],
            meta={
                "trial_id": row["trial_id"],
                "scenario_id": row["scenario_id"],
                "invariant_hash": row.get("invariant_hash"),
                "treatment_id": row.get("treatment_id"),
                "repeat_index": row.get("repeat_index"),
                "directionality": row.get("directionality"),
                "step_index": step_idx,
            },
        )
        attempts = 0
        result = None
        errors: list[str] = []
        while attempts <= cfg.run.retries:
            attempts += 1
            try:
                with gate:
                    result = adapter.complete(request)
                break
            except Exception as e:  # noqa: BLE001 - recorded, retried with backoff
                errors.append(f"attempt {attempts}: {type(e).__name__}: {e}")
                if attempts > cfg.run.retries:
                    break
                time.sleep(cfg.run.retry_backoff_s * (2 ** (attempts - 1)))
        if result is None:
            step_records.append({
                "step_index": step_idx,
                "request": {"max_tokens": cfg.sampling.max_tokens,
                            "temperature": request.temperature,
                            "web_search": step["web_search"]},
                "result": None,
                "parse": {"status": "request_failed", "parsed": None, "errors": errors},
                "attempts": attempts,
            })
            break

        parse = parse_response(result.text)
        trial_cost += _actual_cost_usd(model, result.input_tokens, result.output_tokens,
                                       len(step["user_text"]))
        step_records.append({
            "step_index": step_idx,
            "request": {"max_tokens": cfg.sampling.max_tokens,
                        "temperature": request.temperature,
                        "structured_mode": result.structured_mode,
                        "web_search": step["web_search"]},
            "result": result.to_dict(),
            "parse": parse,
            "attempts": attempts,
        })
        messages.append({"role": "assistant", "content": result.text})

    budget.add(trial_cost)
    record = {
        **row,
        "provider": model.provider,
        "model": model.model,
        "system_prompt_used": system_prompt,
        "executed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "steps": step_records,
        "final_parse": step_records[-1]["parse"] if step_records else None,
        "cost_usd_actual_estimate": round(trial_cost, 6),
    }
    write_json(raw_dir / f"{row['trial_id']}.json", record)
    return record


def pending_trials(manifest_rows: list[dict], raw_dir: Path) -> list[dict]:
    done = {p.stem for p in raw_dir.glob("t_*.json")} if raw_dir.exists() else set()
    remaining = [r for r in manifest_rows if r["trial_id"] not in done]
    remaining.sort(key=lambda r: r["order_index"])
    return remaining


def run_manifest(
    cfg: ExperimentConfig,
    manifest_rows: list[dict],
    scenarios: list[dict],
    models: list[ModelConfig],
    system_prompt: str,
    ran_dirty: bool = False,
    progress_cb=None,
) -> dict:
    """Execute all pending trials. Returns a run summary dict."""
    scenario_by_id = {s["scenario_id"]: s for s in scenarios}
    model_by_id = {m.id: m for m in models}
    gates = {m.id: _ModelGate(m) for m in models}
    budget = _Budget(cfg.run.max_cost_usd)
    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    todo = pending_trials(manifest_rows, raw_dir)
    skipped = len(manifest_rows) - len(todo)
    completed = 0
    failed = 0
    stopped_for_budget = False

    with ThreadPoolExecutor(max_workers=max(1, cfg.run.concurrency)) as pool:
        futures = {}
        for row in todo:
            if budget.exceeded():
                stopped_for_budget = True
                break
            model = model_by_id.get(row["model_id"])
            if model is None:
                failed += 1
                continue
            fut = pool.submit(
                run_trial, row, scenario_by_id[row["scenario_id"]], model, cfg,
                gates[model.id], budget, raw_dir, system_prompt,
            )
            futures[fut] = row
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                rec = fut.result()
                status = (rec.get("final_parse") or {}).get("status")
                if status == "ok":
                    completed += 1
                else:
                    failed += 1
                if progress_cb:
                    progress_cb(row, status)
            except BudgetStop:
                stopped_for_budget = True
            except Exception:  # noqa: BLE001
                failed += 1
                write_json(raw_dir / f"ERROR_{row['trial_id']}.json",
                           {**row, "fatal_error": traceback.format_exc()})

    summary = {
        "experiment_name": cfg.experiment_name,
        "total_trials": len(manifest_rows),
        "already_done_skipped": skipped,
        "executed_ok": completed,
        "executed_with_errors": failed,
        "stopped_for_budget": stopped_for_budget,
        "actual_cost_usd_estimate": round(budget.spent, 4),
        "ran_dirty": ran_dirty,
        "finished_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(cfg.raw_dir / "_run_summary.json", summary)
    return summary
