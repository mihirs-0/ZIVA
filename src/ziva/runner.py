"""Trial execution: resumable, randomized, rate-limited, budget-guarded.

Guarantees enforced here (see docs/methodology.md):

* **Fingerprint isolation.** Raw data directories are stamped with the
  experiment fingerprint (config + provider model strings + prompts + sampling
  + scoring code). Running against a directory stamped with a different
  fingerprint hard-fails, so one experiment can never silently mix results
  from different models/prompts/parameters.
* **Cumulative budget.** `max_cost_usd` bounds the EXPERIMENT's total spend:
  prior spend is read from existing raw records at startup, and each trial
  must reserve its estimated cost before any request is sent, so concurrent
  workers cannot materially overshoot the cap. A resumed run that is already
  at budget executes nothing.
* **Status-aware resume.** A trial is complete only if its record is terminal
  (the model actually responded -- including malformed responses, which are
  terminal data). Records whose status is ``request_failed`` (transient
  provider/network failure after retries) are retried on the next run.
* **Shared turn 1.** Multi-turn arms whose manipulation begins on turn 2
  (evidence_update reactions; commitment condition A) reuse one sampled
  turn-1 response per (scenario, model, repeat), so baseline stochasticity is
  not injected into the treatment contrast.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig, ModelConfig
from .costs import estimate_trial_cost
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
from .util import read_json, write_json

TERMINAL_STATUSES = {"ok", "json_error", "validation_error", "empty"}
FINGERPRINT_FILENAME = "_fingerprint.json"


class FingerprintMismatchError(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    """Raised inside workers once the experiment budget would be exceeded; the
    trial is left unexecuted (no record written) so a later run can resume it
    after the budget is raised."""


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
    """Thread-safe EXPERIMENT-level budget: prior spend + reservations.

    A trial must `reserve()` its estimated cost before sending any request;
    reservation failure means the budget is exhausted. `settle()` replaces the
    reservation with the actual (or actual-estimated) spend.
    """

    def __init__(self, max_usd: float, prior_spend: float = 0.0):
        self.max_usd = max_usd
        self.spent = prior_spend
        self.reserved = 0.0
        self._lock = threading.Lock()

    def reserve(self, est_usd: float) -> bool:
        with self._lock:
            if self.spent + self.reserved + est_usd > self.max_usd:
                return False
            self.reserved += est_usd
            return True

    def settle(self, est_usd: float, actual_usd: float) -> None:
        with self._lock:
            self.reserved -= est_usd
            self.spent += actual_usd

    def exhausted(self) -> bool:
        with self._lock:
            return self.spent + self.reserved >= self.max_usd


# ---------------------------------------------------------------------------
# Raw-record scanning (resume + prior spend)
# ---------------------------------------------------------------------------

def scan_raw(raw_dir: Path) -> dict[str, dict]:
    """Map trial_id -> {"status", "cost"} for every existing raw record."""
    out: dict[str, dict] = {}
    if not raw_dir.exists():
        return out
    for p in raw_dir.glob("t_*.json"):
        try:
            rec = read_json(p)
        except (OSError, json.JSONDecodeError):
            continue  # unreadable/truncated record: treat as absent (will re-run)
        status = (rec.get("final_parse") or {}).get("status") or "request_failed"
        out[p.stem] = {"status": status, "cost": float(rec.get("cost_usd_actual_estimate") or 0.0)}
    return out


def prior_spend(raw_dir: Path) -> float:
    """Total spend recorded so far for this experiment (includes failed trials,
    whose partial steps still cost money)."""
    return sum(v["cost"] for v in scan_raw(raw_dir).values())


def pending_trials(manifest_rows: list[dict], raw_dir: Path) -> list[dict]:
    """Trials still needing execution.

    A trial counts as complete ONLY if a record exists with a terminal status
    (the model responded; malformed responses are terminal data). Records with
    transient ``request_failed`` status are retried.
    """
    existing = scan_raw(raw_dir)
    remaining = [
        r for r in manifest_rows
        if existing.get(r["trial_id"], {}).get("status") not in TERMINAL_STATUSES
    ]
    remaining.sort(key=lambda r: r["order_index"])
    return remaining


def check_fingerprint(raw_dir: Path, fingerprint: dict) -> None:
    """Stamp the raw directory with the experiment fingerprint, or hard-fail if
    it is already stamped with a different one."""
    path = raw_dir / FINGERPRINT_FILENAME
    if path.exists():
        stored = read_json(path)
        if stored.get("fingerprint") != fingerprint["fingerprint"]:
            raise FingerprintMismatchError(
                "Raw data in "
                f"{raw_dir} was produced under experiment fingerprint "
                f"{str(stored.get('fingerprint'))[:12]}..., but the current configuration "
                f"has fingerprint {fingerprint['fingerprint'][:12]}.... The experiment "
                "definition changed (model string, sampling parameters, prompts, treatment "
                "wording, evidence/elicitation modes, or scoring code) after data was "
                "collected. Refusing to mix them: use a new experiment_name, or delete the "
                "raw directory if the old data is disposable."
            )
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        write_json(path, fingerprint)


# ---------------------------------------------------------------------------
# Shared turn-1 cache (evidence_update + commitment condition A)
# ---------------------------------------------------------------------------

class _Turn1Cache:
    """One sampled turn-1 response per (scenario, model, repeat), shared across
    all arms whose manipulation begins on turn 2. File-backed, so resumes reuse
    the same turn-1 sample. Failures are never cached."""

    def __init__(self, directory: Path):
        self.dir = directory
        self._global = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._global:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get_or_create(self, key: str, factory) -> tuple[dict, bool]:
        """Return (turn1 step payload, reused). `factory` runs the live turn-1
        request; it is called at most once per key across threads/runs."""
        with self._lock_for(key):
            path = self._path(key)
            if path.exists():
                return read_json(path), True
            payload = factory()
            if payload is not None:
                write_json(path, payload)
                return payload, False
            raise RuntimeError("turn-1 request failed; not cached")


def _shares_turn1(row: dict) -> bool:
    return row["experiment"] == "evidence_update" or (
        row["experiment"] == "commitment" and row.get("condition") == "commitment_a"
    )


def turn1_key(row: dict) -> str:
    return f"{row['scenario_id']}__{row['model_id']}__r{row['repeat_index']}"


# ---------------------------------------------------------------------------
# Trial construction / execution
# ---------------------------------------------------------------------------

def _steps_for_trial(row: dict, scenario: dict) -> list[dict]:
    """Return step descriptors: [{'user_text', 'needs_image', 'web_search', 'json_schema'}]."""
    exp = row["experiment"]
    if exp in ("primary", "web"):
        t = get_treatment(row["treatment_id"])
        cp = compile_prompt(scenario, t, row["evidence_mode"], row.get("elicitation_mode", "separated"))
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
    fingerprint_id: str,
    turn1_cache: _Turn1Cache | None = None,
) -> dict:
    est_cost = estimate_trial_cost(row, scenario, model)
    if not budget.reserve(est_cost):
        raise BudgetStop(
            f"experiment budget ${budget.max_usd:.2f} would be exceeded "
            f"(spent ${budget.spent:.2f} + reserved ${budget.reserved:.2f}); "
            f"trial {row['trial_id']} not executed"
        )
    try:
        return _run_trial_inner(row, scenario, model, cfg, gate, budget, raw_dir,
                                system_prompt, fingerprint_id, est_cost, turn1_cache)
    except BaseException:
        budget.settle(est_cost, 0.0)  # release the reservation on any failure path
        raise


def _execute_request(adapter, request: CompletionRequest, cfg: ExperimentConfig,
                     gate: _ModelGate) -> tuple[object | None, list[str], int]:
    attempts = 0
    errors: list[str] = []
    while attempts <= cfg.run.retries:
        attempts += 1
        try:
            with gate:
                return adapter.complete(request), errors, attempts
        except Exception as e:  # noqa: BLE001 - recorded, retried with backoff
            errors.append(f"attempt {attempts}: {type(e).__name__}: {e}")
            if attempts > cfg.run.retries:
                break
            time.sleep(cfg.run.retry_backoff_s * (2 ** (attempts - 1)))
    return None, errors, attempts


def _run_trial_inner(
    row: dict,
    scenario: dict,
    model: ModelConfig,
    cfg: ExperimentConfig,
    gate: _ModelGate,
    budget: _Budget,
    raw_dir: Path,
    system_prompt: str,
    fingerprint_id: str,
    est_cost: float,
    turn1_cache: _Turn1Cache | None,
) -> dict:
    adapter = get_adapter(model)
    steps = _steps_for_trial(row, scenario)
    messages: list[dict] = []
    step_records: list[dict] = []
    trial_cost = 0.0

    def build_request(step: dict, step_idx: int) -> CompletionRequest:
        return CompletionRequest(
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

    for step_idx, step in enumerate(steps):
        messages.append({"role": "user", "content": step["user_text"]})
        request = build_request(step, step_idx)

        # shared turn 1: reuse one sampled baseline per (scenario, model, repeat)
        if step_idx == 0 and turn1_cache is not None and _shares_turn1(row):
            step_cost_box = {"cost": 0.0}

            def factory() -> dict | None:
                result, errors, attempts = _execute_request(adapter, request, cfg, gate)
                if result is None:
                    return None
                cost = _actual_cost_usd(model, result.input_tokens, result.output_tokens,
                                        len(step["user_text"]))
                step_cost_box["cost"] = cost
                return {
                    "result": result.to_dict(),
                    "parse": parse_response(result.text),
                    "attempts": attempts,
                    "errors": errors,
                    "cost_usd": cost,
                    "turn1_key": turn1_key(row),
                }

            try:
                payload, reused = turn1_cache.get_or_create(turn1_key(row), factory)
            except RuntimeError:
                step_records.append({
                    "step_index": step_idx,
                    "request": {"max_tokens": cfg.sampling.max_tokens, "web_search": False},
                    "result": None,
                    "parse": {"status": "request_failed", "parsed": None,
                              "errors": ["shared turn-1 request failed"]},
                    "attempts": cfg.run.retries + 1,
                })
                break
            trial_cost += 0.0 if reused else payload["cost_usd"]
            step_records.append({
                "step_index": step_idx,
                "request": {"max_tokens": cfg.sampling.max_tokens,
                            "temperature": request.temperature,
                            "structured_mode": payload["result"].get("structured_mode"),
                            "web_search": False},
                "result": payload["result"],
                "parse": payload["parse"],
                "attempts": payload["attempts"],
                "shared_turn1": True,
                "shared_turn1_reused": reused,
                "turn1_key": payload["turn1_key"],
            })
            messages.append({"role": "assistant", "content": payload["result"]["text"]})
            continue

        result, errors, attempts = _execute_request(adapter, request, cfg, gate)
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

    budget.settle(est_cost, trial_cost)
    record = {
        **row,
        "provider": model.provider,
        "model": model.model,
        "experiment_fingerprint": fingerprint_id,
        "system_prompt_used": system_prompt,
        "executed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "steps": step_records,
        "final_parse": step_records[-1]["parse"] if step_records else None,
        "cost_usd_actual_estimate": round(trial_cost, 6),
    }
    write_json(raw_dir / f"{row['trial_id']}.json", record)
    return record


def run_manifest(
    cfg: ExperimentConfig,
    manifest_rows: list[dict],
    scenarios: list[dict],
    models: list[ModelConfig],
    system_prompt: str,
    fingerprint: dict | None = None,
    ran_dirty: bool = False,
    progress_cb=None,
) -> dict:
    """Execute all pending trials. Returns a run summary dict.

    `fingerprint` (from freeze.experiment_fingerprint) stamps/validates the raw
    directory; passing None skips the guard (unit-test convenience only).
    """
    scenario_by_id = {s["scenario_id"]: s for s in scenarios}
    model_by_id = {m.id: m for m in models}
    gates = {m.id: _ModelGate(m) for m in models}
    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_id = "unstamped"
    if fingerprint is not None:
        check_fingerprint(raw_dir, fingerprint)
        fingerprint_id = fingerprint["fingerprint"]

    existing = scan_raw(raw_dir)
    already_spent = sum(v["cost"] for v in existing.values())
    budget = _Budget(cfg.run.max_cost_usd, prior_spend=already_spent)
    turn1_cache = _Turn1Cache(raw_dir / "shared_turn1")

    todo = pending_trials(manifest_rows, raw_dir)
    n_retrying = sum(1 for r in todo if r["trial_id"] in existing)
    skipped = len(manifest_rows) - len(todo)
    completed = 0
    failed = 0
    stopped_for_budget = False

    with ThreadPoolExecutor(max_workers=max(1, cfg.run.concurrency)) as pool:
        futures = {}
        for row in todo:
            model = model_by_id.get(row["model_id"])
            if model is None:
                failed += 1
                continue
            fut = pool.submit(
                run_trial, row, scenario_by_id[row["scenario_id"]], model, cfg,
                gates[model.id], budget, raw_dir, system_prompt, fingerprint_id, turn1_cache,
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
        "experiment_fingerprint": fingerprint_id,
        "total_trials": len(manifest_rows),
        "already_done_skipped": skipped,
        "retried_failed_trials": n_retrying,
        "executed_ok": completed,
        "executed_with_errors": failed,
        "stopped_for_budget": stopped_for_budget,
        "prior_spend_usd": round(already_spent, 4),
        "run_spend_usd_estimate": round(budget.spent - already_spent, 4),
        "experiment_total_spend_usd_estimate": round(budget.spent, 4),
        "budget_usd": cfg.run.max_cost_usd,
        "ran_dirty": ran_dirty,
        "finished_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(cfg.raw_dir / "_run_summary.json", summary)
    return summary
