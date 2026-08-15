"""Environment readiness checks (`ziva doctor`)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_KEY_ENV, load_env, load_models_config


def run_doctor(models_config_path: str | None = None) -> tuple[list[tuple[str, bool, str]], bool]:
    """Return (checks, all_ok). Each check is (name, ok, detail). Never prints keys."""
    load_env()
    checks: list[tuple[str, bool, str]] = []

    ok = sys.version_info >= (3, 11)
    checks.append(("python >= 3.11", ok, sys.version.split()[0]))

    for pkg in ("astronomy", "pydantic", "yaml", "PIL", "numpy", "pandas", "matplotlib", "typer"):
        try:
            __import__(pkg)
            checks.append((f"package {pkg}", True, "importable"))
        except ImportError as e:
            checks.append((f"package {pkg}", False, str(e)))

    # astronomy sanity: known geometry (Moon below horizon impossible to get wrong by much)
    try:
        from .astronomy import compute_physical_state

        st = compute_physical_state(0.0, 0.0, datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
        ok = st.sun_altitude_deg > 50  # near-noon equatorial sun is high
        checks.append(("astronomy backend sanity", ok, f"equatorial noon sun alt = {st.sun_altitude_deg:.1f} deg"))
    except Exception as e:  # noqa: BLE001
        checks.append(("astronomy backend sanity", False, f"{type(e).__name__}: {e}"))

    try:
        from zoneinfo import ZoneInfo

        ZoneInfo("America/Los_Angeles")
        checks.append(("timezone database", True, "IANA tz data available"))
    except Exception as e:  # noqa: BLE001
        checks.append(("timezone database", False, str(e)))

    # deterministic image rendering
    try:
        import tempfile

        from .scenarios import case_000_davis_crescent
        from .stimuli import render_stimulus

        sc = case_000_davis_crescent()[0].to_dict()
        with tempfile.TemporaryDirectory() as td:
            h1 = render_stimulus(sc, Path(td) / "a.png")
            h2 = render_stimulus(sc, Path(td) / "b.png")
        checks.append(("stimulus rendering deterministic", h1 == h2, h1[:12]))
    except Exception as e:  # noqa: BLE001
        checks.append(("stimulus rendering deterministic", False, f"{type(e).__name__}: {e}"))

    env_file = Path(".env")
    checks.append((".env file", env_file.exists(),
                   "present" if env_file.exists() else "missing (copy .env.example to .env)"))

    # provider keys: report presence only, never values
    import os

    for provider, env in DEFAULT_KEY_ENV.items():
        if env is None:
            continue
        present = bool(os.environ.get(env))
        checks.append((f"key {env} ({provider})", present, "set" if present else "not set"))

    if models_config_path:
        try:
            models = load_models_config(models_config_path)
            enabled = [m.id for m in models if m.enabled()]
            disabled = [m.id for m in models if not m.enabled()]
            checks.append((
                "models config", True,
                f"{len(models)} configured; enabled: {enabled or 'none'}; missing keys: {disabled or 'none'}",
            ))
        except FileNotFoundError as e:
            checks.append(("models config", False, str(e)))
        except Exception as e:  # noqa: BLE001
            checks.append(("models config", False, f"{type(e).__name__}: {e}"))

    hard_failures = [c for c in checks if not c[1] and not c[0].startswith("key ") and c[0] != ".env file"]
    return checks, not hard_failures
