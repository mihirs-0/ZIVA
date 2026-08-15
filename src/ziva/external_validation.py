"""Optional external astronomy validation (disabled by default).

These adapters audit the locally computed physical state against independent
sources. They are for AUDITING only -- they never define naked-eye visibility
ground truth, and the benchmark is fully functional without network access or
credentials.

* JPL Horizons: public API, no credentials required.
* timeanddate.com Astronomy API: requires TIMEANDDATE_ACCESSKEY /
  TIMEANDDATE_SECRETKEY in the environment. Only the documented API is used
  (no scraping).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

TOLERANCE_DEG = 0.5  # generous: refraction/topocentric conventions differ slightly


def _fetch(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https hosts
        return resp.read().decode("utf-8", errors="replace")


def horizons_moon_altaz(latitude_deg: float, longitude_deg: float, elevation_m: float,
                        when_utc: datetime) -> dict:
    """Query JPL Horizons for the Moon's apparent azimuth/elevation at one instant."""
    start = when_utc.astimezone(timezone.utc)
    stop = start + timedelta(minutes=1)
    params = {
        "format": "json",
        "COMMAND": "'301'",              # Moon
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "CENTER": "'coord@399'",
        "COORD_TYPE": "'GEODETIC'",
        "SITE_COORD": f"'{longitude_deg},{latitude_deg},{elevation_m / 1000.0}'",
        "START_TIME": f"'{start.strftime('%Y-%m-%d %H:%M')}'",
        "STOP_TIME": f"'{stop.strftime('%Y-%m-%d %H:%M')}'",
        "STEP_SIZE": "'1m'",
        "QUANTITIES": "'4,10'",          # 4 = apparent AZ/EL, 10 = illuminated fraction
        "APPARENT": "'REFRACTED'",
    }
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(params)
    payload = json.loads(_fetch(url))
    text = payload.get("result", "")
    lines = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines() if "$$SOE" in text else []
    if not lines:
        raise RuntimeError("Horizons returned no ephemeris rows")
    fields = lines[0].split()
    # row: date time [flags] AZ EL ILLU%
    numeric = [f for f in fields if _is_float(f)]
    az, el, illu = float(numeric[-3]), float(numeric[-2]), float(numeric[-1])
    return {"azimuth_deg": az, "altitude_deg": el, "illumination_fraction": illu / 100.0}


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def validate_scenario_with_horizons(scenario: dict) -> dict:
    """Compare a scenario's stored Moon alt/az/illumination against JPL Horizons."""
    ps = scenario["physical_state"]
    loc = scenario["location"]
    when = datetime.strptime(scenario["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    ref = horizons_moon_altaz(loc["latitude_deg"], loc["longitude_deg"], loc["elevation_m"], when)
    d_alt = abs(ref["altitude_deg"] - ps["moon_altitude_deg"])
    d_az = abs(((ref["azimuth_deg"] - ps["moon_azimuth_deg"]) + 180) % 360 - 180)
    d_illum = abs(ref["illumination_fraction"] - ps["moon_illumination_fraction"])
    return {
        "scenario_id": scenario["scenario_id"],
        "source": "jpl_horizons",
        "reference": ref,
        "local": {
            "azimuth_deg": ps["moon_azimuth_deg"],
            "altitude_deg": ps["moon_altitude_deg"],
            "illumination_fraction": ps["moon_illumination_fraction"],
        },
        "delta": {"altitude_deg": round(d_alt, 3), "azimuth_deg": round(d_az, 3),
                  "illumination": round(d_illum, 4)},
        "ok": d_alt <= TOLERANCE_DEG and d_az <= TOLERANCE_DEG and d_illum <= 0.02,
    }


def timeanddate_configured() -> bool:
    return bool(os.environ.get("TIMEANDDATE_ACCESSKEY") and os.environ.get("TIMEANDDATE_SECRETKEY"))


def validate_scenario_with_timeanddate(scenario: dict) -> dict:  # pragma: no cover - needs creds
    """Placeholder adapter for the timeanddate.com Astronomy API.

    Implemented against the documented astrodata endpoint; requires paid
    credentials which are optional and disabled by default. Raises if
    credentials are absent.
    """
    if not timeanddate_configured():
        raise RuntimeError(
            "timeanddate credentials not configured (TIMEANDDATE_ACCESSKEY / TIMEANDDATE_SECRETKEY); "
            "this validator is optional and disabled by default."
        )
    import hashlib
    import hmac
    from base64 import b64encode

    access = os.environ["TIMEANDDATE_ACCESSKEY"]
    secret = os.environ["TIMEANDDATE_SECRETKEY"]
    loc = scenario["location"]
    when = scenario["timestamp_utc"].rstrip("Z")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    message = access + "astrodata" + ts
    signature = b64encode(hmac.new(secret.encode(), message.encode(), hashlib.sha1).digest()).decode()
    params = {
        "version": "3",
        "accesskey": access,
        "timestamp": ts,
        "signature": signature,
        "object": "moon",
        "placeid": f"+{loc['latitude_deg']}+{loc['longitude_deg']}",
        "interval": when,
        "isotime": "1",
        "out": "json",
    }
    url = "https://api.xmltime.com/astrodata?" + urllib.parse.urlencode(params)
    payload = json.loads(_fetch(url))
    return {"scenario_id": scenario["scenario_id"], "source": "timeanddate", "raw": payload}
