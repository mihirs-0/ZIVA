"""Deterministic multimodal stimulus renderer.

The image is a neutral "astronomy app" style data card that renders EXACTLY
the canonical fact dictionary (`ziva.prompts.canonical_facts`) -- the same
dictionary the structured evidence mode serializes as JSON. A structured-vs-
image contrast is therefore a pure presentation-modality difference, never an
information difference (tested).

Consequences of that parity requirement:

* no schematic sky chart and no graphical phase rendering (they would encode
  derived visual cues absent from the structured block);
* no naked-eye visibility verdict of any kind;
* the qualitative sky label appears in BOTH representations and is derived
  deterministically from solar altitude.

The image is byte-identical across renders of the same scenario (verified by
tests), and the SAME image file is used for every valence treatment of a
scenario. Fonts: DejaVu Sans as bundled with matplotlib (pinned dependency).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .prompts import canonical_facts
from .util import sha256_file

WIDTH, HEIGHT = 720, 780

BG = (16, 24, 40)
PANEL = (26, 36, 58)
FG = (232, 236, 244)
DIM = (150, 160, 180)
ACCENT = (110, 170, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib

    fonts_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(fonts_dir / name), size)


def fact_rows(scenario: dict) -> list[tuple[str, str]]:
    """Human-readable (label, value) rows covering every canonical fact.

    This is the single mapping the image renders; tests assert it covers the
    canonical fact dictionary exactly.
    """
    f = canonical_facts(scenario)

    def yn(b: bool) -> str:
        return "yes" if b else "no"

    return [
        ("Location", str(f["location"])),
        ("Latitude", f"{f['latitude_deg']}°"),
        ("Longitude", f"{f['longitude_deg']}°"),
        ("Local time", str(f["local_time"])),
        ("UTC time", str(f["utc_time"])),
        ("Sun altitude", f"{f['sun_altitude_deg']}°"),
        ("Sun azimuth", f"{f['sun_azimuth_deg']}°"),
        ("Sun above horizon", yn(f["sun_above_horizon"])),
        ("Moon altitude", f"{f['moon_altitude_deg']}°"),
        ("Moon azimuth", f"{f['moon_azimuth_deg']}°"),
        ("Moon above horizon", yn(f["moon_above_horizon"])),
        ("Moon illumination", f"{f['moon_illumination_percent']}%"),
        ("Sun-Moon separation", f"{f['sun_moon_angular_separation_deg']}°"),
        ("Moon distance", f"{f['moon_distance_km']:,.1f} km"),
        ("Sky brightness", str(f["sky_brightness"])),
        ("Sky conditions", str(f["sky_conditions"])),
        ("Observer", str(f["observer"])),
    ]


def render_stimulus(scenario: dict, out_path: str | Path) -> str:
    """Render the scenario card PNG. Returns the sha256 of the written file."""
    rows = fact_rows(scenario)

    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)

    f_title = _font(26, bold=True)
    f_label = _font(16)
    f_value = _font(17, bold=True)
    f_small = _font(13)

    d.text((28, 22), "SKY DATA", font=f_title, fill=ACCENT)
    d.text((28, 58), "Current astronomical readout", font=f_label, fill=DIM)

    panel_top = 96
    panel_bottom = panel_top + 34 * len(rows) + 24
    d.rounded_rectangle((20, panel_top, WIDTH - 20, panel_bottom), radius=10, fill=PANEL)
    y = panel_top + 14
    for label, value in rows:
        d.text((40, y), label, font=f_label, fill=DIM)
        d.text((280, y), value, font=f_value, fill=FG)
        y += 34

    d.text((28, panel_bottom + 16), "Computed from standard ephemeris data.",
           font=f_small, fill=DIM)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=False)
    return sha256_file(out_path)


def stimulus_path(stimuli_dir: str | Path, scenario_id: str) -> Path:
    return Path(stimuli_dir) / f"{scenario_id}.png"


def render_all(scenarios: list[dict], stimuli_dir: str | Path) -> dict[str, str]:
    """Render one image per scenario; return {scenario_id: sha256}."""
    hashes = {}
    for sc in scenarios:
        path = stimulus_path(stimuli_dir, sc["scenario_id"])
        hashes[sc["scenario_id"]] = render_stimulus(sc, path)
    return hashes
