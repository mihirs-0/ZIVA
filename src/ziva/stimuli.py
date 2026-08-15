"""Deterministic multimodal stimulus renderer.

Converts a scenario's physical state into a neutral "astronomy app" style
image card. The image:

* is generated purely from scenario JSON (no randomness, no timestamps of
  rendering, no treatment-specific content);
* never states a naked-eye visibility verdict;
* includes a qualitative sky label only because it is derived
  deterministically from solar altitude (same rule as `classify_sky_regime`);
* is byte-identical across renders of the same scenario (verified by tests),
  and the SAME image file is used for every valence treatment of a scenario.

Fonts: DejaVu Sans as bundled with matplotlib (pinned dependency), so glyph
rendering is deterministic for a given environment.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .util import sha256_file

WIDTH, HEIGHT = 800, 620

BG = (16, 24, 40)
PANEL = (26, 36, 58)
FG = (232, 236, 244)
DIM = (150, 160, 180)
ACCENT = (110, 170, 255)
SUN_COLOR = (255, 200, 80)
MOON_COLOR = (210, 215, 230)
HORIZON_COLOR = (90, 100, 125)

REGIME_LABELS = {
    "daylight": "bright daylight",
    "civil_twilight": "civil twilight",
    "nautical_twilight": "nautical twilight",
    "astronomical_twilight": "astronomical twilight",
    "night": "night",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib

    fonts_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(fonts_dir / name), size)


def _fmt(value: float, digits: int = 1, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}"


def render_stimulus(scenario: dict, out_path: str | Path) -> str:
    """Render the scenario card PNG. Returns the sha256 of the written file."""
    ps = scenario["physical_state"]
    loc = scenario["location"]

    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)

    f_title = _font(26, bold=True)
    f_label = _font(16)
    f_value = _font(19, bold=True)
    f_small = _font(14)

    # Header
    d.text((28, 22), "SKY DATA", font=f_title, fill=ACCENT)
    d.text((28, 58), loc["name"], font=f_value, fill=FG)
    d.text((28, 86), f"Local time: {scenario['local_time']}", font=f_label, fill=DIM)
    d.text((28, 108), f"UTC: {scenario['timestamp_utc']}", font=f_label, fill=DIM)
    regime = REGIME_LABELS[ps["sky_regime"]]
    d.text((28, 134), f"Sky: clear -- {regime}", font=f_label, fill=FG)

    # Data panel (left column)
    panel_top = 176
    d.rounded_rectangle((20, panel_top, 392, panel_top + 320), radius=10, fill=PANEL)
    rows = [
        ("Sun altitude", _fmt(ps["sun_altitude_deg"], 1, "°")),
        ("Sun azimuth", _fmt(ps["sun_azimuth_deg"], 1, "°")),
        ("Moon altitude", _fmt(ps["moon_altitude_deg"], 1, "°")),
        ("Moon azimuth", _fmt(ps["moon_azimuth_deg"], 1, "°")),
        ("Moon illumination", _fmt(100 * ps["moon_illumination_fraction"], 1, "%")),
        ("Sun-Moon separation", _fmt(ps["sun_moon_elongation_deg"], 1, "°")),
        ("Moon distance", f"{ps['moon_distance_km']:,.0f} km"),
        ("Latitude / longitude", f"{loc['latitude_deg']:.2f}, {loc['longitude_deg']:.2f}"),
    ]
    y = panel_top + 18
    for label, value in rows:
        d.text((40, y), label, font=f_label, fill=DIM)
        d.text((240, y), value, font=f_value, fill=FG)
        y += 37

    # Schematic sky view (right column): altitude vs. azimuth-difference chart.
    box = (412, panel_top, 780, panel_top + 320)
    d.rounded_rectangle(box, radius=10, fill=PANEL)
    d.text((box[0] + 16, box[1] + 12), "Schematic sky view", font=f_label, fill=DIM)

    cx0, cy0, cx1, cy1 = box[0] + 24, box[1] + 44, box[2] - 24, box[3] - 40
    horizon_y = cy0 + (cy1 - cy0) * 0.5  # altitude 0 at mid-height; range -90..+90

    def sky_xy(alt: float, az: float) -> tuple[float, float]:
        x = cx0 + (az % 360.0) / 360.0 * (cx1 - cx0)
        yy = horizon_y - (alt / 90.0) * (cy1 - cy0) * 0.5
        return x, yy

    d.line((cx0, horizon_y, cx1, horizon_y), fill=HORIZON_COLOR, width=2)
    d.text((cx0, horizon_y + 6), "horizon (alt 0°)", font=f_small, fill=HORIZON_COLOR)
    d.text((cx0, cy1 + 8), "azimuth 0° (N)", font=f_small, fill=DIM)
    d.text((cx1 - 90, cy1 + 8), "azimuth 360°", font=f_small, fill=DIM)

    sx, sy = sky_xy(ps["sun_altitude_deg"], ps["sun_azimuth_deg"])
    mx, my = sky_xy(ps["moon_altitude_deg"], ps["moon_azimuth_deg"])
    d.ellipse((sx - 10, sy - 10, sx + 10, sy + 10), fill=SUN_COLOR)
    d.text((sx - 12, sy + 12), "Sun", font=f_small, fill=SUN_COLOR)

    # Moon disc with illuminated-fraction shading (schematic)
    r = 9
    d.ellipse((mx - r, my - r, mx + r, my + r), outline=MOON_COLOR, width=2)
    frac = ps["moon_illumination_fraction"]
    if frac > 0.001:
        # draw a pie slice proportional to illumination (schematic, not phase-accurate)
        sweep = 360.0 * frac
        d.pieslice((mx - r, my - r, mx + r, my + r), start=-90 - sweep / 2, end=-90 + sweep / 2,
                   fill=MOON_COLOR)
    d.text((mx - 16, my + 12), "Moon", font=f_small, fill=MOON_COLOR)

    # Footer
    d.text(
        (28, HEIGHT - 72),
        f"Moon above horizon: {'yes' if ps['moon_above_horizon'] else 'no'}    "
        f"Sun above horizon: {'yes' if ps['sun_above_horizon'] else 'no'}",
        font=f_label,
        fill=FG,
    )
    d.text(
        (28, HEIGHT - 44),
        "Computed from standard ephemeris data.",
        font=f_small,
        fill=DIM,
    )

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
