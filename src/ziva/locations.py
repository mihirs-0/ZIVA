"""Fixed pool of diverse observer locations for scenario generation.

IANA timezone names are used to render local times; the physical state itself
is always computed from the UTC timestamp, so timezone rendering can never
change the physics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Location:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    timezone: str

    def to_dict(self) -> dict:
        return asdict(self)


LOCATIONS: list[Location] = [
    Location("Davis, California, USA", 38.5449, -121.7405, 16, "America/Los_Angeles"),
    Location("Quito, Ecuador", -0.1807, -78.4678, 2850, "America/Guayaquil"),
    Location("Reykjavik, Iceland", 64.1466, -21.9426, 15, "Atlantic/Reykjavik"),
    Location("Nairobi, Kenya", -1.2921, 36.8219, 1795, "Africa/Nairobi"),
    Location("Ushuaia, Argentina", -54.8019, -68.3030, 23, "America/Argentina/Ushuaia"),
    Location("Tromso, Norway", 69.6492, 18.9553, 10, "Europe/Oslo"),
    Location("Singapore", 1.3521, 103.8198, 15, "Asia/Singapore"),
    Location("Alice Springs, Australia", -23.6980, 133.8807, 545, "Australia/Darwin"),
    Location("La Paz, Bolivia", -16.4897, -68.1193, 3640, "America/La_Paz"),
    Location("Cairo, Egypt", 30.0444, 31.2357, 23, "Africa/Cairo"),
    Location("Sapporo, Japan", 43.0618, 141.3545, 26, "Asia/Tokyo"),
    Location("Cape Town, South Africa", -33.9249, 18.4241, 25, "Africa/Johannesburg"),
    Location("Anchorage, Alaska, USA", 61.2181, -149.9003, 31, "America/Anchorage"),
    Location("Mumbai, India", 19.0760, 72.8777, 14, "Asia/Kolkata"),
    Location("Honolulu, Hawaii, USA", 21.3069, -157.8583, 6, "Pacific/Honolulu"),
    Location("Edinburgh, Scotland, UK", 55.9533, -3.1883, 47, "Europe/London"),
    Location("Santiago, Chile", -33.4489, -70.6693, 570, "America/Santiago"),
    Location("Ulaanbaatar, Mongolia", 47.8864, 106.9057, 1350, "Asia/Ulaanbaatar"),
    Location("Wellington, New Zealand", -41.2865, 174.7762, 13, "Pacific/Auckland"),
    Location("Denver, Colorado, USA", 39.7392, -104.9903, 1609, "America/Denver"),
    Location("Lagos, Nigeria", 6.5244, 3.3792, 41, "Africa/Lagos"),
    Location("Helsinki, Finland", 60.1699, 24.9384, 16, "Europe/Helsinki"),
    Location("Mexico City, Mexico", 19.4326, -99.1332, 2240, "America/Mexico_City"),
    Location("Perth, Australia", -31.9505, 115.8605, 20, "Australia/Perth"),
]

DAVIS = LOCATIONS[0]
