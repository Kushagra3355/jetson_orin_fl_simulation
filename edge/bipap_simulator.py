"""Generate synthetic BiPAP breath records for one classroom patient node.

EDUCATIONAL SIMULATION ONLY. The values are synthetic and are not validated
for diagnosis, treatment, alarm generation or control of medical equipment.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from common.model import FEATURES, normalize_features


@dataclass(frozen=True)
class PatientProfile:
    respiratory_rate: float
    tidal_volume_ml: float
    leak_lpm: float
    spo2_percent: float
    personal_offset: float
    variability: float


PROFILES = {
    "obstructive_pattern": PatientProfile(20.5, 390.0, 18.0, 92.5, 0.55, 1.0),
    "high_leak_pattern": PatientProfile(18.0, 430.0, 28.0, 93.5, 0.30, 1.2),
    "shallow_breath_pattern": PatientProfile(23.0, 310.0, 11.0, 91.8, 0.85, 0.9),
    "stable_pattern": PatientProfile(15.5, 540.0, 7.0, 96.0, -0.35, 0.7),
}

CSV_FIELDS = (
    "timestamp_utc",
    "patient_node",
    "breath_number",
    "respiratory_rate",
    "tidal_volume_ml",
    "leak_lpm",
    "spo2_percent",
    "epap_cmh2o",
    "ipap_cmh2o",
    "delivered_pressure_cmh2o",
    "synthetic_target_adjustment_cmh2o",
)


def pressure_wave(phase: float, epap: float, ipap: float) -> float:
    """Smoothly move between simulated EPAP and IPAP during one breath."""
    inhale_gate = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase - math.pi / 2.0)
    return float(epap + (ipap - epap) * inhale_gate)


def generate_records(
    client_id: str,
    profile_name: str,
    samples: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name}. Choose from {sorted(PROFILES)}")
    if samples < 40:
        raise ValueError("At least 40 samples are required for a useful demonstration")

    profile = PROFILES[profile_name]
    rng = np.random.default_rng(seed)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    rows: list[dict[str, float | int | str]] = []
    for index in range(samples):
        slow_drift = math.sin(index / 37.0)
        rr = rng.normal(profile.respiratory_rate + 0.7 * slow_drift, 1.4 * profile.variability)
        tidal = rng.normal(profile.tidal_volume_ml - 18.0 * slow_drift, 35.0 * profile.variability)
        leak = max(0.0, rng.normal(profile.leak_lpm + 1.8 * slow_drift, 2.3 * profile.variability))
        spo2 = np.clip(rng.normal(profile.spo2_percent - 0.25 * slow_drift, 0.65), 86.0, 99.0)

        feature_row = np.array([[rr, tidal, leak, spo2]], dtype=float)
        z = normalize_features(feature_row)[0]
        true_weights = np.array([0.52, -0.44, 0.35, -0.62], dtype=float)
        target = float(np.clip(z @ true_weights + profile.personal_offset + rng.normal(0.0, 0.16), -3.0, 3.0))

        epap = 6.0
        ipap = float(np.clip(14.0 + target, 9.0, 20.0))
        phase = (index % 24) / 24.0
        rows.append(
            {
                "timestamp_utc": (start + timedelta(seconds=index * 2)).isoformat(),
                "patient_node": client_id,
                "breath_number": index + 1,
                "respiratory_rate": round(float(rr), 4),
                "tidal_volume_ml": round(float(tidal), 4),
                "leak_lpm": round(float(leak), 4),
                "spo2_percent": round(float(spo2), 4),
                "epap_cmh2o": epap,
                "ipap_cmh2o": round(ipap, 4),
                "delivered_pressure_cmh2o": round(pressure_wave(phase, epap, ipap), 4),
                "synthetic_target_adjustment_cmh2o": round(target, 6),
            }
        )
    return rows


def write_local_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def records_to_arrays(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    y = np.array([float(row["synthetic_target_adjustment_cmh2o"]) for row in rows], dtype=float)
    return normalize_features(x), y


def split_records(rows: list[dict[str, object]], test_fraction: float = 0.2) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split = max(1, int(len(rows) * (1.0 - test_fraction)))
    return rows[:split], rows[split:]
