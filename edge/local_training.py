"""Train locally and construct the only payload allowed to leave a patient node."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from common.model import LinearPressureModel, metrics, train_model
from common.privacy import add_pairwise_masks, clip_and_noise_delta, update_digest


ALLOWED_PRIVACY_MODES = {"weights_only", "secure", "dp", "secure_dp"}


def train_local_update(
    client_id: str,
    round_number: int,
    global_vector: list[float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    local_epochs: int,
    learning_rate: float,
    privacy_mode: str,
    clip_norm: float,
    noise_multiplier: float,
    seed: int,
    peer_secrets: Mapping[str, str],
) -> tuple[dict[str, object], LinearPressureModel]:
    if privacy_mode not in ALLOWED_PRIVACY_MODES:
        raise ValueError(f"privacy_mode must be one of {sorted(ALLOWED_PRIVACY_MODES)}")

    global_model = LinearPressureModel.from_vector(global_vector)
    local_model, losses = train_model(
        global_model,
        x_train,
        y_train,
        epochs=local_epochs,
        learning_rate=learning_rate,
    )
    raw_delta = local_model.vector() - global_model.vector()
    original_norm = float(np.linalg.norm(raw_delta))

    protected_delta = raw_delta
    if "dp" in privacy_mode:
        protected_delta, original_norm = clip_and_noise_delta(
            raw_delta,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            seed=seed + round_number * 1009,
        )

    weighted_delta = protected_delta * len(x_train)
    if "secure" in privacy_mode:
        if not peer_secrets:
            raise ValueError("Secure mode needs pairwise peer secrets")
        transmitted = add_pairwise_masks(client_id, weighted_delta, peer_secrets, round_number)
    else:
        transmitted = weighted_delta

    test_metrics = metrics(local_model, x_test, y_test)
    payload = {
        "client_id": client_id,
        "round_number": int(round_number),
        "sample_count": int(len(x_train)),
        "protected_weighted_delta": [round(float(value), 12) for value in transmitted],
        "local_mae": round(test_metrics["mae"], 6),
        "local_rmse": round(test_metrics["rmse"], 6),
        "last_training_loss": round(float(losses[-1]), 8),
        "unclipped_delta_norm": round(original_norm, 8),
        "privacy_mode": privacy_mode,
        "raw_rows_sent": 0,
        "contains_patient_fields": False,
        "update_sha256": update_digest(transmitted),
    }
    return payload, local_model
