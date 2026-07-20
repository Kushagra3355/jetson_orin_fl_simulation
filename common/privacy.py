"""Educational update protection helpers.

Pairwise masks cancel only when every configured participant contributes.
This demonstrates secure aggregation but is not production cryptography.
Differential privacy here is a client level teaching mechanism.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


def clip_and_noise_delta(
    delta: np.ndarray,
    clip_norm: float,
    noise_multiplier: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    values = np.asarray(delta, dtype=float)
    norm = float(np.linalg.norm(values))
    scale = min(1.0, float(clip_norm) / max(norm, 1e-12))
    clipped = values * scale
    if noise_multiplier > 0:
        rng = np.random.default_rng(seed)
        clipped = clipped + rng.normal(0.0, noise_multiplier * clip_norm, size=values.shape)
    return clipped, norm


def _mask_from_secret(secret: str, round_number: int, size: int) -> np.ndarray:
    material = f"{secret}|round={round_number}|size={size}".encode("utf8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(seed).normal(0.0, 2.0, size=size)


def add_pairwise_masks(
    client_id: str,
    weighted_delta: np.ndarray,
    peer_secrets: Mapping[str, str],
    round_number: int,
) -> np.ndarray:
    protected = np.asarray(weighted_delta, dtype=float).copy()
    for peer_id, secret in sorted(peer_secrets.items()):
        mask = _mask_from_secret(secret, round_number, protected.size)
        protected += mask if client_id < peer_id else -mask
    return protected


def update_digest(vector: np.ndarray) -> str:
    canonical = np.asarray(vector, dtype="<f8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


def assert_masks_cancel(masked_vectors: list[np.ndarray], unmasked_sum: np.ndarray, atol: float = 1e-9) -> None:
    if not np.allclose(np.sum(masked_vectors, axis=0), unmasked_sum, atol=atol):
        raise AssertionError("Pairwise masks did not cancel. Check client set and secrets.")
