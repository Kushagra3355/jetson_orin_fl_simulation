"""A transparent linear model for the classroom BiPAP simulation.

The target is a synthetic pressure support correction. It is not a clinical
recommendation and this module must never control a real breathing device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FEATURES = ("respiratory_rate", "tidal_volume_ml", "leak_lpm", "spo2_percent")
FEATURE_MEAN = np.array([18.0, 450.0, 12.0, 94.0], dtype=float)
FEATURE_SCALE = np.array([5.0, 150.0, 8.0, 3.0], dtype=float)
MODEL_SIZE = len(FEATURES) + 1


def normalize_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - FEATURE_MEAN) / FEATURE_SCALE


@dataclass
class LinearPressureModel:
    weights: np.ndarray
    bias: float

    @classmethod
    def zeros(cls) -> "LinearPressureModel":
        return cls(np.zeros(len(FEATURES), dtype=float), 0.0)

    @classmethod
    def from_vector(cls, vector: Iterable[float]) -> "LinearPressureModel":
        values = np.asarray(list(vector), dtype=float)
        if values.shape != (MODEL_SIZE,):
            raise ValueError(f"Expected {MODEL_SIZE} model values, received {values.shape}")
        return cls(values[:-1].copy(), float(values[-1]))

    def vector(self) -> np.ndarray:
        return np.concatenate([self.weights, np.array([self.bias])])

    def predict(self, normalized_x: np.ndarray, clip: bool = True) -> np.ndarray:
        prediction = np.asarray(normalized_x, dtype=float) @ self.weights + self.bias
        return np.clip(prediction, -3.0, 3.0) if clip else prediction


def train_model(
    initial: LinearPressureModel,
    normalized_x: np.ndarray,
    y: np.ndarray,
    epochs: int = 8,
    learning_rate: float = 0.035,
    l2: float = 0.001,
) -> tuple[LinearPressureModel, list[float]]:
    """Train with deterministic full batch gradient descent."""
    x = np.asarray(normalized_x, dtype=float)
    target = np.asarray(y, dtype=float)
    if len(x) != len(target) or len(x) == 0:
        raise ValueError("Training arrays must be nonempty and have equal length")

    model = LinearPressureModel(initial.weights.copy(), initial.bias)
    losses: list[float] = []
    for _ in range(int(epochs)):
        residual = model.predict(x, clip=False) - target
        grad_w = (2.0 / len(x)) * (x.T @ residual) + 2.0 * l2 * model.weights
        grad_b = float((2.0 / len(x)) * residual.sum())
        model.weights -= learning_rate * grad_w
        model.bias -= learning_rate * grad_b
        losses.append(float(np.mean(residual**2)))
    return model, losses


def metrics(model: LinearPressureModel, normalized_x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=float)
    prediction = model.predict(normalized_x)
    residual = prediction - target
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }
