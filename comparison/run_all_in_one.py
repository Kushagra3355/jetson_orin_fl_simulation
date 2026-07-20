"""Run local only, centralized and federated learning on identical synthetic data.

This is the quickest classroom demonstration and also the automated smoke test
for the mathematics used by the distributed Jetson programs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.model import LinearPressureModel, metrics, train_model
from common.privacy import add_pairwise_masks, assert_masks_cancel, clip_and_noise_delta
from edge.bipap_simulator import generate_records, records_to_arrays, split_records, write_local_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENTS = (
    ("jetson01", "obstructive_pattern", 101),
    ("jetson02", "high_leak_pattern", 202),
    ("jetson03", "shallow_breath_pattern", 303),
    ("jetson04", "stable_pattern", 404),
)


def make_datasets(samples: int) -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for client_id, profile, seed in CLIENTS:
        rows = generate_records(client_id, profile, samples, seed)
        path = PROJECT_ROOT / "data" / client_id / "synthetic_bipap_records.csv"
        write_local_csv(rows, path)
        train_rows, test_rows = split_records(rows)
        x_train, y_train = records_to_arrays(train_rows)
        x_test, y_test = records_to_arrays(test_rows)
        datasets[client_id] = {
            "profile": profile,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "csv_path": str(path.relative_to(PROJECT_ROOT)),
        }
    return datasets


def evaluate(model: LinearPressureModel, datasets: dict[str, dict[str, Any]]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_client = {
        client_id: metrics(model, data["x_test"], data["y_test"])
        for client_id, data in datasets.items()
    }
    pooled_x = np.concatenate([data["x_test"] for data in datasets.values()])
    pooled_y = np.concatenate([data["y_test"] for data in datasets.values()])
    overall = metrics(model, pooled_x, pooled_y)
    overall["client_mae_gap"] = float(
        max(item["mae"] for item in per_client.values()) - min(item["mae"] for item in per_client.values())
    )
    return overall, per_client


def centralized(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pooled_x = np.concatenate([data["x_train"] for data in datasets.values()])
    pooled_y = np.concatenate([data["y_train"] for data in datasets.values()])
    model, _ = train_model(LinearPressureModel.zeros(), pooled_x, pooled_y, epochs=220, learning_rate=0.035)
    overall, per_client = evaluate(model, datasets)
    raw_rows = [row for data in datasets.values() for row in data["train_rows"]]
    return {
        "approach": "Centralized learning",
        **overall,
        "per_client": per_client,
        "raw_patient_rows_sent": len(raw_rows),
        "raw_payload_bytes": len(json.dumps(raw_rows).encode("utf8")),
        "model_values_sent": 0,
        "coordinator_can_read_individual_update": True,
        "coordinator_can_read_raw_patient_fields": True,
        "privacy_summary": "All training rows are copied to the coordinator.",
    }


def local_only(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_client: dict[str, dict[str, float]] = {}
    total_absolute = 0.0
    total_squared = 0.0
    total_count = 0
    for client_id, data in datasets.items():
        model, _ = train_model(
            LinearPressureModel.zeros(), data["x_train"], data["y_train"], epochs=220, learning_rate=0.035
        )
        result = metrics(model, data["x_test"], data["y_test"])
        per_client[client_id] = result
        prediction = model.predict(data["x_test"])
        residual = prediction - data["y_test"]
        total_absolute += float(np.abs(residual).sum())
        total_squared += float((residual**2).sum())
        total_count += len(residual)
    maes = [item["mae"] for item in per_client.values()]
    return {
        "approach": "Local only",
        "mae": total_absolute / total_count,
        "rmse": (total_squared / total_count) ** 0.5,
        "client_mae_gap": max(maes) - min(maes),
        "per_client": per_client,
        "raw_patient_rows_sent": 0,
        "raw_payload_bytes": 0,
        "model_values_sent": 0,
        "coordinator_can_read_individual_update": False,
        "coordinator_can_read_raw_patient_fields": False,
        "privacy_summary": "Nothing leaves a node, but no cross patient learning occurs.",
    }


def pair_secrets(client_ids: list[str]) -> dict[str, dict[str, str]]:
    secrets: dict[str, dict[str, str]] = {client_id: {} for client_id in client_ids}
    for index, first in enumerate(client_ids):
        for second in client_ids[index + 1 :]:
            shared = f"classroom-demo-secret-{first}-{second}"
            secrets[first][second] = shared
            secrets[second][first] = shared
    return secrets


def federated(
    datasets: dict[str, dict[str, Any]],
    rounds: int,
    use_dp: bool,
) -> dict[str, Any]:
    client_ids = sorted(datasets)
    secrets = pair_secrets(client_ids)
    global_model = LinearPressureModel.zeros()
    update_json_bytes = 0
    for round_number in range(rounds):
        masked: list[np.ndarray] = []
        unmasked: list[np.ndarray] = []
        total_samples = 0
        for index, client_id in enumerate(client_ids):
            data = datasets[client_id]
            local_model, _ = train_model(
                global_model,
                data["x_train"],
                data["y_train"],
                epochs=8,
                learning_rate=0.035,
            )
            delta = local_model.vector() - global_model.vector()
            if use_dp:
                delta, _ = clip_and_noise_delta(delta, clip_norm=1.0, noise_multiplier=0.02, seed=5000 + round_number * 31 + index)
            weighted = delta * len(data["x_train"])
            protected = add_pairwise_masks(client_id, weighted, secrets[client_id], round_number)
            masked.append(protected)
            unmasked.append(weighted)
            total_samples += len(data["x_train"])
            payload = {
                "client_id": client_id,
                "round_number": round_number,
                "sample_count": len(data["x_train"]),
                "protected_weighted_delta": protected.round(12).tolist(),
                "raw_rows_sent": 0,
            }
            update_json_bytes += len(json.dumps(payload).encode("utf8"))
        assert_masks_cancel(masked, np.sum(unmasked, axis=0), atol=1e-8)
        global_model = LinearPressureModel.from_vector(
            global_model.vector() + np.sum(masked, axis=0) / total_samples
        )

    overall, per_client = evaluate(global_model, datasets)
    mode = "Federated secure plus DP" if use_dp else "Federated secure aggregation"
    return {
        "approach": mode,
        **overall,
        "per_client": per_client,
        "raw_patient_rows_sent": 0,
        "raw_payload_bytes": 0,
        "model_update_bytes": update_json_bytes,
        "model_values_sent": rounds * len(client_ids) * len(global_model.vector()),
        "rounds": rounds,
        "coordinator_can_read_individual_update": False,
        "coordinator_can_read_raw_patient_fields": False,
        "privacy_summary": "Only masked model updates are combined. Raw rows remain local.",
    }


def write_results(results: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf8")
    csv_path = output_dir / "comparison.csv"
    fields = [
        "approach",
        "mae",
        "rmse",
        "client_mae_gap",
        "raw_patient_rows_sent",
        "raw_payload_bytes",
        "model_update_bytes",
        "model_values_sent",
        "coordinator_can_read_raw_patient_fields",
        "coordinator_can_read_individual_update",
    ]
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three learning arrangements")
    parser.add_argument("--samples-per-client", type=int, default=480)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    datasets = make_datasets(args.samples_per_client)
    results = [
        local_only(datasets),
        centralized(datasets),
        federated(datasets, args.rounds, use_dp=False),
        federated(datasets, args.rounds, use_dp=True),
    ]
    json_path, csv_path = write_results(results)
    print("\nApproach                              MAE     Raw rows sent  Raw fields at server")
    print("=" * 84)
    for item in results:
        print(
            f"{item['approach']:<36} {item['mae']:<7.3f} "
            f"{item['raw_patient_rows_sent']:<14} {str(item['coordinator_can_read_raw_patient_fields']):<5}"
        )
    print(f"\nDetailed JSON: {json_path}")
    print(f"Comparison CSV: {csv_path}")
    print("Interpretation: federated learning improves data minimization, not guaranteed accuracy.")
    print("Safety: synthetic educational simulation only; never connect to a real breathing device.")


if __name__ == "__main__":
    main()
