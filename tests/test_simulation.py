from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from common.model import LinearPressureModel
from common.privacy import add_pairwise_masks, assert_masks_cancel
from edge.bipap_simulator import generate_records, records_to_arrays, split_records
from edge.local_training import train_local_update
from server.coordinator import CoordinatorState


class SimulationTests(unittest.TestCase):
    def test_synthetic_records_and_update_contain_no_raw_fields(self) -> None:
        rows = generate_records("jetson01", "obstructive_pattern", 80, 101)
        train_rows, test_rows = split_records(rows)
        x_train, y_train = records_to_arrays(train_rows)
        x_test, y_test = records_to_arrays(test_rows)
        payload, _ = train_local_update(
            client_id="jetson01",
            round_number=0,
            global_vector=LinearPressureModel.zeros().vector().tolist(),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            local_epochs=2,
            learning_rate=0.03,
            privacy_mode="weights_only",
            clip_norm=1.0,
            noise_multiplier=0.0,
            seed=1,
            peer_secrets={},
        )
        forbidden = {"respiratory_rate", "tidal_volume_ml", "leak_lpm", "spo2_percent"}
        self.assertFalse(forbidden.intersection(payload))
        self.assertEqual(payload["raw_rows_sent"], 0)

    def test_pairwise_secure_masks_cancel(self) -> None:
        client_ids = ["jetson01", "jetson02", "jetson03"]
        values = {client_id: np.arange(5, dtype=float) + index for index, client_id in enumerate(client_ids)}
        secrets = {
            "jetson01": {"jetson02": "a", "jetson03": "b"},
            "jetson02": {"jetson01": "a", "jetson03": "c"},
            "jetson03": {"jetson01": "b", "jetson02": "c"},
        }
        masked = [add_pairwise_masks(client_id, values[client_id], secrets[client_id], 4) for client_id in client_ids]
        assert_masks_cancel(masked, sum(values.values()))

    def test_coordinator_rejects_raw_patient_field(self) -> None:
        with TemporaryDirectory() as temporary:
            state = CoordinatorState(["jetson01"], 1, Path(temporary) / "state.json")
            payload = {
                "client_id": "jetson01",
                "round_number": 0,
                "sample_count": 10,
                "protected_weighted_delta": [0.0] * 5,
                "local_mae": 0.1,
                "local_rmse": 0.2,
                "last_training_loss": 0.1,
                "unclipped_delta_norm": 0.2,
                "privacy_mode": "weights_only",
                "raw_rows_sent": 0,
                "contains_patient_fields": False,
                "update_sha256": "demo",
                "spo2_percent": 93.0,
            }
            with self.assertRaises(ValueError):
                state.validate_payload(payload)

    def test_four_client_secure_round_aggregates_without_raw_rows(self) -> None:
        clients = ["jetson01", "jetson02", "jetson03", "jetson04"]
        secrets = {
            client: {
                peer: f"secret-{'-'.join(sorted((client, peer)))}"
                for peer in clients if peer != client
            }
            for client in clients
        }
        base_delta = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        with TemporaryDirectory() as temporary:
            state = CoordinatorState(clients, 4, Path(temporary) / "state.json")
            for index, client in enumerate(clients):
                weighted = (base_delta + index * 0.01) * 20
                protected = add_pairwise_masks(client, weighted, secrets[client], 0)
                payload = {
                    "client_id": client,
                    "round_number": 0,
                    "sample_count": 20,
                    "protected_weighted_delta": protected.tolist(),
                    "local_mae": 0.2,
                    "local_rmse": 0.3,
                    "last_training_loss": 0.1,
                    "unclipped_delta_norm": 0.2,
                    "privacy_mode": "secure_dp",
                    "raw_rows_sent": 0,
                    "contains_patient_fields": False,
                    "update_sha256": f"demo-{index}",
                }
                response = state.submit(payload, encoded_size=300)
            self.assertTrue(response["aggregated"])
            self.assertEqual(state.round_number, 1)
            self.assertEqual(state.status()["server_raw_patient_rows"], 0)
            self.assertEqual(state.history[-1]["raw_patient_rows_received"], 0)


if __name__ == "__main__":
    unittest.main()
