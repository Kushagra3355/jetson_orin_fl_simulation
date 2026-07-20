"""A dependency light federated coordinator and classroom dashboard server.

Run from the project root:
    python -m server.coordinator --config config/server.json

The API intentionally rejects raw patient fields. Secure mode aggregates only
after every configured client has submitted because the teaching masks need to
cancel. This server is a laboratory demonstrator, not production infrastructure.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from common.model import LinearPressureModel, MODEL_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_FIELDS = {
    "client_id",
    "round_number",
    "sample_count",
    "protected_weighted_delta",
    "local_mae",
    "local_rmse",
    "last_training_loss",
    "unclipped_delta_norm",
    "privacy_mode",
    "raw_rows_sent",
    "contains_patient_fields",
    "update_sha256",
}
FORBIDDEN_PATIENT_FIELDS = {
    "timestamp_utc",
    "respiratory_rate",
    "tidal_volume_ml",
    "leak_lpm",
    "spo2_percent",
    "ipap_cmh2o",
    "epap_cmh2o",
    "synthetic_target_adjustment_cmh2o",
}


@dataclass
class CoordinatorState:
    expected_clients: list[str]
    minimum_clients: int
    state_file: Path
    round_number: int = 0
    global_vector: list[float] = field(default_factory=lambda: LinearPressureModel.zeros().vector().tolist())
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    update_bytes_received: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def public_model(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "global_vector": self.global_vector,
            "model_size": MODEL_SIZE,
            "expected_clients": self.expected_clients,
        }

    def status(self) -> dict[str, Any]:
        return {
            **self.public_model(),
            "minimum_clients": self.minimum_clients,
            "pending_clients": sorted(self.pending),
            "pending_count": len(self.pending),
            "server_raw_patient_rows": 0,
            "update_bytes_received": self.update_bytes_received,
            "history": self.history[-20:],
            "safety_notice": "Synthetic educational simulation only. Not medical device software.",
        }

    def validate_payload(self, payload: dict[str, Any]) -> None:
        unknown = set(payload) - PAYLOAD_FIELDS
        if unknown:
            raise ValueError(f"Unexpected fields rejected: {sorted(unknown)}")
        forbidden = set(payload) & FORBIDDEN_PATIENT_FIELDS
        if forbidden:
            raise ValueError(f"Raw patient fields rejected: {sorted(forbidden)}")
        missing = PAYLOAD_FIELDS - set(payload)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        if payload["client_id"] not in self.expected_clients:
            raise ValueError("Client is not in expected_clients")
        if int(payload["round_number"]) != self.round_number:
            raise ValueError(f"Stale round. Server is on round {self.round_number}")
        if int(payload["raw_rows_sent"]) != 0 or bool(payload["contains_patient_fields"]):
            raise ValueError("Privacy contract violated: this API accepts model updates only")
        vector = payload["protected_weighted_delta"]
        if not isinstance(vector, list) or len(vector) != MODEL_SIZE:
            raise ValueError(f"protected_weighted_delta must contain {MODEL_SIZE} numbers")
        if int(payload["sample_count"]) <= 0:
            raise ValueError("sample_count must be positive")

    def submit(self, payload: dict[str, Any], encoded_size: int) -> dict[str, Any]:
        with self.lock:
            self.validate_payload(payload)
            client_id = str(payload["client_id"])
            if client_id in self.pending:
                raise ValueError("This client already submitted in the current round")
            if self.pending:
                existing_mode = next(iter(self.pending.values()))["privacy_mode"]
                if payload["privacy_mode"] != existing_mode:
                    raise ValueError("All clients in a round must use the same privacy_mode")
            self.pending[client_id] = payload
            self.update_bytes_received += encoded_size

            secure_round = "secure" in str(payload["privacy_mode"])
            needed = len(self.expected_clients) if secure_round else self.minimum_clients
            aggregated = len(self.pending) >= needed
            if aggregated:
                self._aggregate_round()
            return {
                "accepted": True,
                "aggregated": aggregated,
                "server_round": self.round_number,
                "pending_clients": sorted(self.pending),
            }

    def _aggregate_round(self) -> None:
        updates = list(self.pending.values())
        total_samples = sum(int(item["sample_count"]) for item in updates)
        protected_sum = np.sum(
            [np.asarray(item["protected_weighted_delta"], dtype=float) for item in updates],
            axis=0,
        )
        mean_delta = protected_sum / total_samples
        self.global_vector = (
            np.asarray(self.global_vector, dtype=float) + mean_delta
        ).round(12).tolist()
        weighted_mae = sum(float(item["local_mae"]) * int(item["sample_count"]) for item in updates) / total_samples
        self.history.append(
            {
                "completed_round": self.round_number,
                "participants": sorted(item["client_id"] for item in updates),
                "privacy_mode": updates[0]["privacy_mode"],
                "total_local_training_rows": total_samples,
                "raw_patient_rows_received": 0,
                "weighted_local_mae": round(weighted_mae, 6),
                "global_vector": self.global_vector,
            }
        )
        self.round_number += 1
        self.pending.clear()
        self.save()

    def reset(self) -> None:
        with self.lock:
            self.round_number = 0
            self.global_vector = LinearPressureModel.zeros().vector().tolist()
            self.pending.clear()
            self.history.clear()
            self.update_bytes_received = 0
            self.save()

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "round_number": self.round_number,
            "global_vector": self.global_vector,
            "history": self.history,
            "update_bytes_received": self.update_bytes_received,
        }
        self.state_file.write_text(json.dumps(content, indent=2), encoding="utf8")

    def restore(self) -> None:
        if not self.state_file.exists():
            return
        content = json.loads(self.state_file.read_text(encoding="utf8"))
        self.round_number = int(content.get("round_number", 0))
        self.global_vector = list(content.get("global_vector", self.global_vector))
        self.history = list(content.get("history", []))
        self.update_bytes_received = int(content.get("update_bytes_received", 0))


class CoordinatorHandler(BaseHTTPRequestHandler):
    state: CoordinatorState
    dashboard_dir = PROJECT_ROOT / "dashboard"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[coordinator] {self.address_string()} {format_string % args}")

    def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/model":
            self._json(self.state.public_model())
            return
        if path == "/api/status":
            self._json(self.state.status())
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                raise ValueError("Payload too large for a model update")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf8")) if raw else {}
            if path == "/api/update":
                self._json(self.state.submit(payload, len(raw)), HTTPStatus.ACCEPTED)
                return
            if path == "/api/reset":
                self.state.reset()
                self._json({"reset": True})
                return
            self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (self.dashboard_dir / relative).resolve()
        if self.dashboard_dir.resolve() not in candidate.parents and candidate != self.dashboard_dir.resolve():
            self._json({"error": "Invalid path"}, HTTPStatus.BAD_REQUEST)
            return
        if not candidate.is_file():
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        data = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def load_settings(config_path: Path) -> tuple[dict[str, Any], CoordinatorState]:
    settings = json.loads(config_path.read_text(encoding="utf8"))
    state_path = Path(settings["state_file"])
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    state = CoordinatorState(
        expected_clients=list(settings["expected_clients"]),
        minimum_clients=int(settings.get("minimum_clients", len(settings["expected_clients"]))),
        state_file=state_path,
    )
    state.restore()
    return settings, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated learning classroom coordinator")
    parser.add_argument("--config", default="config/server.json")
    parser.add_argument("--reset", action="store_true", help="Clear saved rounds before serving")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    settings, state = load_settings(config_path)
    if args.reset:
        state.reset()
    CoordinatorHandler.state = state
    server = ThreadingHTTPServer((settings.get("host", "0.0.0.0"), int(settings.get("port", 8000))), CoordinatorHandler)
    print(f"Dashboard: http://127.0.0.1:{server.server_port}")
    print(f"Expected clients: {', '.join(state.expected_clients)}")
    print("Safety: synthetic education only; never connect this code to a real BiPAP device.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCoordinator stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
