"""Run one synthetic patient node on a Jetson Orin board.

The raw CSV is written under data/<client_id>/ and is never included in the
network payload. Use Ctrl+C to stop watch mode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from edge.bipap_simulator import generate_records, records_to_arrays, split_records, write_local_csv
from edge.local_training import train_local_update


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf8"))
    except HTTPError as exc:
        message = exc.read().decode("utf8", errors="replace")
        raise RuntimeError(f"Server rejected the request: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach coordinator at {url}: {exc.reason}") from exc


def append_audit_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def prepare_local_data(settings: dict[str, Any]) -> tuple[Any, Any, Any, Any, Path]:
    rows = generate_records(
        client_id=settings["client_id"],
        profile_name=settings["patient_profile"],
        samples=int(settings.get("samples", 480)),
        seed=int(settings.get("seed", 101)),
    )
    data_path = PROJECT_ROOT / "data" / settings["client_id"] / "synthetic_bipap_records.csv"
    write_local_csv(rows, data_path)
    train_rows, test_rows = split_records(rows)
    x_train, y_train = records_to_arrays(train_rows)
    x_test, y_test = records_to_arrays(test_rows)
    return x_train, y_train, x_test, y_test, data_path


def run_round(settings: dict[str, Any], arrays: tuple[Any, Any, Any, Any], audit_path: Path) -> dict[str, Any]:
    server_url = str(settings["server_url"]).rstrip("/")
    model_response = http_json(f"{server_url}/api/model")
    round_number = int(model_response["round_number"])
    payload, _ = train_local_update(
        client_id=settings["client_id"],
        round_number=round_number,
        global_vector=model_response["global_vector"],
        x_train=arrays[0],
        y_train=arrays[1],
        x_test=arrays[2],
        y_test=arrays[3],
        local_epochs=int(settings.get("local_epochs", 8)),
        learning_rate=float(settings.get("learning_rate", 0.035)),
        privacy_mode=settings.get("privacy_mode", "secure_dp"),
        clip_norm=float(settings.get("clip_norm", 1.0)),
        noise_multiplier=float(settings.get("noise_multiplier", 0.02)),
        seed=int(settings.get("seed", 101)),
        peer_secrets=dict(settings.get("peer_secrets", {})),
    )
    response = http_json(f"{server_url}/api/update", payload)
    append_audit_log(
        audit_path,
        {
            "event": "model_update_sent",
            "round_number": round_number,
            "raw_rows_sent": 0,
            "payload_fields": sorted(payload),
            "payload_size_bytes": len(json.dumps(payload).encode("utf8")),
            "update_sha256": payload["update_sha256"],
            "server_response": response,
        },
    )
    print(
        f"{settings['client_id']} submitted round {round_number}: "
        f"local MAE {payload['local_mae']:.3f}, raw rows sent 0, "
        f"privacy {payload['privacy_mode']}"
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic BiPAP federated patient node")
    parser.add_argument("--config", required=True, help="Path to this Jetson node JSON config")
    parser.add_argument("--watch", action="store_true", help="Join every new round until stopped")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    settings = json.loads(config_path.read_text(encoding="utf8"))
    x_train, y_train, x_test, y_test, data_path = prepare_local_data(settings)
    arrays = (x_train, y_train, x_test, y_test)
    audit_path = data_path.parent / "privacy_audit.jsonl"
    print(f"Local synthetic records: {data_path}")
    print("These raw records remain on this node. EDUCATIONAL SIMULATION ONLY.")

    submitted_round = -1
    while True:
        try:
            current = http_json(f"{str(settings['server_url']).rstrip('/')}/api/model")
            if int(current["round_number"]) > submitted_round:
                submitted_round = int(current["round_number"])
                run_round(settings, arrays, audit_path)
                if not args.watch:
                    break
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("\nPatient node stopped")
            break
        except RuntimeError as exc:
            print(exc)
            if not args.watch:
                raise
            time.sleep(max(2.0, args.poll_seconds))


if __name__ == "__main__":
    main()
