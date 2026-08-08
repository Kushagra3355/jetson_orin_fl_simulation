"""Create one server config and one private config for every Jetson node."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("obstructive_pattern", "high_leak_pattern", "shallow_breath_pattern", "stable_pattern")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", nargs="+", default=["jetson01", "jetson02", "jetson03", "jetson04"])
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--privacy-mode", choices=["weights_only", "secure", "dp", "secure_dp"], default="secure_dp")
    parser.add_argument("--timestamp-seed", action="store_true", help="Use dynamic timestamp-based seed for synthetic data generation")
    parser.add_argument("--output-dir", default="config/clients")
    args = parser.parse_args()

    clients = sorted(set(args.clients))
    if len(clients) < 2:
        raise SystemExit("Use at least two clients")
    pair_secrets: dict[tuple[str, str], str] = {}
    for index, first in enumerate(clients):
        for second in clients[index + 1 :]:
            pair_secrets[(first, second)] = secrets.token_hex(24)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, client_id in enumerate(clients):
        peers = {}
        for peer_id in clients:
            if peer_id == client_id:
                continue
            key = tuple(sorted((client_id, peer_id)))
            peers[peer_id] = pair_secrets[key]
        config = {
            "client_id": client_id,
            "patient_profile": PROFILES[index % len(PROFILES)],
            "seed": "timestamp" if args.timestamp_seed else 101 + index * 101,
            "server_url": args.server_url,

            "samples": 480,
            "local_epochs": 8,
            "learning_rate": 0.035,
            "privacy_mode": args.privacy_mode,
            "clip_norm": 1.0,
            "noise_multiplier": 0.02,
            "peer_secrets": peers,
        }
        path = output_dir / f"{client_id}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf8")
        print(path)

    server_config = {
        "expected_clients": clients,
        "minimum_clients": len(clients),
        "host": "0.0.0.0",
        "port": 8000,
        "state_file": "state/server_state.json",
    }
    server_path = PROJECT_ROOT / "config" / "server.json"
    server_path.write_text(json.dumps(server_config, indent=2), encoding="utf8")
    print(server_path)
    print("Copy only each node's own JSON file to that Jetson board.")


if __name__ == "__main__":
    main()
