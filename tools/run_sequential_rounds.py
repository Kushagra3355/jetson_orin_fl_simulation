"""Sequential runner for federated learning patient nodes.

Runs clients in strict series (jetson01 -> jetson02 -> jetson03 -> jetson04)
for a specified number of rounds (default: 10), then stops.

Prerequisites:
  1. Generate client configs:
     python tools/generate_client_configs.py --clients jetson01 jetson02 jetson03 jetson04 --server-url http://localhost:8000 --privacy-mode secure_dp
  2. Start the coordinator server in a separate terminal:
     python -m server.coordinator --config config/server.json --reset
  3. Run this script:
     python tools/run_sequential_rounds.py --rounds 10
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge.patient_node import prepare_local_data, run_round




def get_server_round(server_url: str) -> int:
    url = f"{server_url.rstrip('/')}/api/model"
    request = Request(url, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf8"))
        return int(data["round_number"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FL patient nodes in strict sequential series")
    parser.add_argument("--clients", nargs="+", default=["jetson01", "jetson02", "jetson03", "jetson04"])
    parser.add_argument("--rounds", type=int, default=10, help="Number of total rounds to complete (default: 10)")
    parser.add_argument("--timestamp-seed", action="store_true", help="Use current timestamp as random seed for dynamic data generation")
    parser.add_argument("--round-delay", type=float, default=1.0, help="Delay in seconds between completed rounds (default: 1.0)")
    parser.add_argument("--node-delay", type=float, default=0.2, help="Delay in seconds between individual nodes (default: 0.2)")
    args = parser.parse_args()

    client_data: list[tuple[dict[str, Any], tuple[Any, Any, Any, Any], Path]] = []
    
    print(f"Loading local data for {len(args.clients)} nodes: {', '.join(args.clients)}...")
    for client_id in args.clients:
        config_path = PROJECT_ROOT / "config" / "clients" / f"{client_id}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}. Run generate_client_configs.py first.")
        settings = json.loads(config_path.read_text(encoding="utf8"))
        if args.timestamp_seed:
            settings["use_timestamp_seed"] = True
        x_tr, y_tr, x_te, y_te, data_path = prepare_local_data(settings)

        arrays = (x_tr, y_tr, x_te, y_te)
        audit_path = data_path.parent / "privacy_audit.jsonl"
        client_data.append((settings, arrays, audit_path))

    server_url = client_data[0][0]["server_url"]
    print(f"Connecting to coordinator at {server_url}...")
    
    try:
        initial_round = get_server_round(server_url)
    except (URLError, HTTPError, RuntimeError) as exc:
        print(f"Error: Unable to reach coordinator at {server_url}. Make sure python -m server.coordinator is running!")
        return

    print(f"Starting sequential execution for {args.rounds} rounds (Initial Server Round: {initial_round})...\n")

    start_time = time.time()

    for target_round in range(1, args.rounds + 1):
        print(f"==========================================")
        print(f"        ROUND {target_round} / {args.rounds} (Sequential)")
        print(f"==========================================")
        
        for settings, arrays, audit_path in client_data:
            client_id = settings["client_id"]
            print(f"  --> Running {client_id}...", end=" ", flush=True)
            run_round(settings, arrays, audit_path)
            time.sleep(args.node_delay)
        
        print(f"  [Round {target_round} Complete] -> Coordinator updated global model!\n")
        if target_round < args.rounds and args.round_delay > 0:
            time.sleep(args.round_delay)

    elapsed = time.time() - start_time
    print(f"Completed all {args.rounds} sequential rounds in {elapsed:.2f} seconds.")



if __name__ == "__main__":
    main()
