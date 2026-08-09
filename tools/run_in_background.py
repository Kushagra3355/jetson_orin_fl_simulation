"""Background launcher for FL Server and Sequential Rounds.

Runs all steps in one command, launching the server and training loop in the background,
redirecting output to log files, and returning control to the terminal immediately.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Step 1: Generating client configurations...")
    gen_cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "generate_client_configs.py"), "--timestamp-seed"]
    subprocess.run(gen_cmd, check=True)
    print("Client configurations generated successfully.\n")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    server_log_path = logs_dir / "server.log"
    training_log_path = logs_dir / "training.log"

    print("Step 2: Starting server/coordinator in background...")
    server_log_file = open(server_log_path, "w", encoding="utf-8")
    server_cmd = [sys.executable, "-m", "server.coordinator", "--config", "config/server.json", "--reset"]
    
    # Platform-specific background flags
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        close_fds=True if sys.platform != "win32" else False
    )
    print(f"  -> Server running in background (PID: {server_proc.pid})")
    print(f"  -> Server output logged to: {server_log_path}")

    # Wait for server startup
    time.sleep(2.5)

    print("\nStep 3: Starting sequential FL training rounds in background...")
    training_log_file = open(training_log_path, "w", encoding="utf-8")
    training_cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "tools" / "run_sequential_rounds.py"),
        "--rounds", "10",
        "--timestamp-seed",
        "--round-delay", "3.0",
        "--node-delay", "1.0"
    ]
    
    training_proc = subprocess.Popen(
        training_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=training_log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        close_fds=True if sys.platform != "win32" else False
    )
    print(f"  -> Training running in background (PID: {training_proc.pid})")
    print(f"  -> Training output logged to: {training_log_path}")

    print("\n" + "="*60)
    print("Both FL Server (frontend/dashboard) and Model Training are active in the background!")
    print(f"  • Frontend Dashboard: http://localhost:8000")
    print(f"  • View Server Logs:   tail -f logs/server.log  (or Get-Content logs/server.log -Wait)")
    print(f"  • View Training Logs: tail -f logs/training.log  (or Get-Content logs/training.log -Wait)")
    print("You can continue using this terminal for additional commands.")
    print("="*60)

if __name__ == "__main__":
    main()
