"""Background launcher for FL Server.

Launches the server in the background, redirects output to log files,
opens the frontend dashboard, and returns control to the terminal immediately.
Training can be initiated via the "Start Training" button in the frontend dashboard.
"""

import os
import sys
import subprocess
import time
import webbrowser
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

    print("\nStep 3: Opening Frontend Dashboard in web browser...")
    webbrowser.open("http://localhost:8000")

    print("\n" + "="*60)
    print("FL Server (frontend/dashboard) is active in the background!")
    print(f"  • Frontend Dashboard: http://localhost:8000")
    print(f"  • View Server Logs:   tail -f logs/server.log  (or Get-Content logs/server.log -Wait)")
    print(f"  • To start training: Click the 'Start Training' button in the dashboard!")
    print("You can continue using this terminal for additional commands.")
    print("="*60)

if __name__ == "__main__":
    main()
