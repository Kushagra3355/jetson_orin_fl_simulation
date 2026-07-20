"""Check the software environment without changing the Jetson board."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    model_path = Path("/proc/device-tree/model")
    jetson_model = model_path.read_text(errors="ignore").strip("\x00\n") if model_path.exists() else "Not running on Jetson"
    report = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "operating_system": platform.platform(),
        "numpy": np.__version__,
        "jetson_model": jetson_model,
        "status": "ready" if sys.version_info >= (3, 10) else "Python 3.10 or newer recommended",
    }
    print(json.dumps(report, indent=2))
    print("Safety: this package uses synthetic data and must not control medical equipment.")


if __name__ == "__main__":
    main()
