"""Run only the privacy costly centralized baseline for teaching contrast."""

from __future__ import annotations

import argparse
import json

from comparison.run_all_in_one import centralized, make_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-client", type=int, default=480)
    args = parser.parse_args()
    result = centralized(make_datasets(args.samples_per_client))
    print(json.dumps(result, indent=2))
    print("\nPrivacy observation: the coordinator receives every synthetic patient training row.")


if __name__ == "__main__":
    main()
