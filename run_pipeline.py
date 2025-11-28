from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Ordered list of scripts that make up the full data fetch + render flow.
SCRIPT_ORDER = [
    ("Fetch availability", "get_availability.py"),
    ("Fetch one-way prices", "get_prices_oneway.py"),
    ("Fetch round-trip prices", "get_prices_return.py"),
    ("Fetch seatmaps", "get_seatmaps.py"),
    ("Render reports", "flight_search.py"),
]


def run_script(label: str, script_name: str) -> None:
    script_path = ROOT / script_name
    print(f"\n=== {label} ({script_path.name}) ===")
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{script_path.name} failed with exit code {result.returncode}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Amadeus seatmap pipeline (fetch + render)."
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip the final rendering step (flight_search.py) after fetching data.",
    )
    args = parser.parse_args()

    for label, script_name in SCRIPT_ORDER:
        if args.skip_render and script_name == "flight_search.py":
            print("Skipping render step (flight_search.py).")
            break
        run_script(label, script_name)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
