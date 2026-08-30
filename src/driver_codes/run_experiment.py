#!/usr/bin/env python3
"""Run a simulation from one JSON configuration file."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.classes_and_functions.configuration import load_config
from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--no-save", action="store_true", help="Run without writing files")
    args = parser.parse_args()
    result = run_experiment(load_config(args.config), save=not args.no_save)
    print(f"Rendered {len(result['peaks'])} reflections from {len(result['grains'])} grains.")
    for kind, path in result.get("files", {}).items(): print(f"{kind}: {path}")


if __name__ == "__main__": main()
