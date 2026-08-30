#!/usr/bin/env python3
"""Reproducible quaternion/ODF/diffraction demonstration for four textures."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# Permit direct ``python examples/texture_workflow.py`` execution from a clone.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classes_and_functions.texture import estimate_odf, odf_from_dict
from src.classes_and_functions.texture_visualization import plot_odf_section, plot_pole_figure
from src.driver_codes.Simulation_Gaussian_Broadening import diffraction_pattern


CONFIGURATIONS = {
    "random": {"type": "random"},
    "single": {"type": "sharp", "center_quaternion": [1, 0, 0, 0], "spread_deg": 3},
    "fiber": {"type": "fiber", "crystal_direction": [1, 1, 1], "sample_direction": [0, 0, 1], "spread_deg": 8},
    "two_component": {"type": "mixture", "components": [
        {"weight": .6, "type": "component", "center_quaternion": [1, 0, 0, 0], "spread_deg": 5},
        {"weight": .4, "type": "component", "center_quaternion": [0.70710678, 0, 0.70710678, 0], "spread_deg": 8},
    ]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grains", type=int, default=100, help="Grains per texture (increase for convergence).")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("texture_demo_output"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "odf_configurations.json").write_text(json.dumps(CONFIGURATIONS, indent=2))
    for index, (name, config) in enumerate(CONFIGURATIONS.items()):
        odf = odf_from_dict(config)
        result = diffraction_pattern("Cu", "FCC", args.grains, 0.361, odf=odf, seed=args.seed + index)
        empirical = estimate_odf(result["orientations"], spread_deg=8)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_odf_section(odf, ax=axes[0]); axes[0].set_title(f"{name}: target")
        plot_odf_section(empirical, ax=axes[1]); axes[1].set_title(f"{name}: empirical")
        fig.savefig(args.output / f"{name}_odf_sections.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(); plot_pole_figure(result["orientations"], (1, 1, 1), ax=ax)
        fig.savefig(args.output / f"{name}_pole_figure.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        plt.imsave(args.output / f"{name}_diffraction.png", np.log1p(result["image"]), cmap="hot")


if __name__ == "__main__":
    main()
