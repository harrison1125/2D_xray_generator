#!/usr/bin/env python3
"""Generate representative sputter-deposition texture patterns.

The examples model a preferred growth direction that is tilted from the film
normal and a restricted/non-uniform spin about that direction.  They are
intended as dataset seeds, not as a claim that every sputtered film has this
exact ODF.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classes_and_functions.texture import odf_from_dict
from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment


def configurations():
    # [001] growth tilted 12 degrees toward +x: an offset fiber.
    tilted = [float(np.sin(np.deg2rad(12))), 0.0, float(np.cos(np.deg2rad(12)))]
    return {
        "offset_full_ring": {"type": "partial_fiber", "crystal_direction": [1, 1, 1],
                             "sample_direction": tilted, "spread_deg": 7,
                             "spin_width_deg": 360},
        "offset_half_ring": {"type": "partial_fiber", "crystal_direction": [1, 1, 1],
                             "sample_direction": tilted, "spread_deg": 7,
                             "spin_center_deg": 35, "spin_width_deg": 180},
        "lopsided_ring": {"type": "partial_fiber", "crystal_direction": [1, 1, 1],
                          "sample_direction": tilted, "spread_deg": 7,
                          "spin_center_deg": 35, "spin_kappa": 2.5},
        "two_growth_directions": {"type": "mixture", "components": [
            {"weight": 0.7, "type": "partial_fiber", "crystal_direction": [1, 1, 1],
             "sample_direction": tilted, "spread_deg": 7, "spin_center_deg": 35,
             "spin_width_deg": 180},
            {"weight": 0.3, "type": "partial_fiber", "crystal_direction": [1, 1, 1],
             "sample_direction": [-tilted[0], tilted[1], tilted[2]], "spread_deg": 7,
             "spin_center_deg": 210, "spin_width_deg": 150},
        ]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grains", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, default=Path("sputter_texture_output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    (args.output / "odf_configurations.json").write_text(json.dumps(configs, indent=2))
    for index, (name, texture) in enumerate(configs.items()):
        config = {
            "experiment": {"material": "Cu", "crystal_structure": "FCC",
                           "lattice_parameter": 0.361, "wavelength": 0.0514,
                           "num_grains": args.grains, "seed": args.seed + index},
            "texture": texture,
            "output": {"directory": str(args.output), "prefix": name},
        }
        result = run_experiment(config, save=True, odf=odf_from_dict(texture))
        print(f"{name}: {len(result['peaks'])} peaks; image={result['files']['image']}")


if __name__ == "__main__":
    main()
