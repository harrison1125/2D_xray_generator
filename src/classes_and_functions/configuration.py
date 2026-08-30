"""Configuration loading and validation for reproducible simulation runs."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "experiment": {
        "material": "Cu", "crystal_structure": "FCC", "lattice_parameter": 0.361,
        "wavelength": 0.0514, "num_grains": 100, "seed": 12345,
    },
    "grain": {
        "size_mean": 33500.0, "size_std": 5750.0,
        "strain_mean": 0.0, "strain_std": 0.0, "aspect_ratio": 1.5,
    },
    "texture": {"type": "random"},
    "detector": {
        "width_px": 1000, "height_px": 1000, "distance_mm": 86.0,
        "pixel_size_mm": 0.075, "ewald_tolerance": 0.1,
        "central_beam_blocker_fwhm_px": 5.0,
    },
    "scattering": {
        "shape_factor": 0.9, "instrumental_fwhm_px": 4.709640090061899,
        "intensity_scale": 1.0,
    },
    "output": {"directory": "output", "prefix": "simulation"},
}


def _merge(defaults, supplied):
    result = deepcopy(defaults)
    for section, values in supplied.items():
        if section not in result:
            raise ValueError(f"Unknown configuration section {section!r}.")
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section {section!r} must be an object.")
        unknown = set(values) - set(result[section])
        if unknown and section != "texture":
            raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
        result[section].update(values)
    return result


def validate_config(config):
    """Return complete validated config; input values use documented project units."""
    config = _merge(DEFAULT_CONFIG, config)
    experiment, grain, detector = config["experiment"], config["grain"], config["detector"]
    if experiment["crystal_structure"].upper() not in {"SC", "FCC", "BCC", "HCP"}:
        raise ValueError("experiment.crystal_structure must be SC, FCC, BCC, or HCP.")
    if experiment["num_grains"] < 1 or experiment["wavelength"] <= 0 or experiment["lattice_parameter"] <= 0:
        raise ValueError("num_grains, wavelength, and lattice_parameter must be positive.")
    if grain["size_mean"] <= 0 or grain["size_std"] < 0 or grain["strain_std"] < 0 or grain["aspect_ratio"] <= 0:
        raise ValueError("Grain size/aspect ratio must be positive and standard deviations nonnegative.")
    if min(detector["width_px"], detector["height_px"], detector["distance_mm"], detector["pixel_size_mm"]) <= 0:
        raise ValueError("Detector dimensions, distance, and pixel size must be positive.")
    return config


def load_config(path):
    """Load a JSON experiment configuration (kept dependency-free intentionally)."""
    path = Path(path)
    with path.open() as handle:
        return validate_config(json.load(handle))
