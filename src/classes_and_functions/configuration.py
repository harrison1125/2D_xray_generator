"""Configuration loading and validation for reproducible simulation runs."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "experiment": {
        "material": "Cu", "crystal_structure": "FCC", "lattice_parameter": 0.361,
        "unit_cell": None, "max_hkl_index": 4, "length_unit": "nm",
        "wavelength": 0.0514, "num_grains": 100, "seed": 12345,
    },
    "grain": {
        "size_mean": 33500.0, "size_std": 5750.0,
        "size_unit": "nm", "size_distribution": "lognormal",
        "strain_mean": 0.0, "strain_std": 0.0, "aspect_ratio": 1.5,
    },
    "texture": {"type": "random"},
    "detector": {
        "width_px": 1000, "height_px": 1000, "distance_mm": 86.0,
        "pixel_size_mm": 0.075, "ewald_tolerance": 0.1,
        "bin_to_width_px": None, "bin_to_height_px": None,
        "central_beam_blocker_fwhm_px": 5.0,
    },
    "scattering": {
        "shape_factor": 0.9, "instrumental_fwhm_px": 4.709640090061899,
        "intensity_scale": 1.0, "profile": "voigt", "lorentz_model": "none",
    },
    "output": {
        "directory": "output", "prefix": "simulation",
        "store_grains": True, "store_peaks": True,
        "image_format": "auto",
    },
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
    scattering = config["scattering"]
    structure = experiment["crystal_structure"].upper()
    if structure not in {"SC", "FCC", "BCC", "HCP", "MONOCLINIC", "TRICLINIC"}:
        raise ValueError("experiment.crystal_structure must be SC, FCC, BCC, HCP, MONOCLINIC, or TRICLINIC.")
    if experiment["num_grains"] < 1 or experiment["wavelength"] <= 0 or experiment["lattice_parameter"] <= 0:
        raise ValueError("num_grains, wavelength, and lattice_parameter must be positive.")
    if experiment["length_unit"] != "nm":
        raise ValueError("experiment.length_unit currently supports only 'nm'.")
    if not isinstance(experiment["max_hkl_index"], int) or experiment["max_hkl_index"] < 1:
        raise ValueError("experiment.max_hkl_index must be a positive integer.")
    unit_cell = experiment["unit_cell"]
    if structure in {"MONOCLINIC", "TRICLINIC"}:
        required = {"a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg"}
        if not isinstance(unit_cell, dict) or set(unit_cell) != required:
            raise ValueError("MONOCLINIC/TRICLINIC experiments require unit_cell with a, b, c, alpha_deg, beta_deg, gamma_deg.")
        if any(float(unit_cell[key]) <= 0 for key in ("a", "b", "c")):
            raise ValueError("unit_cell lengths must be positive.")
        if any(not 0 < float(unit_cell[key]) < 180 for key in ("alpha_deg", "beta_deg", "gamma_deg")):
            raise ValueError("unit_cell angles must lie strictly between 0 and 180 degrees.")
    if grain["size_mean"] <= 0 or grain["size_std"] < 0 or grain["strain_std"] < 0 or grain["aspect_ratio"] <= 0:
        raise ValueError("Grain size/aspect ratio must be positive and standard deviations nonnegative.")
    if grain["size_unit"] not in {"nm", "um"}:
        raise ValueError("grain.size_unit must be 'nm' or 'um'.")
    if grain["size_distribution"] not in {"lognormal", "normal"}:
        raise ValueError("grain.size_distribution must be 'lognormal' or 'normal'.")
    if min(detector["width_px"], detector["height_px"], detector["distance_mm"], detector["pixel_size_mm"]) <= 0:
        raise ValueError("Detector dimensions, distance, and pixel size must be positive.")
    if detector["ewald_tolerance"] <= 0:
        raise ValueError("detector.ewald_tolerance must be a positive candidate-window width.")
    bin_width, bin_height = detector["bin_to_width_px"], detector["bin_to_height_px"]
    if (bin_width is None) != (bin_height is None):
        raise ValueError("bin_to_width_px and bin_to_height_px must both be set or both be null.")
    if bin_width is not None:
        if (not isinstance(bin_width, int) or not isinstance(bin_height, int)
                or min(bin_width, bin_height) < 1):
            raise ValueError("Binned detector dimensions must be positive integers.")
        if bin_width > detector["width_px"] or bin_height > detector["height_px"]:
            raise ValueError("Binned detector dimensions cannot exceed native dimensions.")
    if scattering["shape_factor"] <= 0 or scattering["instrumental_fwhm_px"] <= 0 or scattering["intensity_scale"] < 0:
        raise ValueError("Shape/instrument widths must be positive and intensity_scale nonnegative.")
    if scattering["profile"] not in {"voigt", "gaussian"}:
        raise ValueError("scattering.profile must be 'voigt' or 'gaussian'.")
    if scattering["lorentz_model"] not in {"none", "legacy"}:
        raise ValueError("scattering.lorentz_model must be 'none' or 'legacy'.")
    if config["output"]["image_format"] not in {"auto", "npy", "tiff"}:
        raise ValueError("output.image_format must be 'auto', 'npy', or 'tiff'.")
    return config


def load_config(path):
    """Load a JSON experiment configuration (kept dependency-free intentionally)."""
    path = Path(path)
    with path.open() as handle:
        return validate_config(json.load(handle))
