"""Configuration-driven Gaussian-broadened powder/texture simulation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.classes_and_functions import StructureFactors
from src.classes_and_functions.configuration import validate_config
from src.classes_and_functions.crystal import GrainCubic, GrainGeneral
from src.classes_and_functions.detector_module import Detector
from src.classes_and_functions.ewald import EwaldSphere
from src.classes_and_functions.experiment import Experiment
from src.classes_and_functions.texture import odf_from_dict

try:
    import tifffile
except ImportError:
    tifffile = None

STRUCTURE_FACTORS = {"SC": StructureFactors.structure_factor_sc, "FCC": StructureFactors.structure_factor_fcc,
                     "BCC": StructureFactors.structure_factor_bcc, "HCP": StructureFactors.structure_factor_hcp,
                     "MONOCLINIC": StructureFactors.structure_factor_primitive,
                     "TRICLINIC": StructureFactors.structure_factor_primitive}


def _json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _block_direct_beam(image, fwhm_px):
    if fwhm_px <= 0: return image
    row, col = np.indices(image.shape)
    center = np.array(image.shape) / 2
    result = image.copy()
    result[np.hypot(row - center[0], col - center[1]) <= fwhm_px / 2] = 0
    return result


def _save_result(result, config):
    output = config["output"]; directory = Path(output["directory"]); directory.mkdir(parents=True, exist_ok=True)
    prefix = output["prefix"]
    paths = {"metadata": directory / f"{prefix}_metadata.json"}
    paths["metadata"].write_text(json.dumps(result["metadata"], indent=2, default=_json_value))
    if output["store_grains"]:
        paths["grains"] = directory / f"{prefix}_grains.json"
        paths["grains"].write_text(json.dumps(result["grains"], indent=2, default=_json_value))
    if output["store_peaks"]:
        paths["peaks"] = directory / f"{prefix}_peaks.json"
        paths["peaks"].write_text(json.dumps(result["peaks"], indent=2, default=_json_value))
    if tifffile is None:
        paths["image"] = directory / f"{prefix}_pattern.npy"; np.save(paths["image"], result["image"].astype(np.float32))
    else:
        paths["image"] = directory / f"{prefix}_pattern.tiff"
        tifffile.imwrite(paths["image"], result["image"].astype(np.float32), metadata={"config": result["metadata"]})
    return paths


def run_experiment(config, *, save=True, odf=None):
    """Run one simulation from a mapping; config plus seed is reproducible."""
    config = validate_config(config)
    exp_cfg, grain_cfg, det_cfg, scattering = (config[key] for key in ("experiment", "grain", "detector", "scattering"))
    rng = np.random.default_rng(exp_cfg["seed"])
    odf = odf_from_dict(config["texture"]) if odf is None else odf
    structure = exp_cfg["crystal_structure"].upper()
    experiment = Experiment(exp_cfg["wavelength"], exp_cfg["material"])
    if structure in {"MONOCLINIC", "TRICLINIC"}:
        cell = exp_cfg["unit_cell"]
        grain = GrainGeneral(grain_cfg["size_mean"], grain_cfg["size_std"]**2,
                             grain_cfg["strain_mean"], grain_cfg["strain_std"]**2,
                             grain_cfg["aspect_ratio"], cell["a"], cell["b"], cell["c"],
                             cell["alpha_deg"], cell["beta_deg"], cell["gamma_deg"], experiment,
                             max_hkl_index=exp_cfg["max_hkl_index"])
    else:
        grain = GrainCubic(grain_cfg["size_mean"], grain_cfg["size_std"]**2,
                           grain_cfg["strain_mean"], grain_cfg["strain_std"]**2,
                           grain_cfg["aspect_ratio"], exp_cfg["lattice_parameter"], experiment,
                           max_hkl_index=exp_cfg["max_hkl_index"])
    width, height = det_cfg["width_px"], det_cfg["height_px"]
    distance = det_cfg["distance_mm"] / det_cfg["pixel_size_mm"]
    image = np.zeros((height, width), dtype=float); grains = []; peaks = []
    orientations = odf.sample(exp_cfg["num_grains"], rng)
    for orientation in orientations:
        grain.set_orientation(orientation); grain.realize_properties(rng)
        grains.append({"orientation": orientation.copy(), "size": grain.grain_size,
                       "strain": grain.grain_strain, "volume": grain.volume})
        detector = Detector(EwaldSphere(grain, experiment, det_cfg["ewald_tolerance"]), experiment,
                            width, height, distance, STRUCTURE_FACTORS[structure],
                            shape_factor=scattering["shape_factor"],
                            instrumental_fwhm_px=scattering["instrumental_fwhm_px"],
                            intensity_scale=scattering["intensity_scale"])
        projection = detector.project_points(); image += projection["image"]; peaks.extend(projection["peaks"])
    # Persist the actual ODF parameters even when a Python ODF object was
    # supplied directly; class names alone are insufficient ground truth for
    # an inference dataset.
    texture_metadata = odf.to_dict() if hasattr(odf, "to_dict") else config["texture"]
    result = {"image": _block_direct_beam(image, det_cfg["central_beam_blocker_fwhm_px"]),
              "orientations": orientations, "grains": grains, "peaks": peaks,
              "metadata": {"configuration": config, "texture": texture_metadata,
                           "odf_class": type(odf).__name__}}
    if save: result["files"] = _save_result(result, config)
    return result


def diffraction_pattern(material_type, CrystalStructure, num_grains, lattice_parameter,
                        odf=None, seed=None, shape_factor=.9, instrumental_fwhm_px=4.709640090061899,
                        intensity_scale=1.0):
    """Backward-compatible wrapper; use ``run_experiment`` for new code."""
    config = {"experiment": {"material": material_type, "crystal_structure": CrystalStructure,
                               "num_grains": num_grains, "lattice_parameter": lattice_parameter, "seed": seed},
              "scattering": {"shape_factor": shape_factor, "instrumental_fwhm_px": instrumental_fwhm_px,
                             "intensity_scale": intensity_scale}}
    if isinstance(odf, dict):
        config["texture"] = odf
        return run_experiment(config, save=True)
    return run_experiment(config, odf=odf, save=True) if odf is not None else run_experiment(config, save=True)
