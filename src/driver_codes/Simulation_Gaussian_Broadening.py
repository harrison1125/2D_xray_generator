"""Configuration-driven single-shot Voigt powder/texture simulation."""
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
SIZE_UNIT_TO_NM = {"nm": 1.0, "um": 1000.0}


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


def _area_bin_axis(values, output_size, axis):
    """Integrate uniform input pixels into arbitrary equal-width output bins."""
    moved = np.moveaxis(np.asarray(values, dtype=float), axis, -1)
    input_size = moved.shape[-1]
    if output_size == input_size:
        return np.asarray(values, dtype=float).copy()
    edges = np.linspace(0.0, float(input_size), output_size + 1)
    indices = np.floor(edges).astype(int)
    fractions = edges - indices
    cumulative = np.concatenate(
        (np.zeros(moved.shape[:-1] + (1,), dtype=float),
         np.cumsum(moved, axis=-1)),
        axis=-1,
    )
    integrated = np.take(cumulative, np.minimum(indices, input_size), axis=-1)
    interior = indices < input_size
    integrated[..., interior] += (
        np.take(moved, indices[interior], axis=-1) * fractions[interior]
    )
    return np.moveaxis(np.diff(integrated, axis=-1), -1, axis)


def _area_bin_image(image, output_height, output_width):
    """Flux-conserving area binning, including non-integer bin ratios."""
    binned_rows = _area_bin_axis(image, int(output_height), axis=0)
    return _area_bin_axis(binned_rows, int(output_width), axis=1)


def _detector_readout_metadata(detector, scattering, image_shape):
    native_height, native_width = detector["height_px"], detector["width_px"]
    output_height, output_width = image_shape
    return {
        "native_shape_px": [native_height, native_width],
        "output_shape_px": [output_height, output_width],
        "native_pixel_size_mm": detector["pixel_size_mm"],
        "effective_pixel_size_mm": [
            detector["pixel_size_mm"] * native_height / output_height,
            detector["pixel_size_mm"] * native_width / output_width,
        ],
        "binning_ratio": [native_height / output_height, native_width / output_width],
        "native_instrumental_fwhm_px": scattering["instrumental_fwhm_px"],
        "effective_instrumental_fwhm_px": [
            scattering["instrumental_fwhm_px"] * output_height / native_height,
            scattering["instrumental_fwhm_px"] * output_width / native_width,
        ],
        "method": "flux_conserving_area",
    }


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
    image_format = output.get("image_format", "auto")
    if image_format == "npy" or (image_format == "auto" and tifffile is None):
        paths["image"] = directory / f"{prefix}_pattern.npy"; np.save(paths["image"], result["image"].astype(np.float32))
    else:
        if tifffile is None:
            raise RuntimeError("TIFF output requested but tifffile is not installed.")
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
    size_to_nm = SIZE_UNIT_TO_NM[grain_cfg["size_unit"]]
    size_mean_nm = grain_cfg["size_mean"] * size_to_nm
    size_std_nm = grain_cfg["size_std"] * size_to_nm
    # Any explicitly supplied cell uses the general reciprocal metric.  This
    # is essential for HCP, where c/a and gamma=120 degrees cannot be
    # represented by GrainCubic's single lattice parameter.
    if exp_cfg["unit_cell"] is not None:
        cell = exp_cfg["unit_cell"]
        grain = GrainGeneral(size_mean_nm, size_std_nm**2,
                             grain_cfg["strain_mean"], grain_cfg["strain_std"]**2,
                             grain_cfg["aspect_ratio"], cell["a"], cell["b"], cell["c"],
                             cell["alpha_deg"], cell["beta_deg"], cell["gamma_deg"], experiment,
                             max_hkl_index=exp_cfg["max_hkl_index"],
                             size_distribution=grain_cfg["size_distribution"],
                             length_to_microns=1e-3)
    else:
        grain = GrainCubic(size_mean_nm, size_std_nm**2,
                           grain_cfg["strain_mean"], grain_cfg["strain_std"]**2,
                           grain_cfg["aspect_ratio"], exp_cfg["lattice_parameter"], experiment,
                           max_hkl_index=exp_cfg["max_hkl_index"],
                           size_distribution=grain_cfg["size_distribution"],
                           length_to_microns=1e-3)
    width, height = det_cfg["width_px"], det_cfg["height_px"]
    distance = det_cfg["distance_mm"] / det_cfg["pixel_size_mm"]
    image = np.zeros((height, width), dtype=float); grains = []; peaks = []
    store_grains = bool(config["output"]["store_grains"])
    store_peaks = bool(config["output"]["store_peaks"])
    peak_count = 0
    orientations = odf.sample(exp_cfg["num_grains"], rng)
    for orientation in orientations:
        grain.set_orientation(orientation); grain.realize_properties(rng)
        if store_grains:
            grains.append({"orientation": orientation.copy(),
                           "size": grain.grain_size / size_to_nm,
                           "size_unit": grain_cfg["size_unit"],
                           "size_nm": grain.grain_size,
                           "strain": grain.grain_strain,
                           "volume_um3": grain.volume_um3})
        detector = Detector(EwaldSphere(grain, experiment, det_cfg["ewald_tolerance"]), experiment,
                            width, height, distance, STRUCTURE_FACTORS[structure],
                            shape_factor=scattering["shape_factor"],
                            instrumental_fwhm_px=scattering["instrumental_fwhm_px"],
                            intensity_scale=scattering["intensity_scale"],
                            pixel_size_grain_units=det_cfg["pixel_size_mm"] * 1e6,
                            profile=scattering["profile"],
                            lorentz_model=scattering["lorentz_model"],
                            incident_convergence_full_angle_mrad=exp_cfg[
                                "incident_convergence_full_angle_mrad"
                            ])
        projection = detector.project_points(image=image, store_peaks=store_peaks)
        peak_count += projection["num_peaks"]
        if store_peaks:
            peaks.extend(projection["peaks"])
    # Persist the actual ODF parameters even when a Python ODF object was
    # supplied directly; class names alone are insufficient ground truth for
    # an inference dataset.
    texture_metadata = odf.to_dict() if hasattr(odf, "to_dict") else config["texture"]
    image = _block_direct_beam(image, det_cfg["central_beam_blocker_fwhm_px"])
    if det_cfg["bin_to_width_px"] is not None:
        image = _area_bin_image(image, det_cfg["bin_to_height_px"],
                                det_cfg["bin_to_width_px"])
    result = {"image": image,
              "orientations": orientations, "grains": grains, "peaks": peaks,
              "num_grains": int(exp_cfg["num_grains"]), "num_peaks": peak_count,
              "metadata": {"configuration": config, "texture": texture_metadata,
                           "odf_class": type(odf).__name__,
                           "grain_lattice_model": type(grain).__name__,
                           "detector_readout": _detector_readout_metadata(
                               det_cfg, scattering, image.shape)}}
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
