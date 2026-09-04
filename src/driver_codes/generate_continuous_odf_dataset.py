#!/usr/bin/env python3
"""Generate a continuous physical-manifold ODF corpus with XRD observations.

Unlike the curated texture-validation driver, this program samples a
fixed-width continuous mixture space on SO(3).  A scrambled Sobol source can
feed either the retained legacy hypercube or material-component/fiber
manifolds.  Each ODF is an explicit ground-truth entity and can be rendered at
several grain counts without changing its label.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
from PIL import Image, PngImagePlugin
try:
    import tifffile
except ImportError:
    tifffile = None
try:
    import h5py
except ImportError:
    h5py = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classes_and_functions.configuration import validate_config
from src.classes_and_functions.continuous_odf import (
    PhysicalTextureSpace, SobolODFSpace, harmonic_coefficients,
    pack_harmonic_coefficients, sobol_odf_grid, sobol_unit_cube,
)
from src.classes_and_functions.directional_targets import write_directional_targets
from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment


ODF_LABEL_VERSION = "continuous_symmetric_dvp_mixture_v1"


# These arrays are written beside the immutable ODF descriptors.  Keeping them
# in the label table lets a resumed corpus reuse the already-validated physical
# design instead of repeating the expensive analytic texture-index calculation
# merely to recreate auxiliary provenance fields.
_DESIGN_METADATA_FIELDS = (
    "texture_index",
    "texture_index_log_weight",
    "sampling_space",
    "inference_mode",
    "process_family",
    "sample_process_domain",
    "component_manifold",
    "component_manifold_kind",
    "component_process_tag",
    "active_component_count",
    "component_active_mask",
    "dirichlet_concentration",
    "design_candidate_index",
)


def _descriptor_fields(component_count: int) -> dict[str, list[int]]:
    """Return the exact packed-mixture layout for a fixed slot count."""
    if component_count < 1:
        raise ValueError("component_count must be positive.")
    weights_start = 1
    centers_start = weights_start + component_count
    widths_start = centers_start + 4 * component_count
    return {
        "background_weight": [0, 1],
        "component_weights": [weights_start, centers_start],
        "component_centers_quaternion_wxyz": [centers_start, widths_start],
        "component_fwhm_deg": [widths_start, widths_start + component_count],
    }


def _descriptor_length(component_count: int) -> int:
    return 1 + 6 * component_count


def _label_version(component_count: int) -> str:
    # Preserve v1 and its 31-value target exactly for established corpora.
    return ODF_LABEL_VERSION if component_count == 5 else "continuous_symmetric_dvp_mixture_v2"


def _merge(base: dict, update: dict) -> dict:
    """Recursively overlay a system-specific simulation mapping."""
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_") or "system"


@contextmanager
def _progress_heartbeat(label: str, interval_seconds: float):
    """Print periodic liveness messages while a simulation is running."""
    started = time.monotonic()
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(interval_seconds):
            elapsed = time.monotonic() - started
            print(f"[progress] still running {label} (elapsed {elapsed:.1f}s)", flush=True)

    thread = threading.Thread(target=report, name="dataset-progress", daemon=True)
    thread.start()
    try:
        yield started
    finally:
        stop.set()
        thread.join()


def _completed_ids(manifest: Path) -> set[str]:
    completed: set[str] = set()
    if not manifest.exists():
        return completed
    for line in manifest.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "complete":
            completed.add(record["run_id"])
    return completed


def _png_preview_root(output_root: Path, run: dict) -> Path:
    relative = Path(run.get("png_preview_directory", "png_preview"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("png_preview_directory must be a subdirectory of the dataset root.")
    return output_root / relative


def _png_preview_pixels(image: np.ndarray, saturation_percentile: float = 99.5):
    """Convert a scientific detector image to a white-on-black log preview."""
    if not 0 < saturation_percentile <= 100:
        raise ValueError("png_saturation_percentile must lie in (0, 100].")
    finite = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    nonnegative = np.clip(finite, 0.0, None)
    maximum = float(nonnegative.max()) if nonnegative.size else 0.0
    if maximum <= 0:
        return np.zeros(nonnegative.shape, dtype=np.uint8), maximum, 0.0
    transformed = np.log1p(nonnegative)
    positive = transformed[transformed > 0]
    ceiling = float(np.percentile(positive, saturation_percentile))
    display = transformed / max(ceiling, np.finfo(float).eps)
    return (np.rint(255.0 * np.clip(display, 0.0, 1.0)).astype(np.uint8),
            maximum, ceiling)


def _write_png_preview(image: np.ndarray, destination: Path, run_id: str,
                       saturation_percentile: float = 99.5) -> None:
    pixels, maximum, ceiling = _png_preview_pixels(image, saturation_percentile)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("run_id", run_id)
    metadata.add_text("display_transform", f"uint8(log1p(max(image,0)); positive-pixel p{saturation_percentile:g}=255)")
    metadata.add_text("source_shape", "x".join(map(str, np.asarray(image).shape)))
    metadata.add_text("source_sum", f"{float(np.nansum(image)):.17g}")
    metadata.add_text("source_max", f"{maximum:.17g}")
    metadata.add_text("log_display_ceiling", f"{ceiling:.17g}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    Image.fromarray(pixels).save(temporary, format="PNG", pnginfo=metadata)
    temporary.replace(destination)


def _write_tiff(image: np.ndarray, destination: Path, metadata: dict) -> None:
    if tifffile is None:
        raise RuntimeError("TIFF output requested but tifffile is not installed.")
    tifffile.imwrite(destination, np.asarray(image, dtype=np.float32),
                     metadata={"config": metadata})


def _load_stored_pattern(run_dir: Path) -> np.ndarray:
    npy_paths = sorted(run_dir.glob("*_pattern.npy"))
    if npy_paths:
        return np.load(npy_paths[0])
    tiff_paths = sorted(run_dir.glob("*_pattern.tif")) + sorted(run_dir.glob("*_pattern.tiff"))
    if tiff_paths:
        with Image.open(tiff_paths[0]) as image:
            return np.asarray(image)
    raise FileNotFoundError(f"No stored detector pattern found in {run_dir}.")


def _sync_png_previews(output_root: Path, run: dict, manifest: Path,
                       *, overwrite: bool = False) -> int:
    """Backfill the flat PNG mirror for completed continuous observations."""
    if not run.get("write_png_previews", True) or not manifest.exists():
        return 0
    preview_root = _png_preview_root(output_root, run)
    saturation = float(run.get("png_saturation_percentile", 99.5))
    created = 0
    for line in manifest.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") != "complete":
            continue
        destination = preview_root / f"{record['run_id']}.png"
        if destination.exists() and not overwrite:
            continue
        _write_png_preview(_load_stored_pattern(Path(record["path"])), destination,
                           record["run_id"], saturation)
        created += 1
    return created


def _representation_mode(run: dict) -> tuple[bool, bool]:
    mode = run.get("output_format", "harmonic_coefficients_and_odf_grid")
    valid = {"parameters_only", "odf_grid", "harmonic_coefficients", "harmonic_coefficients_and_odf_grid"}
    if mode not in valid:
        raise ValueError(f"Unsupported output_format {mode!r}; expected one of {sorted(valid)}.")
    return "odf_grid" in mode, "harmonic" in mode


def _directional_target_settings(run: dict, system: dict) -> dict | None:
    settings = run.get("directional_targets")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("run.directional_targets must be an object.")
    if not settings.get("enabled", False):
        return None
    if settings.get("source", "observation_orientations") != "observation_orientations":
        raise ValueError("directional_targets.source currently supports observation_orientations only.")
    grid_size = int(settings.get("grid_size", 32))
    coefficient_shape = tuple(map(int, settings.get("coefficient_shape", (9, 9))))
    sample_directions = settings.get("sample_directions", [
        {"name": "RD", "direction": [1, 0, 0]},
        {"name": "TD", "direction": [0, 1, 0]},
        {"name": "ND", "direction": [0, 0, 1]},
    ])
    pole_families = system.get("pole_families")
    if grid_size < 4 or len(coefficient_shape) != 2 or min(coefficient_shape) < 1:
        raise ValueError("directional target grid/coefficient dimensions are invalid.")
    if not isinstance(pole_families, list) or not pole_families:
        raise ValueError(f"{system['name']} requires nonempty pole_families.")
    if not isinstance(sample_directions, list) or not sample_directions:
        raise ValueError("directional_targets.sample_directions must be a nonempty list.")
    for family in pole_families:
        if set(family) != {"name", "hkl"} or len(family["hkl"]) != 3:
            raise ValueError("Each pole family must contain only name and three-index hkl.")
    for direction in sample_directions:
        if set(direction) != {"name", "direction"} or len(direction["direction"]) != 3:
            raise ValueError("Each sample direction must contain only name and direction.")
    return {
        "filename": "pf_ipf_targets.npz",
        "grid_size": grid_size,
        "coefficient_shape": coefficient_shape,
        "sample_directions": sample_directions,
        "pole_families": pole_families,
    }


def odf_descriptor(odf) -> np.ndarray:
    """Return the canonical fixed-width, directly ingestible ODF label."""
    return np.concatenate((
        [odf.background_weight],
        odf.component_weights,
        odf.component_centers.reshape(-1),
        odf.component_fwhm_deg,
    )).astype(np.float32)


def _label_arrays(odf, sobol_point: np.ndarray,
                  design_metadata: dict | None = None) -> dict[str, np.ndarray]:
    arrays = {
        "label_version": np.asarray(_label_version(odf.num_components)),
        "descriptor": odf_descriptor(odf),
        "sobol_point": np.asarray(sobol_point, dtype=np.float64),
        "component_centers_quaternion_wxyz": odf.component_centers.astype(np.float64),
        "component_fwhm_deg": odf.component_fwhm_deg.astype(np.float64),
        "component_weights": odf.component_weights.astype(np.float64),
        "background_weight": np.asarray(odf.background_weight, dtype=np.float64),
        "effective_components": np.asarray(odf.effective_components, dtype=np.float64),
        "crystal_symmetry": np.asarray(odf.crystal_symmetry),
        "proper_symmetry_order": np.asarray(len(odf.symmetry_operations), dtype=np.int16),
    }
    if design_metadata:
        arrays.update({key: np.asarray(value) for key, value in design_metadata.items()})
    return arrays


def _write_representation(path: Path, odf, sobol_point: np.ndarray, grid: np.ndarray | None,
                          bandlimit: int | None, *, odf_label_index: int,
                          odf_id: str, grain_count: int, observation_seed: int,
                          design_metadata: dict | None = None) -> None:
    """Write an exact ODF label and observation linkage in one NPZ file."""
    arrays = _label_arrays(odf, sobol_point, design_metadata)
    arrays.update({
        "odf_label_index": np.asarray(odf_label_index, dtype=np.int32),
        "odf_id": np.asarray(odf_id),
        "observation_grain_count": np.asarray(grain_count, dtype=np.int32),
        "observation_seed": np.asarray(observation_seed, dtype=np.uint32),
        "expected_background_grains": np.asarray(grain_count * odf.background_weight, dtype=np.float64),
        "expected_component_grains": (grain_count * odf.component_weights).astype(np.float64),
    })
    if grid is not None:
        arrays["odf_grid_values"] = odf.evaluate(grid).astype(np.float32)
    if bandlimit is not None:
        arrays.update(pack_harmonic_coefficients(harmonic_coefficients(odf, bandlimit)))
        arrays["harmonic_bandlimit"] = np.asarray(bandlimit, dtype=np.int16)
    np.savez_compressed(path, **arrays)


def grain_counts(spec: dict) -> tuple[int, ...]:
    """Return validated observation grain counts, preserving configured order."""
    configured = spec["run"].get("grain_counts")
    if configured is None:
        configured = [spec["base_config"]["experiment"]["num_grains"]]
    counts = tuple(int(value) for value in configured)
    if not counts or any(value < 1 for value in counts) or len(set(counts)) != len(counts):
        raise ValueError("run.grain_counts must contain unique positive integers.")
    return counts


def planned_odf_points(spec: dict) -> int:
    return int(spec["run"]["num_samples_per_symmetry"]) * len(spec["crystal_systems"])


def planned_runs(spec: dict) -> int:
    return planned_odf_points(spec) * len(grain_counts(spec))


def _label_schema(spec: dict) -> dict:
    component_count = int(spec["run"].get("max_texture_components", 5))
    schema = {
        "label_version": _label_version(component_count),
        "descriptor_length": _descriptor_length(component_count),
        "descriptor_dtype": "float32",
        "descriptor_fields_half_open_offsets": _descriptor_fields(component_count),
        "component_count": component_count,
        "component_order": "descending component weight, then ascending FWHM",
        "quaternion_convention": "unit quaternion [w,x,y,z], canonical sign, crystal-symmetry fundamental zone",
        "density_measure": "normalized Haar measure on SO(3)",
        "kernel": "de_la_vallee_poussin",
        "width_definition": "full width at half maximum in degrees",
        "sampling_space": spec["run"].get("sampling_method", "Sobol_sequence"),
        "auxiliary_conditioning_fields": [
            "texture_index", "texture_index_log_weight", "sampling_space",
            "inference_mode", "process_family", "component_manifold",
            "component_manifold_kind", "component_process_tag",
            "sample_process_domain", "active_component_count", "component_active_mask",
            "dirichlet_concentration",
            "design_candidate_index",
        ],
        "authoritative_fields": [
            "background_weight", "component_weights",
            "component_centers_quaternion_wxyz", "component_fwhm_deg",
            "crystal_symmetry",
        ],
        "systems": [
            {"index": index, "name": system["name"],
             "crystal_symmetry": system["orientation_symmetry"]}
            for index, system in enumerate(spec["crystal_systems"])
        ],
        "notes": [
            "The descriptor is exact for the configured fixed-slot analytic mixture; zero-weight slots are inactive and masked explicitly.",
            "Quaternion sign, symmetry reduction, and weight sorting remove common duplicate labels but retain boundary discontinuities.",
            "Observation grain count and seed are conditioning variables, not part of the ODF label.",
            "Texture index and sampler provenance are auxiliary variables; five-slot corpora retain the backward-compatible 31-value v1 descriptor.",
        ],
    }
    directional = spec["run"].get("directional_targets", {})
    if directional.get("enabled", False):
        coefficient_shape = directional.get("coefficient_shape", [9, 9])
        schema["directional_targets"] = {
            "target_version": "realized_pf_ipf_lambert_dct_v1",
            "path_field": "pf_ipf_path",
            "target_source": "exact finite grain orientations used to render each observation",
            "projection": "normalized Lambert equal-area upper hemisphere with antipodal folding",
            "grid_size": int(directional.get("grid_size", 32)),
            "coefficient_basis": "orthonormal 2-D DCT-II, low-frequency rectangular prefix",
            "coefficient_shape": list(map(int, coefficient_shape)),
            "channel_order": "configured PF families followed by configured sample-direction IPFs",
            "systems": [
                {"name": system["name"], "pole_families": system["pole_families"]}
                for system in spec["crystal_systems"]
            ],
            "sample_directions": directional.get("sample_directions", [
                {"name": "RD", "direction": [1, 0, 0]},
                {"name": "TD", "direction": [0, 1, 0]},
                {"name": "ND", "direction": [0, 0, 1]},
            ]),
        }
        schema["notes"].append(
            "PF/IPF targets describe the finite orientation realization for each observation; the analytic mixture descriptor remains the full-ODF target."
        )
    return schema


def _write_json_exact(path: Path, value: dict) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and json.loads(path.read_text()) != json.loads(serialized):
        raise ValueError(f"Existing {path} does not match the requested label schema.")
    if not path.exists():
        path.write_text(serialized)


def _write_system_label_table(path: Path, points: np.ndarray, space,
                              design_metadata: dict[str, np.ndarray]) -> None:
    """Write one nonduplicated exact-label table for all ODFs in a system."""
    if path.exists():
        with np.load(path) as existing:
            if (existing["sobol_point"].shape != points.shape
                    or not np.array_equal(existing["sobol_point"], points)
                    or str(existing["label_version"]) != _label_version(space.max_components)):
                raise ValueError(f"Existing {path} does not match this ODF design.")
        return
    count = len(points)
    component_count = int(space.max_components)
    descriptors = np.empty((count, _descriptor_length(component_count)), dtype=np.float32)
    centers = np.empty((count, component_count, 4), dtype=np.float64)
    widths = np.empty((count, component_count), dtype=np.float64)
    weights = np.empty((count, component_count), dtype=np.float64)
    background = np.empty(count, dtype=np.float64)
    effective = np.empty(count, dtype=np.float64)
    for index, point in enumerate(points):
        odf = space.from_unit_cube(point)
        descriptors[index] = odf_descriptor(odf)
        centers[index] = odf.component_centers
        widths[index] = odf.component_fwhm_deg
        weights[index] = odf.component_weights
        background[index] = odf.background_weight
        effective[index] = odf.effective_components
    arrays = dict(
        label_version=np.asarray(_label_version(component_count)),
        descriptor=descriptors,
        sobol_point=points.astype(np.float64),
        component_centers_quaternion_wxyz=centers,
        component_fwhm_deg=widths,
        component_weights=weights,
        background_weight=background,
        effective_components=effective,
        odf_label_index=np.arange(count, dtype=np.int32),
    )
    arrays.update(design_metadata)
    np.savez_compressed(path, **arrays)


def _load_stored_design(design_path: Path, label_path: Path, *, num_samples: int,
                        dimension: int, component_count: int) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Load a complete persisted physical design for a resumable corpus.

    Both files are required: the raw Sobol coordinates establish the sampling
    design, while the label table contains the expensive analytic diagnostics
    and process provenance derived from those coordinates.  A partial pair is
    deliberately rebuilt rather than silently trusted.
    """
    if not design_path.exists() or not label_path.exists():
        return None
    points = np.load(design_path)
    if points.shape != (num_samples, dimension):
        raise ValueError(
            f"Existing {design_path} has incompatible shape {points.shape}; "
            f"expected {(num_samples, dimension)}."
        )
    expected_version = _label_version(component_count)
    with np.load(label_path, allow_pickle=False) as labels:
        missing = [name for name in _DESIGN_METADATA_FIELDS if name not in labels]
        if missing:
            # Older corpus versions predate the persisted provenance fields.
            # Rebuild their design below to retain backward-compatible resume
            # behavior; current corpus versions take the fast cached path.
            return None
        if (labels["sobol_point"].shape != points.shape
                or not np.array_equal(labels["sobol_point"], points)
                or str(labels["label_version"]) != expected_version):
            raise ValueError(f"Existing {label_path} does not match {design_path}.")
        metadata = {name: labels[name].copy() for name in _DESIGN_METADATA_FIELDS}
    if any(value.shape[0] != num_samples for value in metadata.values()):
        raise ValueError(f"Existing {label_path} has incompatible design metadata shapes.")
    return points, metadata


def _write_scan_index(output_root: Path, systems_data: list[dict], counts: tuple[int, ...],
                      image_format: str) -> None:
    """Write a flat scan/label lookup table suitable for a dataset loader."""
    run_ids, odf_ids, design_pair_ids = [], [], []
    sample_dirs, image_paths, pf_ipf_paths = [], [], []
    system_indices, label_indices, grain_values, observation_indices, descriptors = [], [], [], [], []
    texture_indices, texture_log_weights = [], []
    sampling_spaces, inference_modes, process_families, sample_process_domains = [], [], [], []
    component_process_tags, active_component_counts, component_active_masks = [], [], []
    image_suffix = ".npy" if image_format == "npy" else ".tiff"
    for system_index, item in enumerate(systems_data):
        name = item["name"]
        with np.load(item["label_path"]) as labels:
            label_descriptors = labels["descriptor"]
            # ``NpzFile`` reads its compressed members lazily.  Cache every
            # member used below once: indexing ``labels[name][sample_index]``
            # in this loop re-decompresses the entire member for every ODF and
            # turns a 30,000-row scan index into a many-minute initialization.
            optional = {
                field: labels[field] if field in labels else None
                for field in (
                    "texture_index", "texture_index_log_weight", "sampling_space",
                    "inference_mode", "process_family", "sample_process_domain",
                    "component_process_tag", "active_component_count",
                    "component_active_mask",
                )
            }
            for sample_index in range(len(item["points"])):
                odf_id = f"{name}__odf_{sample_index:06d}"
                for observation_index, count in enumerate(counts):
                    run_id = f"{odf_id}__g{count:06d}"
                    relative_dir = Path(name) / run_id
                    run_ids.append(run_id); odf_ids.append(odf_id)
                    design_pair_ids.append(
                        f"paired_sobol_{sample_index:06d}"
                        if item["paired_design"] else odf_id
                    )
                    sample_dirs.append(str(relative_dir))
                    image_paths.append(str(relative_dir / f"simulation_pattern{image_suffix}"))
                    pf_ipf_paths.append(
                        str(relative_dir / item["directional_targets"]["filename"])
                        if item["directional_targets"] is not None else ""
                    )
                    system_indices.append(system_index); label_indices.append(sample_index)
                    grain_values.append(count); observation_indices.append(observation_index)
                    descriptors.append(label_descriptors[sample_index])
                    texture_indices.append(float(optional["texture_index"][sample_index])
                                           if optional["texture_index"] is not None else np.nan)
                    texture_log_weights.append(float(optional["texture_index_log_weight"][sample_index])
                                               if optional["texture_index_log_weight"] is not None else 0.0)
                    sampling_spaces.append(str(optional["sampling_space"][sample_index])
                                           if optional["sampling_space"] is not None else "legacy_sobol_hypercube")
                    inference_modes.append(str(optional["inference_mode"][sample_index])
                                           if optional["inference_mode"] is not None else "legacy")
                    process_families.append(str(optional["process_family"][sample_index])
                                            if optional["process_family"] is not None else "unconstrained")
                    sample_process_domains.append(str(optional["sample_process_domain"][sample_index])
                                                  if optional["sample_process_domain"] is not None else "unconstrained")
                    component_process_tags.append(
                        optional["component_process_tag"][sample_index]
                        if optional["component_process_tag"] is not None else np.asarray([], dtype=str)
                    )
                    active_component_counts.append(
                        int(optional["active_component_count"][sample_index])
                        if optional["active_component_count"] is not None else -1
                    )
                    component_active_masks.append(
                        optional["component_active_mask"][sample_index]
                        if optional["component_active_mask"] is not None else np.asarray([], dtype=bool)
                    )
    arrays = {
        "run_id": np.asarray(run_ids), "odf_id": np.asarray(odf_ids),
        "design_pair_id": np.asarray(design_pair_ids),
        "sample_directory": np.asarray(sample_dirs), "image_path": np.asarray(image_paths),
        "system_index": np.asarray(system_indices, dtype=np.int16),
        "odf_label_index": np.asarray(label_indices, dtype=np.int32),
        "grain_count": np.asarray(grain_values, dtype=np.int32),
        "observation_index": np.asarray(observation_indices, dtype=np.int16),
        "odf_descriptor": np.asarray(descriptors, dtype=np.float32),
        "texture_index": np.asarray(texture_indices, dtype=np.float64),
        "texture_index_log_weight": np.asarray(texture_log_weights, dtype=np.float64),
        "sampling_space": np.asarray(sampling_spaces),
        "inference_mode": np.asarray(inference_modes),
        "process_family": np.asarray(process_families),
        "sample_process_domain": np.asarray(sample_process_domains),
        "component_process_tag": np.asarray(component_process_tags),
        "active_component_count": np.asarray(active_component_counts, dtype=np.int16),
        "component_active_mask": np.asarray(component_active_masks, dtype=bool),
    }
    if any(pf_ipf_paths):
        arrays["pf_ipf_path"] = np.asarray(pf_ipf_paths)
    path = output_root / "planned_scan_index.npz"
    if path.exists():
        with np.load(path) as existing:
            if (not np.array_equal(existing["run_id"], arrays["run_id"])
                    or not np.array_equal(existing["odf_descriptor"], arrays["odf_descriptor"])):
                raise ValueError(f"Existing {path} does not match this scan plan.")
        return
    np.savez_compressed(path, **arrays)


def _hdf5_dataset(group, name: str, value: np.ndarray) -> None:
    """Write one metadata array with portable UTF-8 handling and compression."""
    array = np.asarray(value)
    if array.dtype.kind in {"U", "O"}:
        data = np.asarray(array, dtype=h5py.string_dtype(encoding="utf-8"))
    else:
        data = array
    keyword = {"compression": "gzip", "shuffle": True} if data.ndim else {}
    group.create_dataset(name, data=data, **keyword)


def _write_hdf5_metadata(output_root: Path, spec: dict, systems_data: list[dict]) -> None:
    """Consolidate exact ODF labels and the planned scan index into HDF5.

    Detector images remain individual NPY/TIFF files to avoid duplicating the
    corpus.  Their paths, each ODF's exact mixture parameters, texture index,
    active-slot mask, and provenance tags are written here in one chunked file.
    The JSONL manifest remains the authoritative live status record.
    """
    if not spec["run"].get("write_hdf5_metadata", False):
        return
    if h5py is None:
        raise RuntimeError(
            "write_hdf5_metadata requires h5py; install the project dependencies first."
        )
    index_path = output_root / "planned_scan_index.npz"
    if not index_path.exists():
        raise FileNotFoundError(f"Cannot create HDF5 metadata without {index_path}.")
    target = output_root / "metadata.h5"
    if target.exists():
        return
    temporary = output_root / "metadata.h5.tmp"
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = "xrd_continuous_odf_metadata_v1"
        handle.attrs["manifest_jsonl"] = str(spec["run"].get("manifest", "manifest.jsonl"))
        handle.attrs["dataset_spec_json"] = json.dumps(spec, sort_keys=True)
        with np.load(index_path, allow_pickle=False) as scan_index:
            scans = handle.create_group("scans")
            for name in scan_index.files:
                _hdf5_dataset(scans, name, scan_index[name])
        odf_group = handle.create_group("odf_labels")
        for item in systems_data:
            system_group = odf_group.create_group(item["name"])
            with np.load(item["label_path"], allow_pickle=False) as labels:
                for name in labels.files:
                    _hdf5_dataset(system_group, name, labels[name])
    temporary.replace(target)


def _make_odf_space(run: dict, system: dict):
    """Construct either the retained legacy chart or the physical manifold."""
    sampling_method = run.get("sampling_method", "Sobol_sequence")
    if sampling_method == "Sobol_sequence":
        return SobolODFSpace(
            crystal_symmetry=system["orientation_symmetry"],
            max_components=int(run.get("max_texture_components", 5)),
            fwhm_range_deg=tuple(run.get("spread_range_degrees", (2.5, 30.0))),
            background_range=tuple(run.get("isotropic_background_range", (0.05, 0.95))),
            dirichlet_concentration_range=tuple(run.get("dirichlet_concentration_range", (0.2, 4.0))),
            spread_scale=run.get("spread_scale", "linear"),
        )
    if sampling_method != "physical_sobol_manifold":
        raise ValueError(
            "sampling_method must be 'Sobol_sequence' or 'physical_sobol_manifold'."
        )

    physical = _merge(run.get("physical_texture", {}), system.get("physical_texture", {}))
    inference_mode = physical.get("inference_mode", "likelihood")
    if inference_mode in {"likelihood", "broad_likelihood"}:
        defaults = {
            "fwhm_strata_degrees": ((8.0, 18.0), (18.0, 32.0), (32.0, 55.0)),
            "fwhm_strata_probabilities": (0.25, 0.45, 0.30),
            "background_range": (0.01, 0.95), "background_beta": (1.4, 1.4),
            "dirichlet_concentration_range": (0.12, 1.2),
            "uniform_center_fraction": 0.25, "component_center_fwhm_deg": 14.0,
            "fiber_concentration": 45.0, "fiber_tube_fwhm_deg": 12.0,
        }
    elif inference_mode == "physical_prior":
        defaults = {
            "fwhm_strata_degrees": ((12.0, 20.0), (20.0, 32.0), (32.0, 45.0)),
            "fwhm_strata_probabilities": (0.30, 0.50, 0.20),
            "background_range": (0.08, 0.78), "background_beta": (2.2, 2.0),
            "dirichlet_concentration_range": (0.08, 0.55),
            "uniform_center_fraction": 0.02, "component_center_fwhm_deg": 10.0,
            "fiber_concentration": 70.0, "fiber_tube_fwhm_deg": 8.0,
        }
    else:
        raise ValueError(
            "physical_texture.inference_mode must be likelihood, broad_likelihood, or physical_prior."
        )
    settings = {**defaults, **physical}
    count_range = settings.get("component_count_range")
    hybrid_probabilities = settings.get("hybrid_process_probabilities", {
        "rolling": 1 / 3,
        "recrystallization": 1 / 3,
        "additive_manufacturing": 1 / 3,
    })
    return PhysicalTextureSpace(
        crystal_symmetry=system["orientation_symmetry"],
        process_family=settings.get("process_family", "broad"),
        inference_mode=inference_mode,
        max_components=int(run.get("max_texture_components", 5)),
        fwhm_strata_deg=tuple(tuple(item) for item in settings["fwhm_strata_degrees"]),
        fwhm_strata_probabilities=tuple(settings["fwhm_strata_probabilities"]),
        background_range=tuple(settings["background_range"]),
        background_beta=tuple(settings["background_beta"]),
        dirichlet_concentration_range=tuple(settings["dirichlet_concentration_range"]),
        uniform_center_fraction=float(settings["uniform_center_fraction"]),
        component_center_fwhm_deg=float(settings["component_center_fwhm_deg"]),
        fiber_concentration=float(settings["fiber_concentration"]),
        fiber_tube_fwhm_deg=float(settings["fiber_tube_fwhm_deg"]),
        component_count_range=(tuple(map(int, count_range)) if count_range is not None else None),
        pvd_crystal_family=settings.get("pvd_crystal_family"),
        hybrid_process_probabilities=tuple(
            (str(name), float(probability))
            for name, probability in hybrid_probabilities.items()
        ),
    )


def _texture_index_log_weight(texture_index: float, physical: dict) -> float:
    lower, upper = map(float, physical.get("texture_index_bounds", (1.1, 35.0)))
    softness = float(physical.get("texture_index_softness", 2.0))
    if not 1 <= lower < upper or softness <= 0:
        raise ValueError("Texture-index bounds must satisfy 1 <= lower < upper and softness > 0.")
    log_weight = (-np.logaddexp(0.0, -(texture_index - lower) / softness)
                  - np.logaddexp(0.0, -(upper - texture_index) / softness))
    floor = float(physical.get("texture_index_log_weight_floor", -10.0))
    if floor > 0:
        raise ValueError("texture_index_log_weight_floor must be nonpositive.")
    return float(max(log_weight, floor))


def _build_design(num_samples: int, space, seed: int,
                  run: dict, system: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create a reproducible design and evaluate its analytic physical checks."""
    physical = _merge(run.get("physical_texture", {}), system.get("physical_texture", {}))
    is_physical = isinstance(space, PhysicalTextureSpace)
    default_policy = "soft_weight" if is_physical and space.inference_mode in {
        "likelihood", "broad_likelihood"
    } else (
        "hard_reject" if is_physical else "none"
    )
    policy = physical.get("texture_index_policy", default_policy)
    if policy not in {"none", "soft_weight", "hard_reject"}:
        raise ValueError("texture_index_policy must be none, soft_weight, or hard_reject.")
    bounds = tuple(map(float, physical.get("texture_index_bounds", (1.1, 35.0))))
    if len(bounds) != 2 or not 1 <= bounds[0] < bounds[1]:
        raise ValueError("texture_index_bounds must satisfy 1 <= lower < upper.")
    candidate_factor = int(physical.get("hard_rejection_candidate_factor", 16))
    if candidate_factor < 1:
        raise ValueError("hard_rejection_candidate_factor must be positive.")
    candidate_count = num_samples * candidate_factor if policy == "hard_reject" else num_samples
    candidates = sobol_unit_cube(candidate_count, space.dimension, seed=seed)

    accepted_points: list[np.ndarray] = []
    candidate_indices: list[int] = []
    indices: list[float] = []
    log_weights: list[float] = []
    sampling_spaces: list[str] = []
    inference_modes: list[str] = []
    process_families: list[str] = []
    component_manifolds: list[list[str]] = []
    component_kinds: list[list[str]] = []
    component_process_tags: list[list[str]] = []
    sample_process_domains: list[str] = []
    active_component_counts: list[int] = []
    component_active_masks: list[list[bool]] = []
    concentrations: list[float] = []
    for candidate_index, point in enumerate(candidates):
        if is_physical:
            odf, decoded = space.decode(point)
        else:
            odf = space.from_unit_cube(point)
            alpha_lo, alpha_hi = space.dirichlet_concentration_range
            decoded = {
                "sampling_space": "legacy_sobol_hypercube",
                "inference_mode": "legacy",
                "process_family": "unconstrained",
                "sample_process_domain": "unconstrained",
                "component_manifold": ["uniform_so3"] * odf.num_components,
                "component_manifold_kind": ["uniform"] * odf.num_components,
                "component_process_tag": ["haar_uniform"] * odf.num_components,
                "active_component_count": odf.num_components,
                "component_active_mask": [True] * odf.num_components,
                "dirichlet_concentration": float(
                    np.exp(np.log(alpha_lo) + point[1] * np.log(alpha_hi / alpha_lo))
                ),
            }
        texture_index = odf.texture_index()
        if policy == "hard_reject" and not bounds[0] <= texture_index <= bounds[1]:
            continue
        log_weight = (_texture_index_log_weight(texture_index, physical)
                      if policy == "soft_weight" else 0.0)
        accepted_points.append(point)
        candidate_indices.append(candidate_index)
        indices.append(texture_index)
        log_weights.append(log_weight)
        sampling_spaces.append(decoded["sampling_space"])
        inference_modes.append(decoded["inference_mode"])
        process_families.append(decoded["process_family"])
        sample_process_domains.append(decoded["sample_process_domain"])
        component_manifolds.append(decoded["component_manifold"])
        component_kinds.append(decoded["component_manifold_kind"])
        component_process_tags.append(decoded["component_process_tag"])
        active_component_counts.append(int(decoded["active_component_count"]))
        component_active_masks.append(decoded["component_active_mask"])
        concentrations.append(decoded["dirichlet_concentration"])
        if len(accepted_points) == num_samples:
            break
    if len(accepted_points) != num_samples:
        raise RuntimeError(
            f"Texture-index rejection accepted {len(accepted_points)} of {candidate_count} "
            f"candidates; increase physical_texture.hard_rejection_candidate_factor or soften the filter."
        )
    metadata = {
        "texture_index": np.asarray(indices, dtype=np.float64),
        "texture_index_log_weight": np.asarray(log_weights, dtype=np.float64),
        "sampling_space": np.asarray(sampling_spaces),
        "inference_mode": np.asarray(inference_modes),
        "process_family": np.asarray(process_families),
        "sample_process_domain": np.asarray(sample_process_domains),
        "component_manifold": np.asarray(component_manifolds),
        "component_manifold_kind": np.asarray(component_kinds),
        "component_process_tag": np.asarray(component_process_tags),
        "active_component_count": np.asarray(active_component_counts, dtype=np.int16),
        "component_active_mask": np.asarray(component_active_masks, dtype=bool),
        "dirichlet_concentration": np.asarray(concentrations, dtype=np.float64),
        "design_candidate_index": np.asarray(candidate_indices, dtype=np.int64),
    }
    return np.asarray(accepted_points, dtype=np.float64), metadata


def generate(spec: dict, *, limit: int | None = None, resume: bool = True,
             shard_index: int = 0, num_shards: int = 1,
             progress_interval: float = 30.0) -> int:
    run = spec["run"]
    if run.get("sampling_method", "Sobol_sequence") not in {
        "Sobol_sequence", "physical_sobol_manifold"
    }:
        raise ValueError(
            "sampling_method must be Sobol_sequence or physical_sobol_manifold."
        )
    if run.get("kernel_type", "de_la_vallee_poussin") != "de_la_vallee_poussin":
        raise ValueError("Only the exactly normalized de_la_vallee_poussin kernel is supported.")
    if not 0 <= shard_index < num_shards or num_shards < 1:
        raise ValueError("shard_index must be in [0, num_shards).")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive.")
    counts = grain_counts(spec)
    if int(run.get("max_texture_components", 5)) < 1:
        raise ValueError("max_texture_components must be positive.")
    output_root = Path(run["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "dataset_spec.json"
    canonical_spec = json.dumps(spec, indent=2, sort_keys=True)
    if spec_path.exists() and json.loads(spec_path.read_text()) != json.loads(canonical_spec):
        raise ValueError(f"{spec_path} differs from the requested specification; choose a new output_directory.")
    if not spec_path.exists():
        spec_path.write_text(canonical_spec + "\n")
    _write_json_exact(output_root / "odf_label_schema.json", _label_schema(spec))
    manifest_name = run.get("manifest", "manifest.jsonl")
    manifest_path = output_root / (manifest_name if num_shards == 1
                                   else f"{Path(manifest_name).stem}.shard_{shard_index:03d}{Path(manifest_name).suffix or '.jsonl'}")
    if resume:
        preview_manifests = [manifest_path]
        if num_shards == 1:
            stem = Path(manifest_name).stem
            suffix = Path(manifest_name).suffix or ".jsonl"
            preview_manifests.extend(output_root.glob(f"{stem}.shard_*{suffix}"))
        created = sum(_sync_png_previews(output_root, run, path) for path in preview_manifests)
        if created:
            print(f"Backfilled {created} PNG previews.")
    completed = _completed_ids(manifest_path) if resume else set()
    if resume and num_shards == 1:
        # A corpus generated in parallel remains resumable through the simpler
        # unsharded command; collect all shard manifests as completion records.
        stem = Path(manifest_name).stem
        suffix = Path(manifest_name).suffix or ".jsonl"
        for shard_manifest in output_root.glob(f"{stem}.shard_*{suffix}"):
            completed.update(_completed_ids(shard_manifest))
    include_grid, include_harmonics = _representation_mode(run)
    grid_points = int(run.get("odf_grid_points", 0))
    if include_grid and grid_points < 1:
        raise ValueError("odf_grid_points must be positive when requesting an ODF grid.")
    bandlimit = int(run.get("harmonic_bandlimit", 0)) if include_harmonics else None
    if bandlimit is not None and bandlimit < 0:
        raise ValueError("harmonic_bandlimit must be nonnegative.")

    num_samples = int(run["num_samples_per_symmetry"])
    design_seed = int(run.get("design_seed", run.get("seed", 0)))
    observation_seed = int(run.get("observation_seed", run.get("seed", 0)))
    image_format = spec.get("base_config", {}).get("output", {}).get("image_format", "auto")
    if image_format == "auto":
        raise ValueError("Continuous corpora require explicit output.image_format for a stable scan index.")
    systems_data = []
    paired_design = bool(run.get("paired_design_across_systems", False))
    paired_observations = bool(run.get("paired_observation_noise_across_systems", False))
    initialization_started = time.monotonic()
    print(
        f"[progress] initializing {len(spec['crystal_systems'])} system designs "
        f"({num_samples} ODFs per system)",
        flush=True,
    )
    for system_index, system in enumerate(spec["crystal_systems"]):
        name = _safe_name(system["name"])
        symmetry = system["orientation_symmetry"]
        space = _make_odf_space(run, system)
        if paired_design and systems_data and space.dimension != systems_data[0]["space"].dimension:
            raise ValueError("paired_design_across_systems requires equal sampling-space dimensions.")
        directional_settings = _directional_target_settings(run, system)
        system_dir = output_root / name
        system_dir.mkdir(parents=True, exist_ok=True)
        design_path = system_dir / "sobol_design.npy"
        label_path = system_dir / "odf_labels.npz"
        stored_design = _load_stored_design(
            design_path, label_path,
            num_samples=num_samples, dimension=space.dimension,
            component_count=space.max_components,
        )
        if stored_design is not None:
            points, design_metadata = stored_design
            print(f"[progress] reusing persisted ODF design for {name} ({num_samples} labels)",
                  flush=True)
        else:
            print(f"[progress] building ODF design for {name} ({num_samples} labels)",
                  flush=True)
            with _progress_heartbeat(f"initializing ODF design for {name}", progress_interval):
                points, design_metadata = _build_design(
                    num_samples, space,
                    seed=(design_seed if paired_design else int(np.random.SeedSequence(
                        design_seed, spawn_key=(system_index,)
                    ).generate_state(1)[0])),
                    run=run, system=system,
                )
            if design_path.exists():
                existing_design = np.load(design_path)
                if existing_design.shape != points.shape or not np.array_equal(existing_design, points):
                    raise ValueError(f"Existing {design_path} does not match this reproducible Sobol design.")
            else:
                np.save(design_path, points)
            print(f"[progress] writing ODF labels for {name}", flush=True)
            with _progress_heartbeat(f"writing ODF labels for {name}", progress_interval):
                _write_system_label_table(label_path, points, space, design_metadata)
        grid = None
        if include_grid:
            grid_path = system_dir / "odf_grid_quaternions.npy"
            if grid_path.exists():
                grid = np.load(grid_path)
                if grid.shape != (grid_points, 4):
                    raise ValueError(f"Existing {grid_path} has incompatible shape {grid.shape}.")
            else:
                grid = sobol_odf_grid(
                    grid_points,
                    seed=(design_seed + 999 if paired_design else int(np.random.SeedSequence(
                        design_seed, spawn_key=(system_index, 999)
                    ).generate_state(1)[0])),
                )
                np.save(grid_path, grid.astype(np.float32))
        systems_data.append({
            "system": system, "name": name, "symmetry": symmetry, "space": space,
            "system_dir": system_dir, "points": points, "grid": grid,
            "label_path": label_path, "design_metadata": design_metadata,
            "directional_targets": directional_settings,
            "paired_design": paired_design,
        })
    print("[progress] writing planned scan index", flush=True)
    with _progress_heartbeat("writing planned scan index", progress_interval):
        _write_scan_index(output_root, systems_data, counts, image_format)
    if run.get("write_hdf5_metadata", False):
        print("[progress] writing HDF5 metadata", flush=True)
        with _progress_heartbeat("writing HDF5 metadata", progress_interval):
            _write_hdf5_metadata(output_root, spec, systems_data)
    print(
        f"[progress] initialization complete in {time.monotonic() - initialization_started:.1f}s",
        flush=True,
    )

    generated = 0
    attempted_start = time.monotonic()
    print(
        f"[progress] starting shard {shard_index + 1}/{num_shards}; "
        f"{planned_runs(spec)} total simulations across all shards",
        flush=True,
    )
    with manifest_path.open("a") as manifest:
        for system_index, item in enumerate(systems_data):
            system, name, symmetry = item["system"], item["name"], item["symmetry"]
            space, system_dir, points, grid = item["space"], item["system_dir"], item["points"], item["grid"]
            design_metadata = item["design_metadata"]
            for sample_index, point in enumerate(points):
                if sample_index % num_shards != shard_index:
                    continue
                odf = space.from_unit_cube(point)
                sample_design_metadata = {
                    key: value[sample_index] for key, value in design_metadata.items()
                }
                odf_id = f"{name}__odf_{sample_index:06d}"
                for observation_index, count in enumerate(counts):
                    run_id = f"{odf_id}__g{count:06d}"
                    if run_id in completed:
                        continue
                    if limit is not None and generated >= limit:
                        return generated
                    sample_dir = system_dir / run_id
                    record = {
                        "run_id": run_id, "odf_id": odf_id,
                        "design_pair_id": (f"paired_sobol_{sample_index:06d}"
                                           if paired_design else odf_id),
                        "odf_label_index": sample_index,
                        "system": system["name"], "orientation_symmetry": symmetry,
                        "sample_index": sample_index, "observation_index": observation_index,
                        "grain_count": count, "status": "running",
                    }
                    run_started = time.monotonic()
                    try:
                        if sample_dir.exists() and any(sample_dir.iterdir()):
                            raise FileExistsError(f"{sample_dir} exists but has no completed manifest record; refusing to overwrite.")
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        seed = int(np.random.SeedSequence(
                            observation_seed,
                            spawn_key=((sample_index, observation_index)
                                       if paired_observations else
                                       (system_index, sample_index, observation_index)),
                        ).generate_state(1)[0])
                        config = _merge(spec["base_config"], system.get("simulation", {}))
                        config["texture"] = odf.to_dict()
                        config.setdefault("experiment", {})["seed"] = seed
                        config["experiment"]["num_grains"] = count
                        config.setdefault("output", {})["directory"] = str(sample_dir)
                        config["output"]["prefix"] = "simulation"
                        # ``write_tiff`` is a continuous-dataset driver option,
                        # not an option understood by the simulation validator.
                        # Accept it in legacy/base configs, but do not pass it
                        # through to validate_config().
                        write_tiff = run.get(
                            "write_tiff",
                            config["output"].pop("write_tiff", True),
                        )
                        config["output"].pop("write_tiff", None)
                        with _progress_heartbeat(run_id, progress_interval):
                            result = run_experiment(validate_config(config), save=True, odf=odf)
                        label_name = "odf_label.npz"
                        _write_representation(
                            sample_dir / label_name, odf, point, grid, bandlimit,
                            odf_label_index=sample_index, odf_id=odf_id,
                            grain_count=count, observation_seed=seed,
                            design_metadata=sample_design_metadata,
                        )
                        directional_settings = item["directional_targets"]
                        if directional_settings is not None:
                            target_path = sample_dir / directional_settings["filename"]
                            experiment = config["experiment"]
                            write_directional_targets(
                                target_path,
                                result["orientations"],
                                crystal_symmetry=symmetry,
                                pole_families=directional_settings["pole_families"],
                                sample_directions=directional_settings["sample_directions"],
                                grid_size=directional_settings["grid_size"],
                                coefficient_shape=directional_settings["coefficient_shape"],
                                unit_cell=experiment.get("unit_cell"),
                            )
                        record.update({
                            "status": "complete", "seed": seed, "path": str(sample_dir),
                            "odf": odf.to_dict(), "num_grains": result["num_grains"],
                            "num_peaks": result["num_peaks"], "odf_label": label_name,
                            "image": str(result["files"]["image"]),
                            "texture_index": float(sample_design_metadata["texture_index"]),
                            "texture_index_log_weight": float(
                                sample_design_metadata["texture_index_log_weight"]
                            ),
                            "sampling_space": str(sample_design_metadata["sampling_space"]),
                            "inference_mode": str(sample_design_metadata["inference_mode"]),
                            "process_family": str(sample_design_metadata["process_family"]),
                            "sample_process_domain": str(
                                sample_design_metadata["sample_process_domain"]
                            ),
                            "component_process_tag": sample_design_metadata["component_process_tag"],
                            "active_component_count": int(
                                sample_design_metadata["active_component_count"]
                            ),
                            "component_active_mask": sample_design_metadata["component_active_mask"],
                        })
                        if directional_settings is not None:
                            record["pf_ipf_targets"] = str(target_path)
                        if write_tiff:
                            tiff_path = sample_dir / "simulation_pattern.tiff"
                            _write_tiff(result["image"], tiff_path, result["metadata"])
                            record["tiff_image"] = str(tiff_path)
                        if run.get("write_png_previews", True):
                            preview_path = _png_preview_root(output_root, run) / f"{run_id}.png"
                            _write_png_preview(
                                result["image"], preview_path, run_id,
                                float(run.get("png_saturation_percentile", 99.5)),
                            )
                            record["preview_png"] = str(preview_path)
                    except Exception as exc:
                        record.update({"status": "error", "error": repr(exc),
                                       "traceback": traceback.format_exc(limit=8)})
                    manifest.write(json.dumps(record, default=_json_default) + "\n")
                    manifest.flush()
                    generated += 1
                    elapsed = time.monotonic() - run_started
                    status = record["status"]
                    print(
                        f"[progress] {status}: {run_id} ({elapsed:.1f}s); "
                        f"attempted {generated} in {time.monotonic() - attempted_start:.1f}s",
                        flush=True,
                    )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="continuous ODF dataset JSON specification")
    parser.add_argument("--limit", type=int, help="stop after this many new simulations")
    parser.add_argument("--no-resume", action="store_true", help="do not skip completed manifest records")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="zero-based shard index for distributed runs")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="number of disjoint sample-index shards")
    parser.add_argument("--progress-interval", type=float, default=30.0,
                        help="seconds between liveness updates during each simulation (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="report planned count only")
    parser.add_argument("--previews-only", action="store_true",
                        help="backfill missing PNG previews from completed manifest records")
    parser.add_argument("--refresh-previews", action="store_true",
                        help="rebuild existing PNG previews when used with --previews-only")
    parser.add_argument("--system", action="append",
                        help="generate only this configured crystal-system name; repeat for a subset")
    parser.add_argument("--samples-per-symmetry", type=int,
                        help="override the configured ODF count per selected system")
    parser.add_argument("--output-directory", type=Path,
                        help="override run.output_directory (required for a production subset/size override)")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.system:
        requested = set(args.system)
        known = {system["name"] for system in spec["crystal_systems"]}
        unknown = requested.difference(known)
        if unknown:
            parser.error(f"unknown --system values: {sorted(unknown)}; choose from {sorted(known)}")
        spec["crystal_systems"] = [
            system for system in spec["crystal_systems"] if system["name"] in requested
        ]
    if args.samples_per_symmetry is not None:
        if args.samples_per_symmetry < 1:
            parser.error("--samples-per-symmetry must be positive")
        spec["run"]["num_samples_per_symmetry"] = args.samples_per_symmetry
    if args.output_directory is not None:
        spec["run"]["output_directory"] = str(args.output_directory)
    if ((args.system or args.samples_per_symmetry is not None)
            and args.output_directory is None and not args.dry_run):
        parser.error("subset/size generation requires --output-directory to protect the full corpus")
    if args.dry_run:
        print(f"Planned ODF points: {planned_odf_points(spec)}")
        print(f"Grain counts per ODF: {list(grain_counts(spec))}")
        print(f"Planned simulations: {planned_runs(spec)}")
        return
    if args.previews_only:
        run = spec["run"]
        output_root = Path(run["output_directory"])
        manifest_name = run.get("manifest", "manifest.jsonl")
        manifest_paths = [output_root / manifest_name]
        manifest_paths.extend(output_root.glob(
            f"{Path(manifest_name).stem}.shard_*{Path(manifest_name).suffix or '.jsonl'}"
        ))
        created = sum(
            _sync_png_previews(output_root, run, manifest, overwrite=args.refresh_previews)
            for manifest in manifest_paths
        )
        print(f"Backfilled {created} PNG previews.")
        return
    generated = generate(spec, limit=args.limit, resume=not args.no_resume,
                         shard_index=args.shard_index, num_shards=args.num_shards,
                         progress_interval=args.progress_interval)
    print(f"Generated or attempted {generated} new simulations (planned total: {planned_runs(spec)}).")


if __name__ == "__main__":
    main()
