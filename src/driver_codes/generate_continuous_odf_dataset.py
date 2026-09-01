#!/usr/bin/env python3
"""Generate a dense, continuous, Sobol-designed ODF/XRD dataset.

Unlike ``generate_texture_dataset.py``, this driver does not classify examples
into hand-authored texture families.  It samples one fixed-width continuous
mixture space on SO(3), explicitly symmetrized for each crystal system.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classes_and_functions.configuration import validate_config
from src.classes_and_functions.continuous_odf import (
    SobolODFSpace, harmonic_coefficients, pack_harmonic_coefficients,
    sobol_odf_grid, sobol_unit_cube,
)
from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment


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


def _representation_mode(run: dict) -> tuple[bool, bool]:
    mode = run.get("output_format", "harmonic_coefficients_and_odf_grid")
    valid = {"parameters_only", "odf_grid", "harmonic_coefficients", "harmonic_coefficients_and_odf_grid"}
    if mode not in valid:
        raise ValueError(f"Unsupported output_format {mode!r}; expected one of {sorted(valid)}.")
    return "odf_grid" in mode, "harmonic" in mode


def _write_representation(path: Path, odf, sobol_point: np.ndarray, grid: np.ndarray | None,
                          bandlimit: int | None) -> None:
    """Write exact latent parameters plus optional ODF views in one NPZ file."""
    arrays: dict[str, np.ndarray] = {
        "sobol_point": np.asarray(sobol_point, dtype=np.float64),
        "component_centers": odf.component_centers.astype(np.float64),
        "component_fwhm_deg": odf.component_fwhm_deg.astype(np.float64),
        "component_weights": odf.component_weights.astype(np.float64),
        "background_weight": np.asarray(odf.background_weight, dtype=np.float64),
        "effective_components": np.asarray(odf.effective_components, dtype=np.float64),
    }
    if grid is not None:
        arrays["odf_grid_values"] = odf.evaluate(grid).astype(np.float32)
    if bandlimit is not None:
        arrays.update(pack_harmonic_coefficients(harmonic_coefficients(odf, bandlimit)))
        arrays["harmonic_bandlimit"] = np.asarray(bandlimit, dtype=np.int16)
    np.savez_compressed(path, **arrays)


def planned_runs(spec: dict) -> int:
    return int(spec["run"]["num_samples_per_symmetry"]) * len(spec["crystal_systems"])


def generate(spec: dict, *, limit: int | None = None, resume: bool = True,
             shard_index: int = 0, num_shards: int = 1) -> int:
    run = spec["run"]
    if run.get("sampling_method", "Sobol_sequence") != "Sobol_sequence":
        raise ValueError("Only Sobol_sequence is supported by the continuous ODF driver.")
    if run.get("kernel_type", "de_la_vallee_poussin") != "de_la_vallee_poussin":
        raise ValueError("Only the exactly normalized de_la_vallee_poussin kernel is supported.")
    if not 0 <= shard_index < num_shards or num_shards < 1:
        raise ValueError("shard_index must be in [0, num_shards).")
    output_root = Path(run["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "dataset_spec.json"
    canonical_spec = json.dumps(spec, indent=2, sort_keys=True)
    if spec_path.exists() and json.loads(spec_path.read_text()) != json.loads(canonical_spec):
        raise ValueError(f"{spec_path} differs from the requested specification; choose a new output_directory.")
    if not spec_path.exists():
        spec_path.write_text(canonical_spec + "\n")
    manifest_name = run.get("manifest", "manifest.jsonl")
    manifest_path = output_root / (manifest_name if num_shards == 1
                                   else f"{Path(manifest_name).stem}.shard_{shard_index:03d}{Path(manifest_name).suffix or '.jsonl'}")
    completed = _completed_ids(manifest_path) if resume else set()
    include_grid, include_harmonics = _representation_mode(run)
    grid_points = int(run.get("odf_grid_points", 0))
    if include_grid and grid_points < 1:
        raise ValueError("odf_grid_points must be positive when requesting an ODF grid.")
    bandlimit = int(run.get("harmonic_bandlimit", 0)) if include_harmonics else None
    if bandlimit is not None and bandlimit < 0:
        raise ValueError("harmonic_bandlimit must be nonnegative.")

    num_samples = int(run["num_samples_per_symmetry"])
    root_seed = int(run.get("seed", 0))
    generated = 0
    with manifest_path.open("a") as manifest:
        for system_index, system in enumerate(spec["crystal_systems"]):
            name = _safe_name(system["name"])
            symmetry = system["orientation_symmetry"]
            space = SobolODFSpace(
                crystal_symmetry=symmetry,
                max_components=int(run.get("max_texture_components", 5)),
                fwhm_range_deg=tuple(run.get("spread_range_degrees", (2.5, 30.0))),
                background_range=tuple(run.get("isotropic_background_range", (0.05, 0.95))),
                dirichlet_concentration_range=tuple(run.get("dirichlet_concentration_range", (0.2, 4.0))),
                spread_scale=run.get("spread_scale", "linear"),
            )
            system_dir = output_root / name
            system_dir.mkdir(parents=True, exist_ok=True)
            points = sobol_unit_cube(num_samples, space.dimension,
                                     seed=int(np.random.SeedSequence(root_seed, spawn_key=(system_index,)).generate_state(1)[0]))
            design_path = system_dir / "sobol_design.npy"
            if design_path.exists():
                existing_design = np.load(design_path)
                if existing_design.shape != points.shape or not np.array_equal(existing_design, points):
                    raise ValueError(f"Existing {design_path} does not match this reproducible Sobol design.")
            else:
                np.save(design_path, points)
            grid = None
            if include_grid:
                grid_path = system_dir / "odf_grid_quaternions.npy"
                if grid_path.exists():
                    grid = np.load(grid_path)
                    if grid.shape != (grid_points, 4):
                        raise ValueError(f"Existing {grid_path} has incompatible shape {grid.shape}.")
                else:
                    grid = sobol_odf_grid(grid_points, seed=int(np.random.SeedSequence(root_seed, spawn_key=(system_index, 999)).generate_state(1)[0]))
                    np.save(grid_path, grid.astype(np.float32))

            for sample_index, point in enumerate(points):
                if sample_index % num_shards != shard_index:
                    continue
                run_id = f"{name}__{sample_index:06d}"
                if run_id in completed:
                    continue
                if limit is not None and generated >= limit:
                    return generated
                sample_dir = system_dir / run_id
                record = {"run_id": run_id, "system": system["name"], "orientation_symmetry": symmetry,
                          "sample_index": sample_index, "status": "running"}
                try:
                    if sample_dir.exists() and any(sample_dir.iterdir()):
                        raise FileExistsError(f"{sample_dir} exists but has no completed manifest record; refusing to overwrite.")
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    odf = space.from_unit_cube(point)
                    seed = int(np.random.SeedSequence(root_seed, spawn_key=(system_index, sample_index)).generate_state(1)[0])
                    config = _merge(spec["base_config"], system.get("simulation", {}))
                    config["texture"] = odf.to_dict()
                    config.setdefault("experiment", {})["seed"] = seed
                    config.setdefault("output", {})["directory"] = str(sample_dir)
                    config["output"]["prefix"] = "simulation"
                    result = run_experiment(validate_config(config), save=True, odf=odf)
                    _write_representation(sample_dir / "odf_representation.npz", odf, point, grid, bandlimit)
                    record.update({"status": "complete", "seed": seed, "path": str(sample_dir),
                                   "odf": odf.to_dict(), "num_grains": len(result["grains"]),
                                   "num_peaks": len(result["peaks"]),
                                   "representation": "odf_representation.npz"})
                except Exception as exc:
                    record.update({"status": "error", "error": repr(exc),
                                   "traceback": traceback.format_exc(limit=8)})
                manifest.write(json.dumps(record, default=_json_default) + "\n")
                manifest.flush()
                generated += 1
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
    parser.add_argument("--dry-run", action="store_true", help="report planned count only")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.dry_run:
        print(f"Planned simulations: {planned_runs(spec)}")
        return
    generated = generate(spec, limit=args.limit, resume=not args.no_resume,
                         shard_index=args.shard_index, num_shards=args.num_shards)
    print(f"Generated or attempted {generated} new simulations (planned total: {planned_runs(spec)}).")


if __name__ == "__main__":
    main()
