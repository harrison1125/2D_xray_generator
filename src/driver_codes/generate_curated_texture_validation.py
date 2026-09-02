#!/usr/bin/env python3
"""Generate the small, curated texture-validation XRD corpus.

The input is a JSON sweep specification.  Each family contains explicit ODF
cases and a replicate count; every replicate receives a deterministic child
seed and its own output directory.  A JSONL manifest is appended after each
successful or failed run, so interrupted long jobs can be resumed safely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image, PngImagePlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classes_and_functions.configuration import validate_config
from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)).strip("_") or "case"


def _validation_cases(spec):
    """Yield ``(family_name, case_name, odf_mapping, replicates)`` records."""
    for family in spec["families"]:
        family_name = family["name"]
        cases = family.get("cases")
        if not cases:
            raise ValueError(f"Family {family_name!r} must contain non-empty 'cases'.")
        for index, odf in enumerate(cases):
            case_name = family.get("case_names", {}).get(str(index), f"case_{index:03d}")
            yield family_name, case_name, odf, int(family.get("replicates", 1))


def planned_validation_runs(spec):
    """Return the number of curated validation runs in a specification."""
    return sum(reps for _, _, _, reps in _validation_cases(spec))


def _png_preview_root(output_root, dataset):
    relative = Path(dataset.get("png_preview_directory", "png_preview"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("png_preview_directory must be a subdirectory of the dataset root.")
    return output_root / relative


def _png_preview_pixels(image, saturation_percentile=99.5):
    """Return an 8-bit, white-on-black log display without changing source data."""
    if not 0 < saturation_percentile <= 100:
        raise ValueError("png_saturation_percentile must lie in (0, 100].")
    finite = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    nonnegative = np.clip(finite, 0.0, None)
    maximum = float(nonnegative.max()) if nonnegative.size else 0.0
    if maximum <= 0:
        return np.zeros(nonnegative.shape, dtype=np.uint8), maximum, 0.0
    transformed = np.log1p(nonnegative)
    positive = transformed[transformed > 0]
    display_ceiling = float(np.percentile(positive, saturation_percentile))
    display = transformed / max(display_ceiling, np.finfo(float).eps)
    return (np.rint(255.0 * np.clip(display, 0.0, 1.0)).astype(np.uint8),
            maximum, display_ceiling)


def _write_png_preview(image, destination, run_id, saturation_percentile=99.5):
    """Atomically write a scrollable PNG visualization for one scientific image."""
    pixels, maximum, display_ceiling = _png_preview_pixels(
        image, saturation_percentile=saturation_percentile
    )
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("run_id", run_id)
    metadata.add_text(
        "display_transform",
        f"uint8(log1p(max(image,0)); positive-pixel p{saturation_percentile:g}=255)",
    )
    metadata.add_text("source_shape", "x".join(map(str, np.asarray(image).shape)))
    metadata.add_text("source_sum", f"{float(np.nansum(image)):.17g}")
    metadata.add_text("source_max", f"{maximum:.17g}")
    metadata.add_text("log_display_ceiling", f"{display_ceiling:.17g}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    Image.fromarray(pixels).save(temporary, format="PNG", pnginfo=metadata)
    temporary.replace(destination)


def _load_stored_pattern(run_dir):
    """Load the scientific image for preview backfilling."""
    run_dir = Path(run_dir)
    npy_paths = sorted(run_dir.glob("*_pattern.npy"))
    if npy_paths:
        return np.load(npy_paths[0])
    tiff_paths = sorted(run_dir.glob("*_pattern.tif")) + sorted(run_dir.glob("*_pattern.tiff"))
    if tiff_paths:
        with Image.open(tiff_paths[0]) as image:
            return np.asarray(image)
    raise FileNotFoundError(f"No stored detector pattern found in {run_dir}.")


def _completed_records(manifest_path):
    records = {}
    if not manifest_path.exists():
        return records
    for line in manifest_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "complete":
            records[record["run_id"]] = record
    return records


def sync_validation_png_previews(spec, *, overwrite=False):
    """Backfill missing flat PNG mirrors for all completed runs."""
    dataset = spec["dataset"]
    if not dataset.get("write_png_previews", True):
        return 0, 0
    output_root = Path(dataset["output_directory"])
    manifest_path = output_root / dataset.get("manifest", "manifest.jsonl")
    records = _completed_records(manifest_path)
    preview_root = _png_preview_root(output_root, dataset)
    saturation_percentile = float(dataset.get("png_saturation_percentile", 99.5))
    created = 0
    for run_id, record in sorted(records.items()):
        destination = preview_root / f"{run_id}.png"
        if destination.exists() and not overwrite:
            continue
        run_dir = Path(record.get("path", output_root / run_id))
        _write_png_preview(_load_stored_pattern(run_dir), destination, run_id,
                           saturation_percentile=saturation_percentile)
        created += 1
    return created, len(records)


def generate_validation_corpus(spec, *, limit=None, resume=True):
    dataset = spec["dataset"]
    output_root = Path(dataset["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / dataset.get("manifest", "manifest.jsonl")
    preview_root = _png_preview_root(output_root, dataset)
    write_previews = dataset.get("write_png_previews", True)
    saturation_percentile = float(dataset.get("png_saturation_percentile", 99.5))
    if resume:
        created, completed_count = sync_validation_png_previews(spec)
        if created:
            print(f"Backfilled {created} PNG previews for {completed_count} completed runs.")
    completed = set(_completed_records(manifest_path)) if resume else set()

    base = spec["base_config"]
    root_seed = int(dataset.get("seed", 0))
    generated = 0
    with manifest_path.open("a") as manifest:
        for run_number, (family_name, case_name, odf, replicates) in enumerate(_validation_cases(spec)):
            for replicate in range(replicates):
                run_id = f"{_safe_name(family_name)}__{_safe_name(case_name)}__r{replicate:04d}"
                if run_id in completed:
                    continue
                if limit is not None and generated >= limit:
                    return generated
                # Include replicate in the child path; otherwise every
                # replicate of a case would receive the same realization.
                child_seed = int(np.random.SeedSequence(root_seed, spawn_key=(run_number, replicate)).generate_state(1)[0])
                run_dir = output_root / run_id
                config = json.loads(json.dumps(base))
                config["texture"] = odf
                config.setdefault("experiment", {})["seed"] = child_seed
                config.setdefault("output", {})["directory"] = str(run_dir)
                config["output"]["prefix"] = "simulation"
                record = {"run_id": run_id, "family": family_name, "case": case_name,
                          "replicate": replicate, "seed": child_seed, "status": "running"}
                try:
                    resolved = validate_config(config)
                    result = run_experiment(resolved, save=True)
                    preview_path = preview_root / f"{run_id}.png"
                    if write_previews:
                        _write_png_preview(
                            result["image"], preview_path, run_id,
                            saturation_percentile=saturation_percentile,
                        )
                    record.update({"status": "complete", "path": str(run_dir),
                                   "odf": result["metadata"].get("texture", odf),
                                   "num_grains": result["num_grains"],
                                   "num_peaks": result["num_peaks"],
                                   "preview_png": str(preview_path) if write_previews else None})
                except Exception as exc:  # retain failure provenance and continue the sweep
                    record.update({"status": "error", "error": repr(exc),
                                   "traceback": traceback.format_exc(limit=8)})
                manifest.write(json.dumps(record, default=_json_default) + "\n")
                manifest.flush()
                generated += 1
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON sweep specification")
    parser.add_argument("--limit", type=int, help="stop after this many new runs")
    parser.add_argument("--no-resume", action="store_true", help="rerun IDs already complete in the manifest")
    parser.add_argument("--dry-run", action="store_true", help="print curated validation run count without simulating")
    parser.add_argument("--previews-only", action="store_true",
                        help="backfill missing PNG previews without simulating")
    parser.add_argument("--refresh-previews", action="store_true",
                        help="overwrite existing PNG previews using current display settings")
    args = parser.parse_args()
    if args.refresh_previews and not args.previews_only:
        parser.error("--refresh-previews requires --previews-only")
    spec = json.loads(args.spec.read_text())
    total = planned_validation_runs(spec)
    if args.dry_run:
        print(f"Planned runs: {total}")
        return
    if args.previews_only:
        created, completed = sync_validation_png_previews(spec, overwrite=args.refresh_previews)
        verb = "Refreshed" if args.refresh_previews else "Created"
        print(f"{verb} {created} PNG previews for {completed} completed runs.")
        return
    generated = generate_validation_corpus(spec, limit=args.limit, resume=not args.no_resume)
    print(f"Generated or attempted {generated} new runs (planned total: {total}).")


if __name__ == "__main__":
    main()
