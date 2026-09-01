#!/usr/bin/env python3
"""Resumable, deterministic generation of synthetic texture/XRD datasets.

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


def _cases(spec):
    """Yield ``(family_name, case_name, odf_mapping, replicates)`` records."""
    for family in spec["families"]:
        family_name = family["name"]
        cases = family.get("cases")
        if not cases:
            raise ValueError(f"Family {family_name!r} must contain non-empty 'cases'.")
        for index, odf in enumerate(cases):
            case_name = family.get("case_names", {}).get(str(index), f"case_{index:03d}")
            yield family_name, case_name, odf, int(family.get("replicates", 1))


def planned_runs(spec):
    """Return the total number of runs implied by a sweep specification."""
    return sum(reps for _, _, _, reps in _cases(spec))


def generate_dataset(spec, *, limit=None, resume=True):
    dataset = spec["dataset"]
    output_root = Path(dataset["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / dataset.get("manifest", "manifest.jsonl")
    completed = set()
    if resume and manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                record = json.loads(line)
                if record.get("status") == "complete":
                    completed.add(record["run_id"])
            except json.JSONDecodeError:
                continue

    base = spec["base_config"]
    root_seed = int(dataset.get("seed", 0))
    generated = 0
    with manifest_path.open("a") as manifest:
        for run_number, (family_name, case_name, odf, replicates) in enumerate(_cases(spec)):
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
                    record.update({"status": "complete", "path": str(run_dir),
                                   "odf": result["metadata"].get("texture", odf),
                                   "num_grains": len(result["grains"]),
                                   "num_peaks": len(result["peaks"])})
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
    parser.add_argument("--dry-run", action="store_true", help="print planned run count without simulating")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    total = planned_runs(spec)
    if args.dry_run:
        print(f"Planned runs: {total}")
        return
    generated = generate_dataset(spec, limit=args.limit, resume=not args.no_resume)
    print(f"Generated or attempted {generated} new runs (planned total: {total}).")


if __name__ == "__main__":
    main()
