# Texture corpus commands

Run commands from the repository root with the project environment.

## Continuous sputter/PVD corpus

Validate the full PVD plan without rendering images:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --dry-run
```

Start with a small staged run. Re-running the same command resumes completed
samples through the manifest:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --limit 100
```

Split one deterministic Sobol design across workers with mutually exclusive
sample indices:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json \
  --num-shards 8 --shard-index 0
```

Do not use `--no-resume` against an existing corpus. Change
`run.output_directory` to begin a new corpus version.

Open generated PNG previews after a run:

```bash
open -a Preview datasets/continuous_odf_sputter_pvd_broad_v1/png_preview/*.png
```

## Curated sputter validation

The compact curated validation sweep is useful for inspecting expected
film-normal FCC behavior before a large continuous simulation:

```bash
uv run src/driver_codes/generate_curated_texture_validation.py \
  configs/curated_sputter_validation.json --dry-run

uv run src/driver_codes/generate_curated_texture_validation.py \
  configs/curated_sputter_validation.json
```

It samples explicit tilt, azimuthal arc, and lopsidedness cases. To rebuild
only visualization mirrors from an existing run:

```bash
uv run src/driver_codes/generate_curated_texture_validation.py \
  configs/curated_sputter_validation.json --previews-only
```

All generated output under `datasets/` is ignored by Git. Quantitative work
should use scientific NumPy/TIFF arrays and structured metadata, not PNG
previews.
