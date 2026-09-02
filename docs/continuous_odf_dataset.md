# Continuous PVD ODF dataset

`generate_continuous_odf_dataset.py` is the maintained generator for the
continuous sputter/PVD ODF corpus. Its complete scientific specification,
including the physical assumptions and Bayesian limitations, is in
[scientific_texture_space.md](scientific_texture_space.md).

## Run the maintained corpus

Validate the configuration without rendering diffraction patterns:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --dry-run
```

The configuration defines 10,000 ODFs for each of FCC Cu, BCC Fe, and HCP Ti.
Every ODF has one 6,000-grain detector observation. Begin with a finite staged
run, then resume it with the same command:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --limit 100
```

For independent workers, partition a single deterministic Sobol design by
index. Each shard writes a distinct portion of the same corpus:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json \
  --num-shards 8 --shard-index 0
```

## Outputs

Output is written to `datasets/continuous_odf_sputter_pvd_broad_v1/`, which is
intentionally ignored by Git. Each system directory contains a shared
`odf_labels.npz`; each observation has `odf_label.npz`, a float32 NumPy image,
a TIFF mirror, and a PNG preview. The run root contains:

- `planned_scan_index.npz` for direct training-loader lookup;
- `odf_label_schema.json` defining every target field;
- `metadata.h5` containing the scan index and ODF label tables; and
- `manifest.jsonl` for resumable generation.

The detector image used for training is a flux-conservingly binned 256 x 256
float32 array. TIFF and PNG outputs are respectively scientific interchange
and visualization mirrors; use the NumPy/TIFF data for quantitative work.

## Target and provenance

The PVD configuration has up to six DVP components, so descriptor version 2 is
37 float32 values:

```text
[background (1), weights (6), centers quaternion wxyz (24), FWHM degrees (6)]
```

Components beyond the per-ODF active count have zero weight. The metadata
therefore includes `active_component_count`, `component_active_mask`,
`component_process_tag`, `sample_process_domain`, and analytic texture index
`J`. Always load `odf_label_schema.json` rather than assuming a fixed 31-value
legacy descriptor. Keep all observations of the same `odf_id` in one
train/validation/test split.

The configured component catalog has 65% film-normal fiber, 20% sharp biaxial
growth, and 15% Haar-uniform nanocrystalline components. Those tags are
provenance labels, not inferred physical truth. See the scientific document
for the finite-fiber approximation and prior-support cautions.
