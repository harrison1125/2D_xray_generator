# Continuous PVD ODF dataset

`generate_continuous_odf_dataset.py` is the maintained generator for the
continuous sputter/PVD ODF corpus. Its complete scientific specification,
including the physical assumptions and Bayesian limitations, is in
[scientific_texture_space.md](scientific_texture_space.md).

## Balanced symmetry benchmark

For class-balanced ODF-versus-PF/IPF training, use
`configs/balanced_symmetry_odf_pf_v1.json`. It samples 25,000 ODFs for each of
FCC, BCC, and HCP. All classes use the same Sobol coordinates, matched random
streams, detector, grain count, size distribution, and ODF hyperparameter
ranges. `design_pair_id` keeps each matched class triplet together during a
grouped train/test split.

The configuration intentionally uses the broad `Sobol_sequence` reference
measure with no texture-index rejection and no process-family prior. That is a
declared sampling measure, not a claim of absolute lack of bias. Material,
lattice, reflection, and detector physics remain part of the benchmark.

Each observation writes `pf_ipf_targets.npz` beside `odf_label.npz`. It holds
three configured pole-family probability maps, RD/TD/ND inverse-pole-figure
maps, a valid Lambert-disk mask, and a compact 9 x 9 orthonormal DCT prefix for
each channel. These directional targets come from the exact finite orientation
realization used to render that diffraction pattern; the ODF descriptor is the
analytic latent distribution.

The 25k-per-class default is approximately 22.5 GB without TIFF mirrors. Use
`--system`, `--samples-per-symmetry`, and a required distinct
`--output-directory` for disk-limited family subsets. A 50k-per-class full
corpus is approximately 45 GB and should not be attempted on a volume with
only 45 GiB free.

## Run the maintained corpus

Validate the configuration without rendering diffraction patterns:

```bash
uv run --no-sync src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --dry-run
```

The configuration defines 10,000 ODFs for each of FCC Cu, BCC Fe, and HCP Ti.
Every ODF has one 6,000-grain detector observation. Begin with a finite staged
run, then resume it with the same command:

```bash
uv run --no-sync src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --limit 100
```

The first run creates (or validates) all 30,000 ODF labels, then writes the
planned scan index and HDF5 metadata before it renders its first image. These
are reported as `initializing`, `writing planned scan index`, and `writing
HDF5 metadata` progress stages. Re-running an initialized corpus reuses its
stored design and skips the analytic texture-index calculation.

To reproduce the earlier unconstrained phase-space corpus across FCC, BCC, and
HCP and its output layout, use the historical
`configs/continuous_odf_phase_space_v2.json` specification instead. It is not
the maintained PVD prior, so use the broad PVD configuration for new film
texture data.

For independent workers, partition a single deterministic Sobol design by
index. Each shard writes a distinct portion of the same corpus:

```bash
uv run --no-sync src/driver_codes/generate_continuous_odf_dataset.py \
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
