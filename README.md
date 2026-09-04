# XRD Generator

This repository generates synthetic 2D X-ray diffraction patterns from
quaternion grain orientations and analytic Orientation Distribution Functions
(ODFs). The maintained dataset path is a sputter/PVD broad-likelihood corpus
for Bayesian inverse inference; it models film-normal fiber texture, sharp
biaxial growth, and a Haar-uniform nanocrystalline escape channel.

The simulator is suitable for controlled identifiability experiments. It is
not yet a calibrated experimental forward model: material structure factors,
instrument response, background, counting noise, and the sample-specific
microstructure prior still require validation against standards.

## Setup

```bash
brew install uv
uv venv .venv
uv pip install -e .
```

Run commands with `uv run`; this consistently uses the project environment.

## Maintained configurations

| Configuration | Purpose |
|---|---|
| `balanced_symmetry_odf_pf_v1.json` | Matched 25k-per-class FCC/BCC/HCP benchmark with analytic ODF and realized PF/IPF targets. |
| `continuous_odf_sputter_pvd_broad.json` | Stage-1 PVD/sputter broad-likelihood ODF corpus: FCC Cu, BCC Fe, and HCP Ti. |
| `continuous_odf_phase_space_v2.json` | Unconstrained phase-space corpus for FCC Cu, BCC Fe, and HCP Ti; reproduces the prior PNG/TIFF layout. |
| `curated_sputter_validation.json` | Small, interpretable Cu PVD validation sweeps for tilt, fiber arc width, and lopsidedness. |
| `fcc_texture_grain_example.json` | Single-run FCC texture/grain simulation example. |

Check the continuous corpus plan without generating data:

```bash
uv run --no-sync src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --dry-run
```

The production specification requests 10,000 ODFs for each crystal system,
with one 6,000-grain observation per ODF. Start a staged run deliberately:

```bash
uv run --no-sync src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --limit 100
```

All dataset output belongs under `datasets/` and is ignored by Git. Each run is
resumable through `manifest.jsonl`; do not use `--no-resume` against an
existing corpus. See [the command reference](docs/texture_corpus_commands.md)
for validation, sharding, and preview commands.

## ODF targets and inference scope

The continuous target is a variable-slot version-2 descriptor. With the PVD
configuration's six component slots, it has 37 float32 values:

```text
[background (1), component weights (6), centers wxyz (24), FWHM degrees (6)]
```

Use the run's `odf_label_schema.json`, not hard-coded offsets. Metadata records
the analytic texture index $J$, active-slot mask, component origin tag, and
process family. The PVD sampler uses the intended per-component allocation:
65% `sputter_fiber`, 20% `sputter_biaxial`, and 15%
`nanocrystalline_uniform`.

The physical construction and its limitations are documented in
[scientific_texture_space.md](docs/scientific_texture_space.md). In particular,
a finite mixture along a film-normal fiber approximates, but is not identical
to, a continuous axisymmetric fiber ODF. A posterior learned from this corpus
is conditional on this generator distribution; use experimentally derived
information such as EBSD as an explicit prior or comparison distribution,
rather than treating the corpus as universal texture support.
