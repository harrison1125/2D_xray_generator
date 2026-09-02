# Running a configured experiment

Use one JSON file for every run. The complete schema and a runnable example
are in [fcc_texture_grain_example.json](../configs/fcc_texture_grain_example.json).
All texture and grain-size inputs are explicit, including the seed.

```bash
source .venv/bin/activate
python src/driver_codes/run_experiment.py \
  configs/fcc_texture_grain_example.json
```

Create the environment from the repository root with:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Conda environments and the former Poetry invocation are deprecated for this
repository. The installed console entry point is also available as:

```bash
xrd-generate configs/fcc_texture_grain_example.json
```

Configuration sections:

- `experiment`: material label, crystal structure, lattice parameter,
  wavelength, grain count, random seed, and the currently supported `nm`
  lattice/wavelength unit.
- `grain`: equivalent-volume size mean/standard deviation, explicit `nm` or
  `um` size unit, positive size distribution (`lognormal` by default),
  microstrain mean/standard deviation, and ellipsoid aspect ratio. The driver
  performs the grain/lattice unit conversion explicitly.
- `texture`: an ODF mapping: `random`, `sharp`, `component`, `fiber`,
  `partial_fiber` (aliases `sputter_fiber` and `offset_fiber`), `empirical`, or
  `mixture`. Partial fibers support tilted growth directions, half-rings via
  `spin_width_deg`, and smooth lopsidedness via `spin_kappa`. It uses the
  quaternion conventions described in
  [orientation_and_texture.md](orientation_and_texture.md).
- `detector`: native pixel dimensions, physical distance/pixel pitch, optional
  `bin_to_width_px`/`bin_to_height_px` readout dimensions, Ewald candidate
  window, and central-beam blocker. Binning is flux-conserving and occurs after
  native-resolution rendering. Candidate reflections receive a continuous
  finite-size excitation weight rather than a flat tolerance acceptance.
- `scattering`: Scherrer shape factor, `voigt` or compatibility `gaussian`
  profile, calibrated instrumental FWHM, static/legacy Lorentz choice, and
  intensity scale per cubic micrometre.
- `output`: result directory and filename prefix.

`experiment.crystal_structure` supports `SC`, `FCC`, `BCC`, `HCP`,
`MONOCLINIC`, and `TRICLINIC`. The low-symmetry paths require an explicit
`unit_cell` object with `a`, `b`, `c`, `alpha_deg`, `beta_deg`, and `gamma_deg`;
they use `GrainGeneral` rather than the cubic reciprocal lattice. Use
`max_hkl_index` to control reciprocal-lattice truncation. `output.store_grains`
and `output.store_peaks` may be set false for large batch datasets where exact
ODF parameters are retained separately.

The driver writes the detector image (`.tiff`, or `.npy` without `tifffile`),
the complete resolved configuration, the realized grain sizes/strains/
quaternions, and per-reflection width/intensity records. The metadata records
every default that was applied, so it is sufficient to reproduce a run.

Curated texture-validation specifications may set `dataset.write_png_previews`,
`dataset.png_preview_directory`, and `dataset.png_saturation_percentile`. The
dataset driver creates a flat preview folder with one PNG named by run ID. PNGs
use a per-image `log1p` grayscale display transform with the brightest 0.5% of
positive pixels saturated by default. They are QA views only and never replace
or modify the scientific image arrays. `--previews-only` backfills missing
files; add `--refresh-previews` to rebuild existing previews.
