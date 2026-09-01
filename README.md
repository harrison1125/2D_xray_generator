Readme
# Quaternion orientations and prescribed texture

The active simulator now accepts Haar-uniform or prescribed ODF grain
orientations as unit quaternions.  See
[orientation and texture documentation](docs/orientation_and_texture.md) and
run `python examples/texture_workflow.py --grains 100 --seed 12345` for the
random, sharp/single-component, fiber, and two-component end-to-end workflow.
Per-grain ellipsoidal shape, peak broadening, and integrated intensity are
documented in [grain_shape_and_intensity.md](docs/grain_shape_and_intensity.md).
Use the single JSON-driven runner described in [configuration.md](docs/configuration.md)
for reproducible experiments.

## Synthetic texture datasets

[`generate_texture_dataset.py`](src/driver_codes/generate_texture_dataset.py)
executes long, resumable sweeps of explicitly specified ODF cases. A sweep JSON
contains a `base_config`, a `families` array, and a `dataset` section:

```bash
python src/driver_codes/generate_texture_dataset.py configs/texture_dataset_proof.json --dry-run
python src/driver_codes/generate_texture_dataset.py configs/texture_dataset_proof.json
python src/driver_codes/generate_texture_dataset.py configs/texture_dataset_broad.json --limit 10
```

Each family contains named ODF `cases` and a `replicates` count. The runner
assigns deterministic child seeds, writes one directory per run, and appends a
`manifest.jsonl` record after every run. Re-running the command resumes from
completed run IDs; use `--no-resume` only when deliberately regenerating them.
The supplied specifications are:

- `texture_dataset_proof.json`: a small smoke/proof corpus;
- `texture_dataset_broad.json`: random, component, fiber, mixtures, offset
  sputter fibers, and multiple-growth cases;
- `texture_dataset_sputter.json`: sweeps of growth-direction tilt, arc width,
  arc position, and smooth lopsidedness.

The generated data are conditional synthetic simulations, not a comprehensive
database of real texture. “Wide range” means coverage of the parameter values
declared in the sweep, not coverage of all crystallographic materials,
deposition mechanisms, instrument geometries, or possible ODFs. Each run's
metadata records the resolved configuration, ODF parameters, seed, grain
realizations, and peak records so that provenance and train/test partitioning
can be audited.

For Bayesian inference, split by ODF parameter region/family and seed—not only
by individual images—so interpolation and extrapolation can be measured. Keep
an out-of-distribution real-data evaluation separate. A posterior over this
corpus is conditional on the simulator; it should not be interpreted as a
physical posterior until the forward model is calibrated against standards and
real scans. In particular, the current diffraction path still uses the
prototype crystal/ detector model documented in
[configuration.md](docs/configuration.md), so HCP/non-cubic geometry,
instrument backgrounds, counting noise, and material-specific scattering must
be validated before scientific claims are made.

For a non-categorical, continuous Sobol design over symmetry-invariant ODF
mixtures, see [Continuous Sobol ODF dataset](docs/continuous_odf_dataset.md).
It includes a cubic-versus-monoclinic proof configuration and a 50,000-samples-
per-symmetry production specification, together with the mathematical and
scientific-validity constraints of the generated corpus.
