# Running a configured experiment

Use one JSON file for every run. The complete schema and a runnable example
are in [fcc_texture_grain_example.json](../configs/fcc_texture_grain_example.json).
All texture and grain-size inputs are explicit, including the seed.

```bash
conda run -n xraygenerator python src/driver_codes/run_experiment.py \
  configs/fcc_texture_grain_example.json
```

When installed with Poetry, the same single driver is available as
`poetry run xrd-generate configs/fcc_texture_grain_example.json`.

Configuration sections:

- `experiment`: material label, crystal structure, lattice parameter,
  wavelength, grain count, and random seed.
- `grain`: equivalent-volume size mean/standard deviation, microstrain
  mean/standard deviation, and ellipsoid aspect ratio. Size and wavelength
  must use the same length unit.
- `texture`: an ODF mapping: `random`, `sharp`, `component`, `fiber`, or
  `mixture`. It uses the quaternion conventions described in
  [orientation_and_texture.md](orientation_and_texture.md).
- `detector`: pixel dimensions, physical distance/pixel pitch, Ewald tolerance,
  and central-beam blocker.
- `scattering`: Scherrer shape factor, instrumental FWHM, and intensity scale.
- `output`: result directory and filename prefix.

The driver writes the detector image (`.tiff`, or `.npy` without `tifffile`),
the complete resolved configuration, the realized grain sizes/strains/
quaternions, and per-reflection width/intensity records. The metadata records
every default that was applied, so it is sufficient to reproduce a run.
