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
