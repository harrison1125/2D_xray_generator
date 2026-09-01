# Orientation and texture conventions

The simulator stores every grain orientation as a normalized quaternion
`[w, x, y, z]`.  It is a right-handed, active crystal-to-sample rotation:
`G_sample = R(q) G_crystal` for column reciprocal vectors.  The grain keeps an
immutable crystal-frame reciprocal lattice and derives its sample-frame lattice
directly from this matrix; diffraction never converts quaternions through Euler
angles.

Euler input/output uses active intrinsic Bunge ZXZ angles `(phi1, Phi, phi2)` in
radians by default (degrees only when requested):
`R = Rz(phi1) Rx(Phi) Rz(phi2)`.  This convention is for compatibility,
debugging, and conventional ODF sections only.  ODF density is normalized with
respect to Haar measure on SO(3), rather than the flat Euler-angle measure.  In
Bunge coordinates the corresponding volume element is proportional to
`sin(Phi) dphi1 dPhi dphi2`.

`RandomODF` uses Shoemake's Haar-uniform quaternion algorithm.  A component has
density `exp(-d(q,q0)^2/(2*s^2))/Z(s)` relative to Haar measure, where `d` is
the SO(3) geodesic angle and `Z` is computed numerically.  A fiber maps its
specified crystal direction near the sample direction with the same finite
angular Gaussian and has uniform spin around the fiber.  Mixtures combine
normalized component densities by their weights.

`PartialFiberODF` (also accepted as `sputter_fiber` or `offset_fiber` in JSON)
extends a fiber with a tilted sample direction and controlled spin about the
fiber. `spin_width_deg=180` gives a half-ring, while `spin_kappa>0` gives a
smooth von-Mises lopsided ring. Spin modulation is normalized to preserve the
ODF's Haar-measure normalization. Every supported ODF has `to_dict()` output,
and object-supplied ODF parameters are retained in simulation metadata.

Crystal symmetry is explicit and never silently baked into sampling.  Utilities
provide the 24 proper cubic and 12 proper hexagonal operations,
symmetry-equivalent orientations, and deterministic representative reduction.
This avoids treating symmetry equivalents as separate texture components during
comparison/visualization while retaining each simulated grain orientation.

Pole figures rotate the supplied crystal direction into sample coordinates,
fold to the upper hemisphere, and use an equal-area projection.  The empirical
ODF is a volume-weighted (when grain `volume` is available) SO(3) kernel density
estimate; otherwise grains have equal weight.

## Input examples

```json
{"odf": {"type": "fiber", "crystal_direction": [1,1,1], "sample_direction": [0,0,1], "spread_deg": 10}}
```

```python
from src.classes_and_functions.texture import OrientationComponent, FiberODF, MixtureODF, RandomODF
random = RandomODF()
single = OrientationComponent([1, 0, 0, 0], spread_deg=3)
fiber = FiberODF([1, 1, 1], [0, 0, 1], spread_deg=8)
two_component = MixtureODF([(0.6, single), (0.4, OrientationComponent([.70710678, 0, .70710678, 0], 8))])
```

Run `python examples/texture_workflow.py --grains 100 --seed 12345` to generate
target/empirical ODF sections, pole figures, and diffraction images for all four
textures.  Increase `--grains` (for example 100, 1000, 10000) for convergence.
