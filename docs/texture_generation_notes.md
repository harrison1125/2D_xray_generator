# Texture generation notes

This document consolidates the current notes on the texture generation engine,
the intended dataset strategy, and the scientific constraints that matter for
Bayesian inference on XRD scans.

## Goal

The immediate objective is to generate a large, dense, continuous, and
mathematically sound ODF dataset that can be passed through the XRD simulator
to create a training corpus for Bayesian texture inference.

The intended inference workflow is:

1. Train on a broad synthetic XRD dataset spanning many texture families.
2. Demonstrate that a single XRD scan is underdetermined without prior
   information.
3. Inject an EBSD-derived or ground-truth prior from a similar sample region
   and show that the posterior contracts onto a much smaller set of plausible
   ODFs.

The key requirement is that the simulation space must be extensible and not
hard-coded around one texture type such as a simple fiber texture.

## What is in place

The codebase now contains support for both the older named texture objects and
the newer continuous ODF sampling path.

### Existing named texture machinery

The existing texture system includes:

- Haar-uniform random orientations.
- Sharp single-component ODFs.
- Fiber textures.
- Mixtures of components.
- Empirical ODF estimation.
- Partial or offset fibers for sputter-like growth patterns.

The partial-fiber path is important because it captures:

- Tilted growth axes.
- Half-rings through limited spin support.
- Smooth lopsided rings through a spin-bias term.
- Competing growth directions.

These are useful for sputter-deposition style behavior and for patterns that
look close to fibers but are not perfectly axisymmetric.

### Continuous ODF sampling path

A separate continuous ODF dataset generator has been added to support a broad,
continuous parameter space instead of a discrete family classifier.

The continuous path is designed to:

- Sample ODFs on `SO(3)` with Haar-consistent parameterization.
- Use low-discrepancy Sobol sampling rather than pseudo-random sampling.
- Support high- and low-symmetry crystal systems.
- Produce non-negative, normalized ODFs by construction.
- Preserve crystal symmetry explicitly.
- Emit both simulation-ready XRD inputs and ODF-level metadata.

## Mathematical model

The intended ODF model is a continuous mixture over `SO(3)`:

```text
f(g) = w0 + sum_i wi * psi(g, gi, sigma_i)
```

where:

- `w0` is the isotropic background.
- `gi` are component centers in `SO(3)`.
- `sigma_i` or FWHM controls angular spread.
- `psi` is a normalized kernel on `SO(3)`.
- `sum_i wi + w0 = 1`.

This ensures:

- `f(g) >= 0`.
- `int_{SO(3)} f(g) dg = 1`.
- The ODF can vary continuously between sharp, broad, and mixed states.

### Kernel choice

The continuous implementation uses a de la Vallee Poussin kernel on `SO(3)`.
Its angular width is parameterized by full width at half maximum in degrees.

That choice matters because it makes the spread parameter physically
interpretable and continuous over the sharp-to-broad range:

- Roughly `2.5 deg` for very sharp textures.
- Up to `30 deg` for broad diffuse textures.

### Orientation parameterization

To avoid distortions from Euler-angle singularities, the design space is
sampled on `SO(3)` using unit quaternions derived from a Sobol sequence.

The correct Haar measure is respected by sampling through a measure-preserving
map into unit quaternions rather than using naive uniform Euler-angle grids.

## Symmetry handling

Crystal symmetry is handled explicitly rather than assumed implicitly.

The current design supports at least:

- Cubic `m-3m` as a high-symmetry case.
- Monoclinic `2/m` as a low-symmetry case.
- Triclinic `-1` as the minimal-symmetry bound when needed.

Only the proper rotational subgroup acts inside `SO(3)`. Improper symmetry
operations such as inversion or mirror reflections are not elements of
`SO(3)` and should not be treated as orientation operators in the ODF
construction.

This is important because the goal is not to create separate labels for every
symmetry-equivalent orientation. The goal is to build a continuous texture
space that remains physically meaningful under the crystal symmetry action.

## Sampling strategy

The parameter space should be continuous, dense, and evenly distributed.

The current sampling logic is built around:

- Scrambled Sobol sequences.
- Quasi-random coverage of the hypercube.
- Haar-uniform orientation sampling through quaternions.
- Continuous component count through fixed component slots with soft weights.
- Continuously varying isotropic background fraction.

The intent is to avoid a brittle texture taxonomy with hard family boundaries.
Instead, the dataset should contain overlapping regions of parameter space so
that the inverse problem is genuinely ambiguous when priors are absent.

### Why the overlap matters

This is the core identifiability point.

If the dataset is too categorical, the inference model will learn to separate
named families rather than infer the latent orientation state.

A better synthetic corpus contains:

- Random/isotropic limits.
- Localized SO(3) components.
- Sharp and broad fibers.
- Multimodal mixtures.
- Tilted sputter fibers.
- Half-rings and arcs.
- Smooth lopsided rings.
- Competing growth directions.

These families should overlap enough that two distinct ODFs can generate very
similar single-shot diffraction patterns.

That is not a bug. It is the desired underdetermination that motivates the
Bayesian model.

## Representation question

Separating textures into named families is useful for bookkeeping, but it can
become misleading if it is treated as the true geometry of the problem.

The safer framing is:

- Texture families are convenient regions or motifs in a continuous latent
  space.
- The real target is a differentiable, continuous ODF manifold or parameter
  space.
- Family labels should not become hard class boundaries in the inverse model.

In other words, the simulator should allow both:

- Smooth interpolation between motifs.
- Multiple motifs producing similar diffraction patterns.

That preserves the intuition that a scan may belong to multiple plausible
regions of latent space before priors are applied.

## Dataset design

The current dataset plan is to generate a large synthetic corpus across many
texture families and symmetry settings.

Recommended run structure:

- A small proof-of-concept dataset first.
- A production phase-space dataset with many samples per symmetry class.
- Separate held-out regions for calibration and validation.
- Separate held-out symmetry systems if generalization is being tested.

The production corpus should include at least:

- Cubic symmetry.
- Monoclinic symmetry.

Triclinic can be added as an extreme low-symmetry test if needed.

## Output and metadata

Each simulated sample should retain:

- The exact ODF parameters.
- The Sobol design coordinates.
- The component centers.
- The spreads/FWHMs.
- The weights and isotropic background.
- The effective component count.
- The full simulation configuration.

That metadata is the real ground truth. The XRD image is the observation.

If harmonic coefficients are written out, they should be treated as a compact
representation or diagnostic representation, not as a substitute for the
analytic ODF parameters when the kernels are sharp.

Similarly, any common ODF grid is a practical approximation, not an exact
representation of a sharp 2.5 degree kernel.

## Scientific validity notes

The dataset must be documented carefully.

Important caveats:

- The low-symmetry path is useful for testing identifiability and symmetry
  logic, but it is not automatically a calibrated materials model.
- Structure factor, atomic basis, form factors, absorption, background, and
  detector calibration all matter for quantitative claims.
- Reciprocal-lattice truncation must be checked against the accessible
  q-range.
- Very sharp textures may not be faithfully reconstructed from a coarse global
  ODF grid or a low harmonic bandlimit.

The readme and run notes should be explicit that this is a comprehensive
synthetic texture database for inference research, not a finished physical
model of every material system.

## Why this helps the Bayesian model

This setup is designed to support two experiments:

1. No-prior inference on a single XRD scan.
2. Prior-conditioned inference with an EBSD prior from a nearby sample
   location.

The first experiment should expose multi-modality and degeneracy.
The second should show that a meaningful prior collapses the posterior onto a
smaller region of ODF space.

That is the right behavior for an underdetermined inverse problem.

## Practical guidance

When running the dataset generation:

- Start with a small proof corpus.
- Verify the distribution of ODF spreads and weights.
- Check that symmetry handling behaves correctly.
- Confirm that distinct ODFs do in fact produce similar patterns where
  expected.
- Scale to the production dataset only after those checks pass.

The most important quality criterion is not just size. It is whether the
sampled phase space is continuous, dense, overlapping, and mathematically
consistent enough to support later Bayesian inference.
