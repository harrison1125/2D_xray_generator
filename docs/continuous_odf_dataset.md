# Continuous Sobol ODF dataset

`generate_continuous_odf_dataset.py` generates a continuous latent ODF design
for simulation-based Bayesian inference. It is deliberately separate from the
small set of named texture examples: a sample is a point in one shared mixture
space, not a one-hot `fiber`/`component` class.

Run the small validation corpus first:

```bash
python src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_proof.json --dry-run
python src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_proof.json
```

The production specification requests 50,000 ODFs for each of cubic and
monoclinic symmetry (100,000 scans total):

```bash
python src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_phase_space_50k.json --limit 100
```

Use `--limit` to stage a long run. Completion is recorded after each sample in
`manifest.jsonl`; rerunning resumes completed IDs. The driver refuses to
overwrite a non-empty run directory without a corresponding completed manifest
record, which protects interrupted datasets from silent corruption.

For independent HPC jobs, split only the sample indices—not the Sobol design:

```bash
python src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_phase_space_50k.json --num-shards 8 --shard-index 0
```

Run indices are assigned by `sample_index % num_shards`, so every shard sees a
disjoint subset of the same precomputed Sobol design. Each shard writes its own
manifest file, avoiding concurrent append races; output sample directories are
already disjoint.

## Mathematical model

All ODFs are densities with respect to normalized Haar measure on `SO(3)`:

```text
f(q) = w0 + sum_i wi / |H| sum_(h in H) K_si(d(q, qi h))
```

`w0 >= 0`, `wi >= 0`, and `w0 + sum_i wi = 1`. Thus non-negativity and
normalization hold by construction. `H` is the proper rotational subgroup of
the crystal point group. For cubic symmetry it has 24 operations; for
monoclinic `2/m`, it has identity and the proper two-fold rotation. Inversion
and mirror operations are improper and are not elements of `SO(3)`, so they do
not appear in the orientation action. Consequently `f(q h) = f(q)`.

The kernel is de la Vallée Poussin:

```text
K_s(theta) = C_s cos(theta/2)^(2s)
C_s = sqrt(pi) Gamma(s + 2) / Gamma(s + 1/2)
```

where `theta` is the SO(3) geodesic angle. `component_fwhm_deg` is explicitly
the *full* width at half maximum, with

```text
s = log(1/2) / [2 log(cos(FWHM/4))].
```

This is not the Gaussian `sigma` used by the older named ODF classes. The
range `[2.5, 30]` degrees in the supplied configuration therefore has an
unambiguous physical meaning.

## Sobol design

For five component slots the Sobol hypercube has 27 dimensions:

```text
background fraction (1)
Dirichlet concentration (1)
five × [Haar-uniform SO(3) center (3), FWHM (1), weight coordinate (1)]
```

Centers use the measure-preserving Shoemake map from three unit-cube variables
to unit quaternions. Component weights use inverse-Gamma transforms followed
by normalization, giving an exact Dirichlet distribution conditional on the
continuously sampled concentration. This gives a continuous effective number
of components between roughly one and five without adding a categorical family
boundary. A symmetric group average applies crystal symmetry after sampling.

The implementation uses scrambled Sobol points only. For non-power-of-two
sizes such as 50,000 it takes a prefix of a 65,536-point Sobol net; it does not
fill the remainder with pseudo-random samples. Scrambling permits deterministic
independent designs for cubic and monoclinic systems while retaining low
discrepancy.

## Outputs and ground truth

Each sample directory contains a simulated image, resolved metadata, and
`odf_representation.npz`. The NPZ includes the exact ODF parameters,
the originating Sobol point, background fraction, component centers, FWHMs,
weights, and effective component count. These parameters are the primary
ground truth.

When enabled, every crystal-system directory also contains a shared
`odf_grid_quaternions.npy`; the per-sample NPZ contains `odf_grid_values` on
that grid. It is a diagnostic/common-grid representation, not an exact
representation of a 2.5-degree kernel: a practical global SO(3) grid will not
resolve every very sharp peak. Use the analytic parameters or high-bandwidth
harmonics for reconstruction rather than treating the 1024-point grid as the
truth.

The harmonic arrays use the standard complex Wigner-D basis with
`F_l = integral f(g) D_l(g)* dg`. They are analytically calculated from the
DVP mixture rather than estimated by Monte Carlo; `F_0 = [[1]]` verifies
normalization. They are nevertheless *truncated* at `harmonic_bandlimit`.
A bandlimit of 6 is compact metadata and cannot faithfully reconstruct a
2.5-degree feature. Raise the bandlimit substantially for a reconstruction
target, with corresponding CPU and storage cost.

## Interpretation and validity

This database intentionally contains overlapping ODFs. Different continuous
ODFs may make indistinguishable single-shot XRD patterns due to symmetry,
limited detector acceptance, finite grain realization, and omitted nuisance
physics. That degeneracy is the likelihood-only experiment; do not convert the
continuous labels into a texture-family classifier. A correct no-prior model
should retain posterior mass on several ODF regions. An EBSD-derived prior can
then be introduced as a prior over the same continuous ODF parameters or a
compatible ODF representation.

The low-symmetry path uses a real monoclinic reciprocal lattice and a primitive
structure-factor placeholder. It is suitable for testing symmetry/identifiability
logic, but it is **not** a calibrated material model: replace the unit cell,
atomic basis, form factors, absorption, background, detector calibration, and
noise model before making quantitative materials claims. `max_hkl_index` also
truncates the reciprocal lattice for computational control, so it must be
increased and validated against the accessible q-range for production work.

At the provided 512×512 float32 resolution, 100,000 uncompressed images alone
are about 100 GB, before metadata. Start with `--limit`, benchmark throughput,
and decide whether to retain grain and peak records (the production JSON turns
both off to control file volume). Keep held-out Sobol regions, held-out symmetry
systems, and real scans separate from training data when evaluating posterior
calibration.
