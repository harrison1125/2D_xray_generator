# Physical texture manifolds for continuous ODF generation

## Scope and design decision

This document specifies the physical texture sampler used by
`generate_continuous_odf_dataset.py`, its exact relationship to the versioned
ODF target, and the consequences for Bayesian inference. It is written
in standard Markdown with display equations so it can be read directly on
GitHub, in VS Code, Obsidian, Typora, or imported into Notion.

The implementation deliberately does **not** make a fiber-only manifold and
$1.1 \le J \le 35$ a universal hard prior. That proposal would be appropriate
for a narrowly defined process prior, but it is unsafe for learning a broadly
valid likelihood:

- a hard fiber catalog assigns zero probability to non-standard, tilted, or
  mixed textures even when an experiment genuinely contains one;
- the engineering interval $1.1 \le J \le 35$ is not a law of nature—nearly
  random materials can have $J < 1.1$, while epitaxial films and directionally
  solidified material can have $J > 35$;
- hard rejection changes a Sobol net into a nonuniform accepted subset and can
  make neural posterior estimators collapse onto the most common named mode;
- canonical quaternion representatives are discontinuous at fundamental-zone
  and mixture-ordering boundaries. Tightening the sampler does not remove that
  target-topology problem;
- an ODF component FWHM is an **orientation/misorientation spread**, not a
  diffraction-peak width. Dislocation microstrain, coherent domain size,
  instrumental resolution, and grain mosaic blocks belong in the scattering
  nuisance model. Treating all of those effects as ODF FWHM would confound
  texture with line broadening.

The compromise is two explicit operating modes that produce the same analytic
ODF construction. The descriptor is versioned: legacy five-slot designs use
31 float values, while the maintained six-slot PVD design uses 37.

1. **Mode A — likelihood / broad coverage.** Named manifolds improve sampling
   efficiency, while a nonzero Haar-uniform center channel preserves support
   away from them. The texture-index interval is a soft, floored log weight,
   not a rejection boundary. Broader background, concentration, and FWHM
   ranges are used.
2. **Mode B — physical prior / process-specific.** Rolling,
   recrystallization, or additive-manufacturing probabilities concentrate the
   catalog. Sparse Dirichlet weights, a small uniform escape fraction, and an
   optional hard $J$ filter define a deliberately informative simulation
   prior for amortized posterior estimation.

The retained legacy mode remains available so an existing corpus can be
reproduced and used as an OOD control. It should not be the default design for
a new production corpus.

### PVD / sputter thin-film likelihood manifold

`configs/continuous_odf_sputter_pvd_broad.json` is a separate Stage-1
likelihood proposal. It does not reuse rolling, recrystallization, or additive
bulk-processing probabilities. Each active component has one of these
provenance tags:

| Tag | Unconditional probability | Construction |
|---|---:|---|
| `sputter_fiber` | 0.65 | Film-normal fiber center, with its crystal axis aligned to sample ND |
| `sputter_biaxial` | 0.20 | Narrow plane-plus-in-plane-direction component for epitaxial or seed-layer growth |
| `nanocrystalline_uniform` | 0.15 | Haar-uniform center escape |

The 0.15 uniform probability is selected first. Conditional on using the PVD
catalog, fiber and biaxial probabilities are respectively \(13/17\) and
\(4/17\), so the unconditional probabilities are exactly 0.65 and 0.20.
The FCC catalog contains \(\langle111\rangle\parallel\mathrm{ND}\) and
\(\langle100\rangle\parallel\mathrm{ND}\) fibers; BCC contains
\(\langle110\rangle\parallel\mathrm{ND}\); HCP contains basal c-axis and
prismatic a-axis film-normal fibers. Biaxial components use both a plane normal
mapped to ND and an in-plane direction mapped to RD.

All ordinary PVD component FWHMs are sampled from 5--30 degrees. Biaxial
components use 5--10 degrees, so strongly aligned cases occur naturally, but
there is no hard texture-index filter: \(J\) is calculated and recorded for
every ODF. The corpus uses 1--6 active components with a Dirichlet
\(\alpha=1\) distribution over their non-background weight. It writes a
37-float, six-slot v2 descriptor

$$
[w_0,\;w_1,\ldots,w_6,\;q_1,\ldots,q_6,\;\Delta_1,\ldots,\Delta_6],
$$

plus an `active_component_count` and `component_active_mask`. Zero-weight
slots are not part of the ODF. Existing five-slot corpora retain their
31-float v1 descriptor unchanged.

The HDF5 `metadata.h5` file is a consolidated view of exact ODF labels and the
planned scan index; detector images remain NPY/TIFF files and are referenced by
path. JSONL remains the authoritative per-run status log.

> **Fiber-model limitation.** A `sputter_fiber` entry currently draws a local
> DVP component whose center lies on a film-normal fiber. A finite mixture of
> these local components is a practical fiber-texture approximation, not an
> exactly axially invariant fiber ODF. If strict cylindrical invariance is
> required for a particular scientific claim, use an axisymmetric fiber kernel
> or tile the fiber azimuth densely; 1--6 local components cannot prove exact
> rotational invariance about ND.

---

## Why the legacy 27-dimensional cube is inefficient

The legacy source coordinate is

$$
u = \left(u_b,u_\alpha,
  \{u_{i,q_1},u_{i,q_2},u_{i,q_3},u_{i,\Delta},u_{i,w}\}_{i=1}^{5}\right)
  \in [0,1]^{27}.
$$

It maps every center independently through the Shoemake Haar map, maps every
width independently, and obtains mixture weights from five inverse-Gamma
quantiles. This construction is mathematically valid, normalized, and useful
as a broad control distribution. It is not a physically neutral description
of deformed polycrystals.

With $N=10^5$ points in $d=27$, even a very loose characteristic spacing
is $N^{-1/d}\approx0.65$ of the unit-cube side. The design is therefore not a
dense scan of a 27-dimensional space. Most five-center combinations do not
represent a deformation path, recrystallization mechanism, or directional
solidification history.

Independent sharp components also create likelihoods with narrow separated
wells. If a pattern discrepancy is

$$
\mathcal L(\theta) \propto
\exp\!\left[-\frac12
(I_{\mathrm{exp}}-I_{\mathrm{sim}}(\theta))^\mathsf T
\Sigma^{-1}
(I_{\mathrm{exp}}-I_{\mathrm{sim}}(\theta))\right],
$$

small changes to a sharp center may move intensity between detector arcs. The
result is a steep canyon near a matching orientation and a nearly flat region
where the relevant reflection leaves the detector. Multiple components
multiply the number of such basins.

Finally, a detector image is not an injective observation of an ODF on
$SO(3)$. Crystal symmetry, Friedel-pair intensity equivalence, overlapping
$hkl$ rings, finite detector acceptance, nuisance broadening, and a finite
grain realization all induce aliases. Sampling arbitrary multimodal ODFs
creates many additional aliases without matching the frequency with which
those states arise in real processing.

---

## Orientation and symmetry convention

All code and saved labels use unit quaternions in scalar-first order,

$$
q=(w,x,y,z), \qquad \lVert q\rVert_2=1, \qquad q\equiv -q.
$$

They are right-handed **active crystal-to-sample** rotations acting on column
vectors:

$$
\mathbf v_s = R(q)\mathbf v_c.
$$

The sample basis is

$$
\mathrm{RD}=\mathbf e_x,\qquad
\mathrm{TD}=\mathbf e_y,\qquad
\mathrm{ND}=\mathbf e_z.
$$

Crystal symmetry acts on the right. If $H\subset SO(3)$ is the proper
rotational point group, $q$ and $qh$, $h\in H$, describe equivalent
crystal orientations. Cubic symmetry uses 24 proper rotations, hexagonal
symmetry uses 12, and the repository's monoclinic convention uses the proper
$C_2$ operation about its Cartesian crystal-frame z axis. Mirrors and inversion are not rotations
in $SO(3)$ and are not inserted into $H$.

> **Monoclinic convention warning.** Many crystallographic packages use the
> unique $b$-axis setting for monoclinic crystals. This repository currently
> implements the proper two-fold operation about Cartesian z and describes it
> in code as the c axis. For a non-orthogonal cell, the direct-lattice c vector
> constructed by the simulator need not coincide with z. Material data and
> external Euler angles must therefore be converted and this convention
> validated, or the symmetry definition must be changed consistently before
> generation. The physical sampler follows the existing Cartesian convention
> rather than silently imposing another one.

For a conventional rolling component $\{hkl\}\langle uvw\rangle$, define the
orthonormal crystal triad

$$
C=\begin{bmatrix}
\widehat{[uvw]} &
\widehat{[hkl]}\times\widehat{[uvw]} &
\widehat{[hkl]}
\end{bmatrix}.
$$

The active orientation is

$$
R=C^\mathsf T,
$$

so that $R\widehat{[uvw]}=\mathrm{RD}$ and
$R\widehat{[hkl]}=\mathrm{ND}$. The implementation rejects a component if
$[uvw]\cdot[hkl]\ne0$.

The default cubic catalog contains the following explicit representatives:

| Name | Plane/direction or constraint | Typical role |
|---|---:|---|
| Copper | $\{112\}\langle 11\bar1\rangle$ | FCC rolling, beta fiber |
| S | $\{123\}\langle 63\bar4\rangle$ | FCC rolling, beta fiber |
| Brass | $\{110\}\langle 1\bar12\rangle$ | FCC rolling, beta fiber |
| Goss | $\{110\}\langle001\rangle$ | rolling/recrystallization |
| Cube | $\{001\}\langle100\rangle$ | recrystallization |
| Alpha fiber | $\langle110\rangle\parallel\mathrm{RD}$ | rolling fiber |
| Gamma fiber | $\langle111\rangle\parallel\mathrm{ND}$ | sheet texture |
| Build fiber | $\langle001\rangle\parallel\mathrm{ND}$ | cube/build-axis texture |
| Beta fiber | Brass $\rightarrow$ S $\rightarrow$ Copper | piecewise-geodesic rolling manifold |

Conditional on choosing the physical rather than Haar-escape channel, the
cubic category probabilities are:

| Category | Broad | Rolling | Recrystallization | Additive manufacturing |
|---|---:|---:|---:|---:|
| Copper | 0.1111 | 0.15 | 0.06 | 0.03 |
| S | 0.1111 | 0.14 | 0.04 | 0.02 |
| Brass | 0.1111 | 0.13 | 0.04 | 0.02 |
| Goss | 0.1111 | 0.06 | 0.23 | 0.05 |
| Cube | 0.1111 | 0.05 | 0.40 | 0.14 |
| Alpha fiber | 0.1111 | 0.20 | 0.04 | 0.03 |
| Gamma fiber | 0.1111 | 0.07 | 0.12 | 0.22 |
| Build fiber | 0.1111 | 0.03 | 0.05 | 0.44 |
| Beta fiber | 0.1111 | 0.17 | 0.02 | 0.05 |

Catalog probabilities are templates, not universal metallurgical constants.
They should be calibrated by alloy, phase, strain path, temperature, and
recrystallized fraction before Mode B results are interpreted quantitatively.

Hexagonal templates align $\langle0001\rangle$ with ND or RD and a prismatic
axis with RD. Generic lower-symmetry templates align the simulator's Cartesian
crystal-frame axes with RD, TD, or ND. Those relations make processing axes explicit but do not claim
to represent a calibrated monoclinic material prior.

The hexagonal category probabilities are 0.50 for the basal axis along ND,
0.20 for the basal axis along RD, and 0.30 for a prismatic axis along RD.
The generic lower-symmetry probabilities are 0.35 for crystal-frame z along
ND, 0.25 for y along TD, 0.25 for x along RD, and 0.15 for the aligned
Cartesian-frame component. These axes follow the simulator's orientation
frame. For a non-orthogonal monoclinic cell they must not be interpreted as
metric-correct direct-lattice directions without an explicit cell-basis
conversion.

---

## Analytic ODF

### Symmetric DVP mixture

The saved target always represents

$$
f(q)=w_0+\sum_{i=1}^{M} w_i S_i(q),
\qquad
w_0+\sum_{i=1}^{M}w_i=1,
\qquad M=5,
$$

with symmetry-averaged components

$$
S_i(q)=\frac1{|H|}\sum_{h\in H}
K_{s_i}\!\left(d(q,q_i h)\right).
$$

All densities are relative to normalized Haar measure $d\mu(q)$, for which
$\int_{SO(3)}d\mu=1$. The background density is therefore exactly one and its
mixture coefficient $w_0$ is the untextured grain fraction.

The de la Vallée Poussin kernel is

$$
K_s(\theta)=C_s\cos^{2s}\!\left(\frac\theta2\right),
\qquad
C_s=\sqrt\pi\frac{\Gamma(s+2)}{\Gamma(s+\tfrac12)}.
$$

It integrates to one. For a requested full width at half maximum
$\Delta\in(0,180^\circ)$,

$$
s(\Delta)=
\frac{\log(1/2)}
{2\log\!\left[\cos(\Delta/4)\right]}.
$$

The half-maximum lies at geodesic distance $\Delta/2$. This definition is not
a Gaussian standard deviation and is not a detector peak width.

### Point-component perturbations

A named component center $q_\star$ is perturbed with the same normalized DVP
law. Under $K_s\,d\mu$,

$$
T=\cos^2(\theta/2)\sim
\operatorname{Beta}\!\left(s+\tfrac12,\tfrac32\right).
$$

One remapped Sobol coordinate is sent through the inverse Beta CDF for $T$;
two coordinates select a uniform axis on $S^2$. The center is

$$
q=q_\star\otimes q_{\mathrm{axis}}(\theta).
$$

The default component-center dispersion FWHM is $14^\circ$ in Mode A and
$10^\circ$ in Mode B.

### Fiber projections

For a crystal direction $\mathbf c$ and processing direction $\mathbf a$,
the ideal fiber is

$$
\mathcal F(\mathbf c,\mathbf a)
=\{q\in SO(3):R(q)\widehat{\mathbf c}=\widehat{\mathbf a}\}.
$$

An ideal center on this one-dimensional manifold is obtained from a minimal
alignment rotation $q_{c\rightarrow a}$ and a spin $\psi$:

$$
q_{\mathcal F}(\psi)=q_{\mathbf a}(\psi)\otimes q_{c\rightarrow a},
\qquad \psi\sim\operatorname{Uniform}(0,2\pi).
$$

To create a smooth tube, the target direction is drawn from a von Mises–Fisher
law on $S^2$:

$$
p(\mathbf t\mid\mathbf a,\kappa)
=\frac{\kappa}{4\pi\sinh\kappa}
\exp(\kappa\mathbf a^\mathsf T\mathbf t).
$$

Writing $x=\cos\beta=\mathbf a^\mathsf T\mathbf t$, its inverse CDF is
evaluated stably as

$$
x=1+\frac1\kappa
\log\!\left[u+(1-u)e^{-2\kappa}\right].
$$

The azimuth around $\mathbf a$ and spin $\psi$ use the other two remapped
coordinates. Mode A uses $\kappa=45$; Mode B uses $\kappa=70$.

The cubic beta fiber is represented as two quaternion geodesic segments,

$$
q_\beta(t)=
\begin{cases}
\operatorname{slerp}(q_{\mathrm{Brass}},q_S,2t),&0\le t<1/2,\\
\operatorname{slerp}(q_S,q_{\mathrm{Copper}},2t-1),&1/2\le t\le1.
\end{cases}
$$

A truncated radial Gaussian perturbation is applied in the two-dimensional
Lie-algebra plane normal to the local path tangent. This is a computationally
stable tube around the canonical Brass–S–Copper path; it is not a claim that a
specific alloy has uniform density along that path.

---

## Exact texture index

The texture index is

$$
J=\int_{SO(3)} f(q)^2\,d\mu(q).
$$

Because $f$ is a normalized density, Cauchy–Schwarz gives $J\ge1$, with
$J=1$ only for the uniform ODF.

Let $q_a,q_b$ be two DVP centers,
$\rho=|q_a^\mathsf Tq_b|$, and let their exponents be $s,t$. Their exact
pair overlap is

$$
\begin{aligned}
A(s,t,\rho)
&=\int_{SO(3)}K_s(d(q,q_a))K_t(d(q,q_b))\,d\mu(q)\\
&=\frac{\Gamma(s+2)\Gamma(t+2)}{\Gamma(s+t+2)}
{}_2F_1\!\left(-s,-t;\tfrac12;\rho^2\right).
\end{aligned}
$$

This follows by lifting Haar-uniform rotations to antipodally identified
uniform quaternions on $S^3$ and using the absolute joint moments of two
correlated Gaussian projections. For exactly coincident centers,

$$
A(s,t,1)=\frac{C_sC_t}{C_{s+t}}.
$$

Haar invariance reduces the double symmetry sum to one group sum:

$$
\int S_i(q)S_j(q)\,d\mu(q)
=\frac1{|H|}\sum_{h\in H}
A\!\left(s_i,s_j,|q_i^\mathsf T(q_jh)|\right).
$$

Therefore the implemented analytic index is

$$
\boxed{
J=2w_0-w_0^2+
\sum_{i=1}^M\sum_{j=1}^M w_iw_j
\frac1{|H|}\sum_{h\in H}
A\!\left(s_i,s_j,|q_i^\mathsf T(q_jh)|\right)
}.
$$

The code evaluates the hypergeometric expression in log space. For very sharp,
nearly coincident kernels, an unscaled hypergeometric function can overflow
even though the final product is finite. A connection formula about
$\rho^2=1$, followed by the equivalent one-dimensional $S^3$ angular integral,
provides a stable fallback. Terms whose rigorous maximum-product bound is below
$10^{-14}$ are omitted. This avoids the failure mode of a coarse Monte Carlo
grid, which can miss a sharp component entirely.

### Hard and soft policies

Mode B may use literal rejection:

$$
\mathcal A=\{f:1.1\le J(f)\le35\}.
$$

The saved `design_candidate_index` records which point in the original
scrambled Sobol prefix survived. Rejection preserves determinism but not the
original net balance.

Mode A uses a soft log weight,

$$
\log \omega_J=
\log\sigma\!\left(\frac{J-J_{\min}}{\tau}\right)
+\log\sigma\!\left(\frac{J_{\max}-J}{\tau}\right),
$$

with $\tau=2$ by default. It is floored at $-10$, so no point inside the
base generator has zero weight. The design itself is retained; downstream
code may use `texture_index_log_weight` for weighted risk, resampling, or a
change-of-measure correction. It must not accidentally treat this auxiliary
weight as part of the physical likelihood.

---

## Mapping process coordinates to the versioned target

The legacy five-slot sampler uses 27 Sobol coordinates. The maintained PVD
sampler uses 33: background, Dirichlet concentration, six blocks of center,
width, and weight coordinates, plus an active-component-count coordinate.
They do not represent independent physical degrees of freedom. Each
three-number center block is a compact mixture transform:

1. The first coordinate selects the Haar escape channel or a catalog manifold.
2. Within the selected interval, the residual coordinate is rescaled to
   $[0,1]$. For a point component it is a DVP radial quantile; for a fiber it
   is spin; for the beta fiber it is path position.
3. The remaining two coordinates supply the angular perturbation variables.

For the untextured fraction,

$$
z_0=F^{-1}_{\operatorname{Beta}(a_b,b_b)}(u_b),\qquad
w_0=b_{\min}+(b_{\max}-b_{\min})z_0.
$$

For the sparse component weights, first map

$$
\alpha=\exp\!\left[
\log\alpha_{\min}+u_\alpha
\log\!\left(\frac{\alpha_{\max}}{\alpha_{\min}}\right)
\right],
$$

then use inverse-Gamma quantiles

$$
r_i=F^{-1}_{\operatorname{Gamma}(\alpha,1)}(u_{i,w}),
\qquad
w_i=(1-w_0)\frac{r_i}{\sum_jr_j}.
$$

This is an exact symmetric Dirichlet draw conditional on $\alpha$. Values
$\alpha<1$ favor corners of the simplex, so one to three components tend to
dominate without setting the other slots identically to zero. The maintained
PVD configuration fixes $\alpha=1.0$ to retain both sparse-looking and complex
mixtures; earlier experimental modes may configure an interval.

Each width coordinate first selects one of three FWHM strata and then uses log
interpolation inside that stratum:

$$
\Delta_i=\exp\!\left[
\log\Delta_{k,\min}+v_i
\log\!\left(\frac{\Delta_{k,\max}}{\Delta_{k,\min}}\right)
\right].
$$

The resulting components are reduced to the repository's deterministic
crystal-symmetry representative and sorted by descending weight, breaking ties
by ascending FWHM. For $M$ allocated slots, the float32 target is

$$
\texttt{odf\_target}=
\left[
w_0,
w_1,\ldots,w_M,
q_{1,w},q_{1,x},q_{1,y},q_{1,z},\ldots,q_{M,z},
\Delta_1,\ldots,\Delta_M
\right]\in\mathbb R^{1+6M}.
$$

| Half-open target offsets | Length | Meaning |
|---:|---:|---|
| `[0, 1)` | 1 | isotropic background weight $w_0$ |
| `[1, 1+M)` | $M$ | component weights $w_i$ |
| `[1+M, 1+5M)` | $4M$ | canonical `wxyz` quaternions |
| `[1+5M, 1+6M)` | $M$ | full DVP FWHMs in degrees |

The process family, component-manifold names, active mask, $J$, soft log
weight, Dirichlet concentration, and candidate index are saved as auxiliary
arrays. They are **not** appended to the target. Descriptor v1 has $M=5$ and
31 values; the maintained PVD descriptor v2 has $M=6$ and 37 values, so a
loader must read `odf_label_schema.json` rather than assume v1 offsets.

---

## Parameter-space specification

| Quantity | Legacy 27D Sobol control | Maintained sputter/PVD broad-likelihood proposal | Scientific reason |
|---|---|---|---|
| Source coordinates | 27; five fixed slots | 33; six allocated slots plus active-count coordinate | Fixed, scrambled Sobol designs remain reproducible while PVD supports 1--6 components |
| Crystal coverage | Haar centers under declared crystal symmetry | FCC Cu, BCC Fe, and HCP Ti PVD catalogs | Represents the stated thin-film scope without pretending all bulk process histories are PVD |
| Center law | Independent Haar $SO(3)$ | 65% film-normal fiber, 20% biaxial growth, 15% Haar escape per active component | PVD growth preference while retaining a global-support route |
| Film-normal families | none | FCC $\langle111\rangle$ primary / $\langle100\rangle$ secondary; BCC $\langle110\rangle$; HCP basal/prismatic | Surface-energy and growth-direction texture templates |
| Component count | 5 fixed nonzero slots | Discrete 1--6 active slots | Covers simple films and complex multi-peak films without encoding zero-weight components as physics |
| Component weights | Configurable Dirichlet | Dirichlet $\alpha=1.0$ | Permits dominant and comparable mixture weights without forcing one texture class |
| DVP FWHM | Legacy configuration range | 5--30 degrees; biaxial subset 5--10 degrees | Covers biased through broad sputtered growth and includes sharp seed-layer states |
| Background $w_0$ | Legacy configuration range | Beta(1,1) scaled to 0--0.96 | Preserves near-isotropic/nanocrystalline states alongside textured films |
| Texture index $J$ | Not necessarily stored | Analytic $J$ recorded; soft policy with effectively unbounded limits | No $J$ truncation: near-isotropic and sharp/epitaxial tails remain in the corpus |
| Descriptor | 31 floats for five slots | 37 floats for six slots, plus masks/tags in metadata | Avoids artificial fixed-five-component assumptions in downstream inference |
| Intended use | OOD/control comparison | Stage-1 likelihood training and candidate-solution discovery | Separates likelihood coverage from an experiment-derived EBSD prior |

The PVD values are engineering proposal bounds, not material-certified
distributions. Refine them from independent film measurements and retain the
uniform channel whenever extrapolation matters.

---

## Configuration and execution

The maintained production proposal is
`configs/continuous_odf_sputter_pvd_broad.json`. It is explicitly a PVD
broad-likelihood distribution, not a universal texture prior or a bulk rolling
model. Validate the plan without generating data, then stage the run:

```bash
uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --dry-run

uv run src/driver_codes/generate_continuous_odf_dataset.py \
  configs/continuous_odf_sputter_pvd_broad.json --limit 100
```

The active PVD sampling block is:

```json
{
  "sampling_method": "physical_sobol_manifold",
  "physical_texture": {
    "inference_mode": "broad_likelihood",
    "process_family": "sputter_pvd",
    "uniform_center_fraction": 0.15,
    "component_count_range": [1, 6],
    "dirichlet_concentration_range": [1.0, 1.0],
    "fwhm_strata_degrees": [[5.0, 10.0], [10.0, 20.0], [20.0, 30.0]],
    "texture_index_bounds": [1.0, 1000000.0],
    "texture_index_policy": "soft_weight",
    "texture_index_softness": 1000000.0,
    "texture_index_log_weight_floor": -1.0
  }
}
```

An EBSD-informed physical-prior specification should be introduced as a new,
versioned proposal rather than by mutating this likelihood corpus. It may use
an explicitly documented process family and a stricter prior, but it must
retain an OOD diagnostic route back to the PVD broad-likelihood model.

The maintained `process_family` is `sputter_pvd`. Legacy experimental families
(`broad`, `rolling`, `recrystallization`, and `additive_manufacturing`) remain
explicit code paths for reproducing or comparing earlier designs, but no longer
have maintained production JSON specifications. Unknown names are rejected so a
misspelled physical prior cannot silently become a broad distribution.

A crystal-system entry may include its own `physical_texture` mapping. It is
recursively overlaid on the run-level mapping, allowing—for example—a broad
cubic design and a tighter lower-symmetry design in one reproducible corpus.

The legacy control is selected explicitly with
`"sampling_method": "Sobol_sequence"` and its original
`spread_range_degrees`, `isotropic_background_range`, and
`dirichlet_concentration_range` keys.

### Saved metadata

Each system-level `odf_labels.npz` and observation-level `odf_label.npz`
contains the authoritative physical fields plus:

| Array | Shape at system level | Use |
|---|---:|---|
| `texture_index` | `(N,)` | analytic $J$ diagnostic/conditioning |
| `texture_index_log_weight` | `(N,)` | Mode A soft physical weight |
| `sampling_space` | `(N,)` | provenance/version |
| `inference_mode` | `(N,)` | prevents accidental prior mixing |
| `process_family` | `(N,)` | process conditioning |
| `component_manifold` | `(N, M)` | named center source after weight sorting |
| `component_manifold_kind` | `(N, M)` | uniform/component/fiber/beta-fiber source |
| `active_component_count` | `(N,)` | number of nonzero component slots |
| `component_active_mask` | `(N, M)` | active-slot indicator |
| `component_process_tag` | `(N, M)` | PVD provenance tag or legacy source tag |
| `sample_process_domain` | `(N,)` | ODF-level process provenance |
| `dirichlet_concentration` | `(N,)` | sampled concentration |
| `design_candidate_index` | `(N,)` | original candidate index before hard rejection |

The flat `planned_scan_index.npz` repeats $J$, its log weight, sampling space,
inference mode, and process family for efficient dataset loading.

---

## Bayesian interpretation

### Separate the simulator distribution from the scientific prior

Let $q_A(\theta)$ denote the broad generator, $q_B(\theta)$ the
process-specific generator, and $p(x\mid\theta)$ the XRD simulator. These
objects serve different purposes:

- A likelihood or likelihood-ratio estimator should learn
  $p(x\mid\theta)$ over the support needed at inference time. Train it mostly
  with Mode A and condition explicitly on nuisance variables such as grain
  count, detector geometry, strain, size, and noise.
- A direct neural posterior trained on pairs
  $(\theta,x)\sim q_B(\theta)p(x\mid\theta)$ estimates the posterior under
  the Mode B prior. It cannot later be called “prior free.”
- If training under proposal $q$ but targeting prior $p$, use a method with
  a valid proposal correction. A soft $J$ weight alone is not a substitute
  for knowing the full density ratio when the manifold mixture also changes.

For sequential neural inference, retain at least a fixed fraction of Mode A in
later rounds. Proposal-only simulations around an early posterior mode can
erase aliases and yield apparently precise but miscalibrated results.

### The descriptor is constrained, not a Euclidean box

Downstream samplers must not interpret the $1+6M$ descriptor values as
independent uniform variables:

$$
w_i\ge0,\qquad w_0+\sum_iw_i=1,
\qquad \lVert q_i\rVert=1,\qquad q_i\equiv-q_i,
\qquad q_i\equiv q_i h.
$$

Recommended inference parameterizations are:

- use a logit for $w_0$ and a softmax/stick-breaking transform for normalized
  component weights;
- use log FWHM or a bounded spline transform for widths;
- use unit-quaternion-aware distributions (Bingham, projected normal, or
  tangent-space proposals) and symmetry-marginalized densities;
- treat permutation sorting and fundamental-zone boundaries as charts, not as
  smooth physics. Prefer a permutation-invariant set encoder or explicitly
  sum over equivalent component assignments in the likelihood;
- include `process_family`, grain count, and simulator nuisance parameters as
  conditioning variables when multiple domains share one model.

### MCMC and HMC

For MAP, HMC, or MCMC, evaluate the likelihood on an unconstrained transform
of weights and widths. Quaternion proposals should remain on $S^3$, with
antipodal and crystal equivalents handled explicitly. Crossing the stored
canonical boundary can cause a discontinuity in raw target coordinates even
though the ODF and image change smoothly. HMC in canonical quaternion
coordinates can therefore diverge at a chart boundary; sampling redundant
quaternions and symmetrizing the density is safer.

Multi-chain sampling is required when several $hkl$ families or texture
components explain the same arcs. Initialize chains from multiple named modes
and uniform escape candidates. Report mode weights and posterior predictive
patterns, not only a posterior mean ODF, which may lie between physically
distinct modes.

### Variational inference

A single mean-field Gaussian is generally inadequate: it cannot represent the
simplex correlations, antipodal quaternion topology, component permutations,
or XRD aliases. Use a mixture variational family, structured simplex and
quaternion factors, or importance-weighted objectives. Validate against
simulation-based calibration because reverse-KL objectives tend to choose one
likelihood mode.

### Normalizing flows and amortized posteriors

Flows should operate on valid transformed coordinates or directly on
manifolds. A raw Euclidean descriptor flow wastes capacity learning unit norms
and can put density on invalid quaternions and weights. Set/graph encoders for
the active components, followed by a symmetry-aware decoder, avoid treating
the weight-sorted slot identity as a physical label.

For Mode B, train with a small but explicit mixture of broad Mode A examples or
an OOD head. An ensemble or density-ratio diagnostic can flag patterns that
are unlikely under the process prior. When an experimental pattern is OOD,
fall back to a Mode A likelihood model and state that the process prior was
relaxed.

### FWHM annealing and local minima

The three width strata ensure that training contains sharp, intermediate, and
smooth ODFs. During gradient-based inversion, define an annealed width

$$
\Delta_i^{(k)}=a_k\Delta_i,
\qquad a_0>a_1>\cdots>a_K=1,
$$

for example $a_k\in\{2.5,2.0,1.5,1.2,1.0\}$, clipped below $180^\circ$, and
recompute $s(\Delta_i^{(k)})$ exactly. Optimize centers and weights first
against the smoothed ODF, then reduce $a_k$. This continuation can widen
likelihood basins, but it changes the forward model at each stage; the final
posterior or acceptance ratio must use $a_K=1$. Annealing is an optimizer aid,
not evidence that the broad intermediate model generated the data.

For HMC, ordinary between-step width changes violate stationarity. Use
annealing only for initialization, or use a formally valid tempering method
such as replica exchange or sequential Monte Carlo.

### Aliasing and identifiability checklist

Before claiming an inferred texture is unique:

1. generate posterior predictive detector images and compare ring-wise and
   azimuthal residuals;
2. compare symmetry-equivalent and component-permuted ODFs before declaring
   separate modes;
3. include nuisance uncertainty for detector calibration, strain, crystallite
   size, phase fraction, background, and finite grain count;
4. test held-out Haar-escape and $J$-tail cases;
5. use multiple sample tilts, wavelengths, detectors, or complementary pole
   figures/EBSD when a single 2D pattern is structurally non-identifying;
6. evaluate posterior coverage with simulation-based calibration separately
   for Mode A, Mode B, and intentionally OOD textures.

---

## Validation and known limitations

Automated tests check:

- exact DVP normalization and the closed-form single-component texture index;
- cubic symmetry invariance;
- the explicit Copper mapping to RD and ND;
- deterministic 27-coordinate physical decoding;
- lower-symmetry Cartesian-axis manifolds and their symmetry invariance;
- sparse effective component counts and occurrence of both fibers and the
  uniform escape channel;
- Mode A nonpositive soft log weights and Mode B hard $J$ acceptance;
- descriptor schema/version consistency across repeated grain-count
  observations.

Remaining limitations are material, not hidden implementation details:

- catalog probabilities are expert templates rather than fitted distributions;
- the beta fiber is a piecewise quaternion-geodesic approximation;
- ODF widths do not replace a dislocation-informed diffraction line-profile
  model;
- the synthetic monoclinic structure remains a primitive placeholder and uses
  the repository-specific Cartesian-z monoclinic convention;
- hexagonal manifold names are orientation templates; the current forward
  driver still needs a metric-correct hexagonal lattice and calibrated c/a
  ratio before they support quantitative HCP claims;
- hard-rejected designs are not Sobol nets after acceptance;
- canonical labels retain chart discontinuities even though the represented
  ODF is symmetry invariant;
- a finite-slot ODF mixture is not an identifiable coordinate system for every
  XRD geometry. Bayesian uncertainty must reflect that fact.

The recommended scientific workflow is to start with Mode A, fit and validate
the likelihood and nuisance model, calibrate a process prior from independent
texture measurements, then train or fine-tune Mode B while preserving an OOD
route back to the broad likelihood.
