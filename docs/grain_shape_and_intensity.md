# Grain size, spot shape, projection, and intensity

Updated: 2026-09-01 20:48 EDT

## Model provenance and scope

The active single-shot renderer now adapts the physically compatible parts of
`fable_xrdsim`:

- `scattering_factors._scherrer`: equivalent-volume size and Scherrer width;
- `Detector._get_intersection`: rays originate at the sample and intersect a
  real-space detector plane;
- `Detector._render_voigt_peaks`: size broadening is the Lorentzian part of a
  Voigt profile and the detector point-spread function is Gaussian;
- `Detector._render_projected_volumes`: intensity scales with illuminated
  volume and real-space grain morphology affects the detector footprint.

Two fable features are intentionally not copied. Its
`1/(sin(2theta) abs(sin(eta)))` Lorentz factor is for a rotating scan, not a
static exposure, and its exact path-length projection requires a tetrahedral
mesh. This repository instead projects its existing ellipsoid through an exact
source-to-plane Jacobian and uses the resulting second moments.

## Units and grain population

Experiment wavelength and lattice lengths are nanometres. Grain size accepts
`size_unit: "nm"` or `"um"` and is converted internally to nanometres for
reciprocal-space calculations and to cubic micrometres for intensity.

The existing default is retained but is now explicit:

- mean equivalent diameter: 33,500 nm = 33.5 µm;
- standard deviation: 5,750 nm = 5.75 µm;
- detector pitch: 0.075 mm = 75 µm.

Grain size is a material/microstructure input; it cannot be inferred from
detector pixel pitch. A 75 µm pixel only determines whether a given grain
footprint is resolved. Configured runs use a log-normal diameter distribution
with the requested arithmetic mean and standard deviation. This guarantees
positive sizes without clipping a normal distribution at zero. The legacy
`normal` option remains available.

For aspect ratio `r=c/a`, the volume-preserving ellipsoid semi-axes are

`a = d / (2 r^(1/3)), c = r a`

and its volume is

`V_um3 = pi d_um^3 / 6`.

The ellipsoid long axis is crystal `[001]` and rotates with the grain
orientation.

## Single-shot reflection acceptance

`detector.ewald_tolerance` is now only a candidate-search window in reciprocal
nanometres. A candidate's continuous excitation error is

`s = abs(|k_in + G| - |k_in|)`.

For coherent length `L`, the reciprocal Scherrer FWHM is `K/L`. The static
shot weights the reflection with a Lorentzian section through that broadened
reciprocal point:

`W_exc = 1 / (1 + (2 s L / K)^2)`.

Thus a candidate at half the reciprocal FWHM has half intensity. This replaces
the old flat behavior where every point inside a wide `0.1 nm^-1` shell was
equally “on Bragg.” The candidate window still must be wide enough for the
smallest crystallites being simulated.

When `experiment.incident_convergence_full_angle_mrad` is nonzero, this
finite-size Lorentzian is averaged over a uniform one-dimensional incident-ray
distribution in the reflection scattering plane. A value of `3.5` therefore
integrates from -1.75 to +1.75 mrad. For reciprocal FWHM `Gamma`, full angle
`2 alpha`, and `A = sin(2theta) / lambda`, the normalized excitation weight is

`Gamma / (4 A alpha) * [atan(2(s + A alpha)/Gamma) - atan(2(s - A alpha)/Gamma)]`.

The angular average conserves fixed total incident flux: it redistributes
intensity from a few exact-Bragg grains to the many grains whose rocking curves
intersect the convergence interval. This is a uniform 1-D scattering-plane
approximation, not a coherent wave-optics model of an axisymmetric cone.

## Spot-width calculation

For each reciprocal vector, the coherent length `L` is the central ellipsoid
chord parallel to that vector. With `theta = (2theta)/2`:

- Lorentzian size FWHM:
  `beta_size = K lambda / (L cos(theta))`;
- Gaussian Williamson-Hall microstrain FWHM:
  `beta_strain = 4 abs(epsilon) tan(theta)`.

The radial detector coordinate is `r = D tan(2theta)`, so an angular FWHM is
mapped to pixels by

`beta_px = beta_rad D_px sec(2theta)^2`.

Real-space ellipsoid projection is included as a Gaussian-equivalent
second moment. A uniform ellipsoid has crystal-frame covariance
`diag(a^2, a^2, c^2)/5`. After rotation to the sample frame, the ray/plane
Jacobian for detector coordinates `[row(z), col(y)]` is

`J = [[-kz/kx, 0, 1], [-ky/kx, 1, 0]]`.

The detector covariance is

`Sigma_proj_px = J Sigma_grain J.T / pixel_pitch^2`.

Its radial and tangential variances are combined with detector PSF and strain:

`sigma_radial^2 = sigma_PSF^2 + sigma_strain^2 + sigma_proj,radial^2`

`sigma_tangent^2 = sigma_PSF^2 + sigma_proj,tangent^2`.

The rendered radial profile is a Voigt function with Lorentz HWHM
`gamma = beta_size_px/2` and Gaussian width `sigma_radial`. The tangential
profile is Gaussian. The reported radial Voigt FWHM uses

`FWHM_V ~= 0.5346 FWHM_L + sqrt(0.2166 FWHM_L^2 + FWHM_G^2)`.

At the current defaults, a 33.5 µm spherical grain on 75 µm pixels contributes
about 0.10 pixel projected sigma at normal incidence. Its Scherrer detector
width is roughly 0.001 pixel for the current wavelength and geometry. Both are
far below the configured instrumental FWHM of 4.70964 pixels
(`sigma_PSF=2 px`), so the present images remain detector-PSF dominated.
That 4.70964-pixel value is a retained renderer setting, not a prediction from
grain size; it should be replaced by a measured detector/optics calibration.

## Projection and intensity

The outgoing ray is `k_out = k_in + G` and now starts at the real-space sample
origin. For the centered detector plane `x=D`:

`y_D = D ky/kx, z_D = D kz/kx`.

The old code incorrectly used the reciprocal-space Ewald-sphere center as if it
were the real-space ray origin.

Each spot kernel is discretely normalized before detector clipping, so its sum
is the integrated reflection intensity and peaks cut by a detector edge lose
the out-of-bounds fraction. The relative integrated intensity is

`I = intensity_scale V_um3 |F_hkl|^2 P W_exc L_model`

with unpolarized

`P = (1 + cos(2theta)^2)/2`.

`L_model` is 1 by default for a static shot. The former
`1/abs(sin(2theta))` term is available only as
`lorentz_model: "legacy"` for comparison. Absolute counts are still not
predicted: beam flux, illuminated fraction, exposure, atomic form factors,
absorption, detector efficiency, background, and counting noise are not yet
modeled. Every configured grain is currently treated as fully illuminated. If
grain diameter is changed while `num_grains` is held fixed, the total simulated
material volume changes; a fixed-volume sample model must instead couple grain
count to the size distribution.

## Configuration example

```json
{
  "experiment": {"length_unit": "nm", "wavelength": 0.0514},
  "grain": {
    "size_mean": 33500.0,
    "size_std": 5750.0,
    "size_unit": "nm",
    "size_distribution": "lognormal",
    "aspect_ratio": 1.5
  },
  "detector": {
    "width_px": 1000,
    "height_px": 1000,
    "distance_mm": 86.0,
    "pixel_size_mm": 0.075,
    "bin_to_width_px": 256,
    "bin_to_height_px": 256
  },
  "scattering": {
    "shape_factor": 0.9,
    "profile": "voigt",
    "instrumental_fwhm_px": 4.709640090061899,
    "lorentz_model": "none",
    "intensity_scale": 1.0
  }
}
```

Per-peak metadata now separates Scherrer, strain, projected-size, excitation,
and final radial/tangential widths. Per-grain metadata records the configured
size/unit, internal nanometre size, and physical volume in µm³.

When binning is requested, diffraction and the native PSF are calculated on
the physical detector first. Equal-area integration then maps the native image
to the requested output shape, including non-integer ratios such as
1000/256 = 3.90625. Integrated intensity is conserved. Readout metadata records
the native and output shapes, effective pixel pitch, binning ratio, and PSF in
output-pixel units; peak coordinates remain in native detector pixels.
