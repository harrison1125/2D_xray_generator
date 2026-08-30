# Grain shape, broadening, and detector intensity

Each realized grain now retains `grain_size`, `grain_strain`, and `volume`.
`grain_size` is the equivalent-volume spherical diameter, in the same length
unit as wavelength and lattice parameters.  `aspect_ratio` defines a
volume-preserving ellipsoid with semi-axes `[a, a, c]`, where `c/a` is the
aspect ratio and the long axis is crystal `[001]`.  Grain shape therefore
rotates with the crystal orientation.

For each diffracted reciprocal vector, the simulator uses the ellipsoid chord
parallel to that vector as the coherent length `L`.  The radial FWHM in
scattering angle is

`sqrt((K lambda / (L cos(theta)))^2 + (abs(strain) tan(theta))^2)`

with `K=0.9` by default.  The first term is Scherrer broadening; the second is
the repository's deliberately simplified legacy microstrain term.  It is
converted to detector pixels using `r = D tan(2theta)` and combined in
quadrature with the configurable instrumental FWHM.  The tangential FWHM is
instrumental only.  Thus non-spherical grains create orientation-dependent,
elliptical detector spots.

Each spot is a discretely normalized elliptical Gaussian.  Its image sum—not
its peak height—is the integrated reflection intensity:

`intensity_scale * (grain_volume/reference_volume) * |F_hkl|^2 * Lp * P`

where `Lp = 1/abs(sin(2theta))` is the existing transmission Lorentz factor
and `P = (1 + cos(2theta)^2)/2` is the unpolarized polarization factor.  This
means changing peak width does not spuriously change total intensity, while a
larger illuminated grain contributes proportionally more intensity.  The
default instrumental FWHM is 4.70964 pixels (`sigma=2 px`), preserving the
old renderer's baseline width when size and strain broadening are negligible.

Use the existing driver options, for example:

```python
result = diffraction_pattern("Cu", "FCC", 1000, 0.361, seed=123,
                             shape_factor=0.9,
                             instrumental_fwhm_px=4.7,
                             intensity_scale=1.0)
```
