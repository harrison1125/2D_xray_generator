"""Continuous, symmetry-invariant ODFs and Sobol phase-space sampling.

The primary distribution is a convex mixture of a Haar-uniform background and
de la Vallée Poussin kernels on SO(3).  All densities are relative to
normalized Haar measure, so their integral is exactly one by construction.

For a rotation distance ``theta`` and exponent ``s``, the kernel is::

    K_s(theta) = C_s cos(theta / 2) ** (2 s)

with ``C_s = sqrt(pi) Gamma(s + 2) / Gamma(s + 1/2)``.  ``s`` is selected
from a requested *full* width at half maximum (FWHM).  Symmetry is enforced by
averaging every component over the proper rotational subgroup of the crystal
point group; for a crystal-to-sample orientation q, this is the right action
q -> q h.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, log2
from typing import Any
import warnings

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import gammaincinv, gammaln
from scipy.stats import qmc

from .orientation import (axis_angle_to_quaternion, canonicalize_quaternion,
                          compose_rotations, quaternion_to_rotation_matrix,
                          uniform_quaternions)
from .texture import ODF, get_symmetry_operations, reduce_to_fundamental_zone


_TINY = np.finfo(float).tiny


def sobol_quaternions(unit_cube: np.ndarray) -> np.ndarray:
    """Map Sobol coordinates ``[..., 3]`` exactly to Haar-uniform SO(3).

    This is Shoemake's map from a three-dimensional unit cube to unit
    quaternions.  It is measure preserving for Haar measure, unlike uniform
    sampling of Euler angles or quaternion Cartesian coordinates.
    """
    u = np.asarray(unit_cube, dtype=float)
    if u.shape[-1] != 3 or np.any((u < 0) | (u > 1)):
        raise ValueError("unit_cube must have final dimension 3 with values in [0, 1].")
    u1, u2, u3 = np.moveaxis(u, -1, 0)
    q = np.stack((
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
    ), axis=-1)
    return canonicalize_quaternion(q)


def sobol_unit_cube(n: int, dimension: int, seed: int | None = None) -> np.ndarray:
    """Return a scrambled Sobol prefix without pseudo-random fallback.

    Sobol nets are strongest at powers of two.  For arbitrary ``n`` (such as
    50,000) we generate the enclosing power-of-two net and retain its prefix;
    no independent pseudo-random points are inserted into the design.
    """
    if n < 1 or dimension < 1:
        raise ValueError("n and dimension must be positive.")
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = engine.random_base2(ceil(log2(n)))
    return points[:n]


def dvp_exponent_from_fwhm(fwhm_deg: np.ndarray | float) -> np.ndarray:
    """Return de la Vallée Poussin exponents for a full angular FWHM.

    The half maximum occurs at geodesic distance ``FWHM / 2``.  The finite
    range below 180 degrees is deliberate: it avoids treating the constant
    kernel (whose FWHM is undefined) as a finite-width component.
    """
    fwhm = np.asarray(fwhm_deg, dtype=float)
    if np.any((fwhm <= 0) | (fwhm >= 180)):
        raise ValueError("DVP FWHM must lie strictly between 0 and 180 degrees.")
    return np.log(0.5) / (2 * np.log(np.cos(np.deg2rad(fwhm) / 4)))


def dvp_log_normalization(exponent: np.ndarray | float) -> np.ndarray:
    """Log normalizer under normalized Haar measure on SO(3)."""
    s = np.asarray(exponent, dtype=float)
    if np.any(s < 0):
        raise ValueError("DVP exponent must be nonnegative.")
    return 0.5 * np.log(np.pi) + gammaln(s + 2) - gammaln(s + 0.5)


def _sample_dvp_about(centers: np.ndarray, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a normalized DVP kernel exactly, conditional on its centers."""
    n = len(centers)
    # t = cos(theta/2)^2 is Beta(s + 1/2, 3/2) under K_s dHaar.
    t = rng.beta(exponent + 0.5, 1.5, size=n)
    angle = 2 * np.arccos(np.sqrt(np.clip(t, 0, 1)))
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    return canonicalize_quaternion(compose_rotations(centers, axis_angle_to_quaternion(axis, angle)))


class ContinuousSymmetricMixtureODF(ODF):
    """A normalized continuous ODF with crystal-symmetry invariance.

    ``background_weight`` is the isotropic fraction.  Component weights must
    sum to ``1 - background_weight``.  Components are symmetrized over the
    proper rotation group, so ``evaluate(q h) == evaluate(q)`` for each crystal
    symmetry operation ``h`` (up to floating-point error).
    """

    def __init__(self, component_centers: np.ndarray, component_fwhm_deg: np.ndarray,
                 component_weights: np.ndarray, background_weight: float,
                 crystal_symmetry: str):
        centers = canonicalize_quaternion(np.asarray(component_centers, dtype=float))
        fwhm = np.asarray(component_fwhm_deg, dtype=float)
        weights = np.asarray(component_weights, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 4 or len(centers) == 0:
            raise ValueError("component_centers must have shape (n_components, 4).")
        if fwhm.shape != (len(centers),) or weights.shape != (len(centers),):
            raise ValueError("Each component needs one FWHM and one weight.")
        if not 0 <= background_weight <= 1 or np.any(weights < 0):
            raise ValueError("ODF weights must be nonnegative.")
        if not np.isclose(background_weight + weights.sum(), 1.0, atol=1e-12):
            raise ValueError("background_weight plus component_weights must sum to one.")
        self.crystal_symmetry = str(crystal_symmetry).lower()
        self.symmetry_operations = get_symmetry_operations(self.crystal_symmetry)
        self.component_centers = centers
        self.component_fwhm_deg = fwhm
        self.component_weights = weights
        self.background_weight = float(background_weight)
        self.component_exponents = dvp_exponent_from_fwhm(fwhm)
        self.component_log_normalizers = dvp_log_normalization(self.component_exponents)
        self.symmetry_centers = canonicalize_quaternion(compose_rotations(
            centers[:, None, :], self.symmetry_operations[None, :, :]
        ))

    @property
    def num_components(self) -> int:
        return len(self.component_weights)

    @property
    def effective_components(self) -> float:
        """Inverse Simpson count of the non-background component weights."""
        total = self.component_weights.sum()
        if total == 0:
            return 0.0
        p = self.component_weights / total
        return float(1 / np.sum(p * p))

    def evaluate(self, orientation: np.ndarray) -> np.ndarray:
        q = canonicalize_quaternion(np.asarray(orientation, dtype=float))
        # Shape: orientation_shape + (component, symmetry_operation).
        dot = np.abs(np.einsum("...d,khd->...kh", q, self.symmetry_centers))
        log_kernel = (self.component_log_normalizers.reshape((1,) * (dot.ndim - 2) + (-1, 1))
                      + 2 * self.component_exponents.reshape((1,) * (dot.ndim - 2) + (-1, 1))
                      * np.log(np.clip(dot, _TINY, 1.0)))
        component_density = np.exp(log_kernel).mean(axis=-1)
        return self.background_weight + np.sum(component_density * self.component_weights, axis=-1)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative.")
        rng = np.random.default_rng() if rng is None else rng
        probabilities = np.concatenate(([self.background_weight], self.component_weights))
        choices = rng.choice(self.num_components + 1, size=n, p=probabilities)
        result = np.empty((n, 4), dtype=float)
        background = choices == 0
        result[background] = uniform_quaternions(int(background.sum()), rng)
        for component in range(self.num_components):
            mask = choices == component + 1
            count = int(mask.sum())
            if not count:
                continue
            operation_index = rng.integers(len(self.symmetry_operations), size=count)
            centers = self.symmetry_centers[component, operation_index]
            result[mask] = _sample_dvp_about(centers, self.component_exponents[component], rng)
        return canonicalize_quaternion(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "continuous_symmetric_mixture",
            "kernel_type": "de_la_vallee_poussin",
            "crystal_symmetry": self.crystal_symmetry,
            "background_weight": self.background_weight,
            "component_centers": self.component_centers.tolist(),
            "component_fwhm_deg": self.component_fwhm_deg.tolist(),
            "component_weights": self.component_weights.tolist(),
            "effective_components": self.effective_components,
            "normalization": "exact_relative_to_normalized_haar",
            "proper_symmetry_order": int(len(self.symmetry_operations)),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContinuousSymmetricMixtureODF":
        if value.get("kernel_type", "de_la_vallee_poussin") != "de_la_vallee_poussin":
            raise ValueError("Only de_la_vallee_poussin is implemented by this ODF.")
        return cls(value["component_centers"], value["component_fwhm_deg"],
                   value["component_weights"], value["background_weight"],
                   value["crystal_symmetry"])


@dataclass(frozen=True)
class SobolODFSpace:
    """Sobol design over a fixed-width, continuous mixture parameterization.

    All ``max_components`` slots exist for every ODF.  A Sobol-mapped Dirichlet
    distribution makes some weights arbitrarily small, yielding a continuous
    effective-component count rather than a categorical family boundary.
    """

    crystal_symmetry: str
    max_components: int = 5
    fwhm_range_deg: tuple[float, float] = (2.5, 30.0)
    background_range: tuple[float, float] = (0.05, 0.95)
    dirichlet_concentration_range: tuple[float, float] = (0.2, 4.0)
    spread_scale: str = "linear"

    @property
    def dimension(self) -> int:
        # background, Dirichlet concentration, then q(3), FWHM, and gamma
        # coordinate for every fixed mixture slot.
        return 2 + self.max_components * 5

    def __post_init__(self):
        if self.max_components < 1:
            raise ValueError("max_components must be positive.")
        lo, hi = self.fwhm_range_deg
        if not (0 < lo <= hi < 180):
            raise ValueError("FWHM range must lie in (0, 180).")
        b0, b1 = self.background_range
        if not (0 <= b0 <= b1 <= 1):
            raise ValueError("background range must lie in [0, 1].")
        a0, a1 = self.dirichlet_concentration_range
        if not (0 < a0 <= a1):
            raise ValueError("Dirichlet concentration range must be positive.")
        if self.spread_scale not in {"linear", "log"}:
            raise ValueError("spread_scale must be 'linear' or 'log'.")
        get_symmetry_operations(self.crystal_symmetry)

    def _spread(self, u: float) -> float:
        lo, hi = self.fwhm_range_deg
        if self.spread_scale == "linear":
            return float(lo + u * (hi - lo))
        return float(np.exp(np.log(lo) + u * np.log(hi / lo)))

    def from_unit_cube(self, unit_cube: np.ndarray) -> ContinuousSymmetricMixtureODF:
        u = np.asarray(unit_cube, dtype=float)
        if u.shape != (self.dimension,) or np.any((u < 0) | (u > 1)):
            raise ValueError(f"Expected one Sobol point of shape ({self.dimension},).")
        background = self.background_range[0] + u[0] * (self.background_range[1] - self.background_range[0])
        alpha_lo, alpha_hi = self.dirichlet_concentration_range
        alpha = float(np.exp(np.log(alpha_lo) + u[1] * np.log(alpha_hi / alpha_lo)))
        centers, fwhm, gamma_quantiles = [], [], []
        offset = 2
        for _ in range(self.max_components):
            center = sobol_quaternions(u[offset:offset + 3])
            # Canonicalize the orbit for unambiguous stored latent labels;
            # the evaluated ODF remains explicitly symmetrized below.
            centers.append(reduce_to_fundamental_zone(center, self.crystal_symmetry))
            fwhm.append(self._spread(float(u[offset + 3])))
            gamma_quantiles.append(float(np.clip(u[offset + 4], 1e-15, 1 - 1e-15)))
            offset += 5
        raw = gammaincinv(alpha, np.asarray(gamma_quantiles))
        weights = (1 - background) * raw / raw.sum()
        # Eliminate mixture label-switching in metadata without changing f(g).
        order = np.lexsort((np.asarray(fwhm), -weights))
        return ContinuousSymmetricMixtureODF(np.asarray(centers)[order], np.asarray(fwhm)[order],
                                             weights[order], background, self.crystal_symmetry)

    def sample(self, n: int, seed: int | None = None) -> tuple[np.ndarray, list[ContinuousSymmetricMixtureODF]]:
        points = sobol_unit_cube(n, self.dimension, seed)
        return points, [self.from_unit_cube(point) for point in points]


def sobol_odf_grid(num_points: int, seed: int = 0) -> np.ndarray:
    """A reusable Haar-uniform SO(3) grid for tabulated ODF labels."""
    return sobol_quaternions(sobol_unit_cube(num_points, 3, seed))


def _chebyshev_moments_under_dvp(exponent: float, maximum_order: int) -> np.ndarray:
    """E[cos(2 m x)] for x=theta/2 under the normalized DVP kernel."""
    # t=cos(x)^2 ~ Beta(s+1/2, 3/2); cos(2m x)=T_m(2t-1).
    from numpy.polynomial import Polynomial
    a, b = exponent + 0.5, 1.5
    moments = np.exp(gammaln(a + np.arange(maximum_order + 1)) - gammaln(a)
                     + gammaln(a + b) - gammaln(a + b + np.arange(maximum_order + 1)))
    z = Polynomial([-1.0, 2.0])
    polynomials = [Polynomial([1.0])]
    if maximum_order:
        polynomials.append(z)
    for _ in range(2, maximum_order + 1):
        polynomials.append(2 * z * polynomials[-1] - polynomials[-2])
    return np.array([np.dot(p.coef, moments[:len(p.coef)]) for p in polynomials])


def dvp_harmonic_scalars(exponent: float, bandlimit: int) -> np.ndarray:
    """Exact SO(3) Fourier scalars of a normalized DVP kernel.

    For a central kernel the degree-l coefficient is ``k_l I``.  This uses
    the character identity chi_l(theta)=1+2 sum_{m=1}^l cos(m theta), avoiding
    numerical SO(3) quadrature for every generated ODF.
    """
    values = _chebyshev_moments_under_dvp(exponent, bandlimit)
    scalars = np.empty(bandlimit + 1)
    for ell in range(bandlimit + 1):
        scalars[ell] = (1 + 2 * values[1:ell + 1].sum()) / (2 * ell + 1)
    return scalars


def wigner_d_matrix(degree: int, quaternion: np.ndarray) -> np.ndarray:
    """Return the standard complex Wigner-D matrix for an active rotation.

    The implementation uses an intrinsic ZYZ Euler decomposition solely for
    the harmonic basis.  It is independent of the simulator's Bunge display
    convention and is mathematically a representation of the same rotation.
    """
    if degree < 0 or int(degree) != degree:
        raise ValueError("degree must be a nonnegative integer.")
    degree = int(degree)
    matrix = quaternion_to_rotation_matrix(quaternion)
    # Euler charts are singular at beta=0 or pi, but the Wigner matrix is not.
    # SciPy warns while choosing an arbitrary third angle there; suppress that
    # representational warning rather than emitting it for symmetry identities.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        alpha, beta, gamma = Rotation.from_matrix(matrix).as_euler("ZYZ")
    m_values = np.arange(-degree, degree + 1)
    small_d = np.zeros((2 * degree + 1, 2 * degree + 1), dtype=float)
    cos_half, sin_half = np.cos(beta / 2), np.sin(beta / 2)
    for row, mp in enumerate(m_values):
        for col, m in enumerate(m_values):
            log_prefactor = 0.5 * (gammaln(degree + mp + 1) + gammaln(degree - mp + 1)
                                   + gammaln(degree + m + 1) + gammaln(degree - m + 1))
            lower = max(0, m - mp)
            upper = min(degree + m, degree - mp)
            for k in range(lower, upper + 1):
                cos_power = 2 * degree + m - mp - 2 * k
                sin_power = mp - m + 2 * k
                if cos_power < 0 or sin_power < 0:
                    continue
                log_denominator = (gammaln(degree + m - k + 1) + gammaln(k + 1)
                                   + gammaln(mp - m + k + 1) + gammaln(degree - mp - k + 1))
                small_d[row, col] += ((-1) ** int(mp - m + k)
                                      * np.exp(log_prefactor - log_denominator)
                                      * cos_half ** cos_power * sin_half ** sin_power)
    return (np.exp(-1j * np.outer(m_values, np.array([alpha])))
            * small_d
            * np.exp(-1j * np.outer(np.ones_like(m_values), m_values * gamma)))


@lru_cache(maxsize=128)
def _symmetry_harmonic_projector(crystal_symmetry: str, degree: int) -> np.ndarray:
    """Average D(h)^* over a proper crystal symmetry group."""
    operations = get_symmetry_operations(crystal_symmetry)
    return sum((wigner_d_matrix(degree, operation).conj().T for operation in operations)) / len(operations)


def harmonic_coefficients(odf: ContinuousSymmetricMixtureODF, bandlimit: int) -> list[np.ndarray]:
    """Return exact truncated SO(3) Fourier matrices for this analytic ODF.

    Coefficients use ``F_l = integral f(g) D_l(g)^* dg`` under normalized Haar
    measure.  Thus ``F_0 = [[1]]`` for every normalized ODF.  The returned
    matrices are exact up to floating-point arithmetic, not sampled estimates.
    """
    if bandlimit < 0:
        raise ValueError("bandlimit must be nonnegative.")
    output = []
    kernel_scalars = [dvp_harmonic_scalars(s, bandlimit) for s in odf.component_exponents]
    for ell in range(bandlimit + 1):
        size = 2 * ell + 1
        coefficient = np.zeros((size, size), dtype=complex)
        if ell == 0:
            coefficient[0, 0] = odf.background_weight
        projector = _symmetry_harmonic_projector(odf.crystal_symmetry, ell)
        for component, weight in enumerate(odf.component_weights):
            # D(q h)^* = D(h)^* D(q)^*, so group averaging is a fixed
            # symmetry projector rather than an O(|H|) loop per component.
            mean_dagger = projector @ wigner_d_matrix(ell, odf.component_centers[component]).conj().T
            coefficient += weight * kernel_scalars[component][ell] * mean_dagger
        output.append(coefficient)
    return output


def pack_harmonic_coefficients(coefficients: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Pack variable-sized Fourier matrices into arrays suitable for NPZ."""
    offsets = [0]
    flat = []
    for coefficient in coefficients:
        flat.append(coefficient.ravel())
        offsets.append(offsets[-1] + coefficient.size)
    values = np.concatenate(flat) if flat else np.empty(0, complex)
    return {"harmonic_offsets": np.asarray(offsets, dtype=np.int64),
            "harmonic_real": values.real.astype(np.float64),
            "harmonic_imag": values.imag.astype(np.float64)}
