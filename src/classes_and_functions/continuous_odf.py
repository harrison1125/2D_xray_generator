"""Continuous, symmetry-invariant ODFs and physical-manifold sampling.

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
from scipy.integrate import quad
from scipy.special import betaincinv, gammaincinv, gammaln, gammasgn, hyp2f1
from scipy.stats import qmc

from .orientation import (axis_angle_to_quaternion, canonicalize_quaternion,
                          compose_rotations, quaternion_to_rotation_matrix,
                          rotation_matrix_to_quaternion, uniform_quaternions)
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


def _dvp_overlap_quadrature(exponent_a: float, exponent_b: float,
                            quaternion_dot: float) -> float:
    """Stable fallback for the analytic DVP pair-overlap expression.

    The closed form used by :func:`dvp_pair_overlap` contains a hypergeometric
    function whose unscaled value can overflow for very sharp, nearly aligned
    kernels.  This one-dimensional integral is the same analytic reduction on
    S3, evaluated after subtracting its maximum log integrand.
    """
    s, t = float(exponent_a), float(exponent_b)
    delta = float(np.arccos(np.clip(abs(quaternion_dot), 0.0, 1.0)))
    tangent = np.tan(min(delta, np.pi / 2 - 1e-14))
    if tangent < 1e-14:
        phi_max = 0.0
    else:
        discriminant = np.hypot(s + t, 2 * np.sqrt(s * t) * tangent)
        x = 2 * t * tangent / (s + t + discriminant)
        phi_max = float(np.arctan(x))
    maximum = (2 * s * np.log(max(abs(np.cos(phi_max)), _TINY))
               + 2 * t * np.log(max(abs(np.cos(phi_max - delta)), _TINY)))

    def scaled_integrand(phi: float) -> float:
        value = (2 * s * np.log(max(abs(np.cos(phi)), _TINY))
                 + 2 * t * np.log(max(abs(np.cos(phi - delta)), _TINY)))
        return float(np.exp(value - maximum))

    breakpoints = sorted({0.0, delta, np.pi / 2, delta + np.pi / 2,
                          np.pi, delta + np.pi, 3 * np.pi / 2, 2 * np.pi})
    angular = quad(scaled_integrand, 0.0, 2 * np.pi, points=breakpoints[1:-1],
                   epsabs=1e-11, epsrel=2e-10, limit=160)[0]
    log_value = (float(dvp_log_normalization(s) + dvp_log_normalization(t))
                 - np.log(np.pi) - np.log(2 * (s + t + 1))
                 + maximum + np.log(angular))
    return float(np.exp(log_value))


def dvp_pair_overlap(exponent_a: np.ndarray | float,
                     exponent_b: np.ndarray | float,
                     quaternion_dot: np.ndarray | float) -> np.ndarray:
    r"""Integrate the product of two normalized DVP kernels exactly.

    For centers with absolute quaternion inner product ``rho``, the result is

    .. math::

       \frac{\Gamma(s+2)\Gamma(t+2)}{\Gamma(s+t+2)}
       {}_2F_1(-s,-t;\tfrac12;\rho^2).

    The implementation evaluates this closed form in log space, drops only
    terms with a rigorous maximum-based bound below ``1e-14``, and uses the
    equivalent one-dimensional S3 integral when SciPy's unscaled hypergeometric
    evaluation overflows.
    """
    s, t, rho = np.broadcast_arrays(
        np.asarray(exponent_a, dtype=float),
        np.asarray(exponent_b, dtype=float),
        np.clip(np.abs(np.asarray(quaternion_dot, dtype=float)), 0.0, 1.0),
    )
    if np.any((s < 0) | (t < 0)):
        raise ValueError("DVP exponents must be nonnegative.")
    shape = s.shape
    sf, tf, rf = s.ravel(), t.ravel(), rho.ravel()
    result = np.zeros_like(sf)

    aligned = rf >= 1 - 4 * np.finfo(float).eps
    if np.any(aligned):
        log_aligned = (dvp_log_normalization(sf[aligned])
                       + dvp_log_normalization(tf[aligned])
                       - dvp_log_normalization(sf[aligned] + tf[aligned]))
        result[aligned] = np.exp(log_aligned)

    remaining = ~aligned
    if np.any(remaining):
        sr, tr, rr = sf[remaining], tf[remaining], rf[remaining]
        delta = np.arccos(rr)
        tangent = np.tan(np.minimum(delta, np.pi / 2 - 1e-14))
        discriminant = np.hypot(sr + tr, 2 * np.sqrt(sr * tr) * tangent)
        x = np.divide(2 * tr * tangent, sr + tr + discriminant,
                      out=np.zeros_like(tangent), where=(sr + tr + discriminant) > 0)
        phi = np.arctan(x)
        maximum = (2 * sr * np.log(np.clip(np.abs(np.cos(phi)), _TINY, None))
                   + 2 * tr * np.log(np.clip(np.abs(np.cos(phi - delta)), _TINY, None)))
        log_upper = dvp_log_normalization(sr) + dvp_log_normalization(tr) + maximum
        material = log_upper >= np.log(1e-14)
        local = np.zeros_like(sr)
        if np.any(material):
            sm, tm, rm = sr[material], tr[material], rr[material]
            log_prefactor = gammaln(sm + 2) + gammaln(tm + 2) - gammaln(sm + tm + 2)
            hyper = hyp2f1(-sm, -tm, 0.5, rm * rm)
            valid = np.isfinite(hyper) & (hyper > 0)
            values = np.empty_like(sm)
            values[valid] = np.exp(log_prefactor[valid] + np.log(hyper[valid]))

            fallback = ~valid
            if np.any(fallback):
                # Use both signed terms of the 2F1 connection formula about
                # z=1.  Their ratio is well scaled for the near-aligned cases
                # that overflow the direct expression.  If cancellation is
                # still unresolved in float64, use the angular integral.
                ss, tt, zz = sm[fallback], tm[fallback], rm[fallback] ** 2
                complement = 1 - zz
                log_a = (gammaln(0.5) + gammaln(ss + tt + 0.5)
                         - gammaln(ss + 0.5) - gammaln(tt + 0.5))
                first = hyp2f1(-ss, -tt, -ss - tt + 0.5, complement)
                sign_b = (gammasgn(-ss - tt - 0.5)
                          * gammasgn(-ss) * gammasgn(-tt))
                log_b = (gammaln(0.5) + gammaln(-ss - tt - 0.5)
                         - gammaln(-ss) - gammaln(-tt))
                second_hyper = hyp2f1(
                    ss + 0.5, tt + 0.5, ss + tt + 1.5, complement
                )
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    relative_second = sign_b * np.exp(
                        log_b - log_a + (ss + tt + 0.5) * np.log(complement)
                        + np.log(second_hyper)
                    )
                relative_second = np.where(sign_b == 0, 0.0, relative_second)
                connection = first + relative_second
                connection_valid = np.isfinite(connection) & (connection > 0)
                fallback_values = np.empty_like(ss)
                fallback_values[connection_valid] = np.exp(
                    log_prefactor[fallback][connection_valid]
                    + log_a[connection_valid]
                    + np.log(connection[connection_valid])
                )
                for index in np.flatnonzero(~connection_valid):
                    fallback_values[index] = _dvp_overlap_quadrature(
                        ss[index], tt[index], np.sqrt(zz[index])
                    )
                values[fallback] = fallback_values
            local[material] = values
        result[remaining] = local
    return result.reshape(shape)


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

    def texture_index(self) -> float:
        r"""Return ``integral f(g)^2 dg`` under normalized Haar measure.

        The component-component term uses the analytic DVP overlap and reduces
        the apparent ``|H|^2`` symmetry sum to ``|H|`` by Haar invariance.
        This is deterministic and does not miss sharp peaks as a Monte Carlo
        ODF grid can.
        """
        # Fixed-width descriptor schemas may reserve inactive zero-weight slots.
        # They are mathematically absent from the ODF and must not make the
        # analytic diagnostic scale with the maximum allocated slot count.
        active = self.component_weights > 0
        if not np.any(active):
            return 1.0
        centers = self.component_centers[active]
        exponents = self.component_exponents[active]
        weights = self.component_weights[active]
        right_orbits = canonicalize_quaternion(compose_rotations(
            centers[:, None, :],
            self.symmetry_operations[None, :, :],
        ))
        dots = np.abs(np.einsum("id,jhd->ijh", centers, right_orbits))
        exponents_a = np.broadcast_to(
            exponents[:, None, None], dots.shape
        )
        exponents_b = np.broadcast_to(
            exponents[None, :, None], dots.shape
        )
        overlaps = dvp_pair_overlap(exponents_a, exponents_b, dots).mean(axis=-1)
        textured = float(np.einsum(
            "i,ij,j->", weights, overlaps, weights
        ))
        background_term = 2 * self.background_weight - self.background_weight ** 2
        # J >= 1 follows from Cauchy-Schwarz for a normalized density.  Permit
        # a few ulps of analytic/numeric error, but do not hide a real failure.
        value = background_term + textured
        if value < 1 - 2e-10 or not np.isfinite(value):
            raise FloatingPointError(f"Invalid analytic texture index {value!r}.")
        return float(max(1.0, value))

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


def _unit_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or norm == 0:
        raise ValueError("Expected a nonzero three-vector.")
    return vector / norm


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source, target = _unit_vector(source), _unit_vector(target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine > 1 - 1e-13:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.cross(source, target)
    if cosine < -1 + 1e-13:
        trial = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, trial)
    return axis_angle_to_quaternion(axis, np.arccos(cosine))


def orientation_from_plane_direction(plane_normal: np.ndarray | list[float],
                                     rolling_direction: np.ndarray | list[float]) -> np.ndarray:
    r"""Map a conventional ``{hkl}<uvw>`` component into RD/TD/ND.

    ``rolling_direction`` maps to sample RD = x and ``plane_normal`` maps to
    sample ND = z.  The two crystal vectors must be orthogonal.  The resulting
    active matrix maps the right-handed crystal triad
    ``[uvw], [hkl] cross [uvw], [hkl]`` onto ``RD, TD, ND``.
    """
    normal = _unit_vector(plane_normal)
    direction = _unit_vector(rolling_direction)
    if abs(normal @ direction) > 1e-10:
        raise ValueError("The rolling direction must lie in the specified crystal plane.")
    transverse = _unit_vector(np.cross(normal, direction))
    crystal_basis = np.column_stack((direction, transverse, normal))
    return rotation_matrix_to_quaternion(crystal_basis.T)


def _quaternion_slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    dot = float(first @ second)
    if dot < 0:
        second, dot = -second, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1 - 1e-10:
        return canonicalize_quaternion((1 - fraction) * first + fraction * second)
    angle = np.arccos(dot)
    return canonicalize_quaternion(
        np.sin((1 - fraction) * angle) / np.sin(angle) * first
        + np.sin(fraction * angle) / np.sin(angle) * second
    )


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float).copy()
    result[..., 1:] *= -1
    return result


def _dvp_quantile_about(center: np.ndarray, fwhm_deg: float,
                        coordinates: np.ndarray) -> np.ndarray:
    exponent = float(dvp_exponent_from_fwhm(fwhm_deg))
    radial, axis_z, axis_phi = np.clip(coordinates, 1e-14, 1 - 1e-14)
    cosine_squared = betaincinv(exponent + 0.5, 1.5, radial)
    angle = 2 * np.arccos(np.sqrt(np.clip(cosine_squared, 0.0, 1.0)))
    z = 2 * axis_z - 1
    azimuth = 2 * np.pi * axis_phi
    axis = np.array([np.sqrt(1 - z * z) * np.cos(azimuth),
                     np.sqrt(1 - z * z) * np.sin(azimuth), z])
    return canonicalize_quaternion(compose_rotations(
        center, axis_angle_to_quaternion(axis, angle)
    ))


def _vmf_cosine_quantile(unit_coordinate: float, concentration: float) -> float:
    """Inverse CDF for the S2 vMF polar cosine without exponential overflow."""
    if concentration <= 1e-8:
        return 2 * unit_coordinate - 1
    return float(1 + np.log(unit_coordinate
                            + (1 - unit_coordinate) * np.exp(-2 * concentration))
                 / concentration)


def _fiber_center(crystal_direction: np.ndarray, sample_direction: np.ndarray,
                  spin_coordinate: float, polar_coordinate: float,
                  azimuth_coordinate: float, concentration: float) -> np.ndarray:
    sample_direction = _unit_vector(sample_direction)
    trial = np.array([1.0, 0.0, 0.0]) if abs(sample_direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_a = _unit_vector(np.cross(sample_direction, trial))
    tangent_b = np.cross(sample_direction, tangent_a)
    cosine = _vmf_cosine_quantile(polar_coordinate, concentration)
    azimuth = 2 * np.pi * azimuth_coordinate
    target = (cosine * sample_direction
              + np.sqrt(max(0.0, 1 - cosine * cosine))
              * (np.cos(azimuth) * tangent_a + np.sin(azimuth) * tangent_b))
    base = _rotation_between_vectors(crystal_direction, target)
    spin = axis_angle_to_quaternion(target, 2 * np.pi * spin_coordinate)
    return canonicalize_quaternion(compose_rotations(spin, base))


def _cubic_anchor_catalog() -> dict[str, np.ndarray]:
    """Canonical FCC rolling/recrystallization component representatives."""
    return {
        "copper_{112}<11-1>": orientation_from_plane_direction([1, 1, 2], [1, 1, -1]),
        "s_{123}<63-4>": orientation_from_plane_direction([1, 2, 3], [6, 3, -4]),
        "brass_{110}<1-12>": orientation_from_plane_direction([1, 1, 0], [1, -1, 2]),
        "goss_{110}<001>": orientation_from_plane_direction([1, 1, 0], [0, 0, 1]),
        "cube_{001}<100>": orientation_from_plane_direction([0, 0, 1], [1, 0, 0]),
    }


def _closest_symmetry_equivalent(reference: np.ndarray, orientation: np.ndarray,
                                 crystal_symmetry: str) -> np.ndarray:
    """Choose the orbit representative on the shortest branch from reference."""
    candidates = compose_rotations(
        np.broadcast_to(orientation, get_symmetry_operations(crystal_symmetry).shape),
        get_symmetry_operations(crystal_symmetry),
    )
    dots = candidates @ reference
    index = int(np.argmax(np.abs(dots)))
    candidate = candidates[index]
    return candidate if dots[index] >= 0 else -candidate


def _pvd_catalog(crystal_symmetry: str, crystal_family: str | None) -> list[dict[str, Any]]:
    """Return PVD/sputter texture templates with a film-normal sample axis.

    The probabilities are conditional on selecting the non-Haar channel:
    13/17 gives an unconditional 0.65 fiber probability after the configured
    0.15 Haar channel, and 4/17 gives an unconditional 0.20 biaxial
    probability.  Biaxial components are deliberately narrow, but no texture
    index rejection is performed; the recorded index remains a diagnostic.
    """
    symmetry = crystal_symmetry.lower()
    family = (crystal_family or symmetry).lower()
    fiber_probability, biaxial_probability = 13 / 17, 4 / 17
    nd = np.array([0.0, 0.0, 1.0])
    if family == "fcc":
        return [
            {"name": "fcc_<111>//film_nd", "kind": "fiber",
             "crystal_direction": np.array([1.0, 1.0, 1.0]),
             "sample_direction": nd, "probability": fiber_probability * 0.75,
             "process_tag": "sputter_fiber"},
            {"name": "fcc_<100>//film_nd", "kind": "fiber",
             "crystal_direction": np.array([1.0, 0.0, 0.0]),
             "sample_direction": nd, "probability": fiber_probability * 0.25,
             "process_tag": "sputter_fiber"},
            {"name": "fcc_(111)[1-10]_biaxial", "kind": "component",
             "center": orientation_from_plane_direction([1, 1, 1], [1, -1, 0]),
             "probability": biaxial_probability * 0.65,
             "fwhm_range_deg": (5.0, 10.0), "process_tag": "sputter_biaxial"},
            {"name": "fcc_(001)[100]_biaxial", "kind": "component",
             "center": orientation_from_plane_direction([0, 0, 1], [1, 0, 0]),
             "probability": biaxial_probability * 0.35,
             "fwhm_range_deg": (5.0, 10.0), "process_tag": "sputter_biaxial"},
        ]
    if family == "bcc":
        return [
            {"name": "bcc_<110>//film_nd", "kind": "fiber",
             "crystal_direction": np.array([1.0, 1.0, 0.0]),
             "sample_direction": nd, "probability": fiber_probability,
             "process_tag": "sputter_fiber"},
            {"name": "bcc_(110)[1-11]_biaxial", "kind": "component",
             "center": orientation_from_plane_direction([1, 1, 0], [1, -1, 1]),
             "probability": biaxial_probability,
             "fwhm_range_deg": (5.0, 10.0), "process_tag": "sputter_biaxial"},
        ]
    if family in {"hcp", "hexagonal"}:
        return [
            {"name": "hcp_<0001>//film_nd", "kind": "fiber",
             "crystal_direction": np.array([0.0, 0.0, 1.0]),
             "sample_direction": nd, "probability": fiber_probability * 0.70,
             "process_tag": "sputter_fiber"},
            {"name": "hcp_prismatic_a//film_nd", "kind": "fiber",
             "crystal_direction": np.array([1.0, 0.0, 0.0]),
             "sample_direction": nd, "probability": fiber_probability * 0.30,
             "process_tag": "sputter_fiber"},
            {"name": "hcp_(0001)[10-10]_biaxial", "kind": "component",
             "center": orientation_from_plane_direction([0, 0, 1], [1, 0, 0]),
             "probability": biaxial_probability * 0.70,
             "fwhm_range_deg": (5.0, 10.0), "process_tag": "sputter_biaxial"},
            {"name": "hcp_prismatic_biaxial", "kind": "component",
             "center": orientation_from_plane_direction([1, 0, 0], [0, 1, 0]),
             "probability": biaxial_probability * 0.30,
             "fwhm_range_deg": (5.0, 10.0), "process_tag": "sputter_biaxial"},
        ]
    raise ValueError("sputter_pvd requires pvd_crystal_family fcc, bcc, or hcp.")


def _catalog_for(crystal_symmetry: str, process_family: str,
                 pvd_crystal_family: str | None = None) -> list[dict[str, Any]]:
    symmetry, process = crystal_symmetry.lower(), process_family.lower()
    if process == "sputter_pvd":
        return _pvd_catalog(symmetry, pvd_crystal_family)
    if symmetry in {"cubic", "fcc", "bcc", "sc"}:
        anchors = _cubic_anchor_catalog()
        entries: dict[str, dict[str, Any]] = {
            name: {"name": name, "kind": "component", "center": center}
            for name, center in anchors.items()
        }
        entries["alpha_fiber_<110>//rd"] = {
            "name": "alpha_fiber_<110>//rd", "kind": "fiber",
            "crystal_direction": np.array([1.0, 1.0, 0.0]),
            "sample_direction": np.array([1.0, 0.0, 0.0]),
        }
        entries["gamma_fiber_<111>//nd"] = {
            "name": "gamma_fiber_<111>//nd", "kind": "fiber",
            "crystal_direction": np.array([1.0, 1.0, 1.0]),
            "sample_direction": np.array([0.0, 0.0, 1.0]),
        }
        entries["build_fiber_<001>//nd"] = {
            "name": "build_fiber_<001>//nd", "kind": "fiber",
            "crystal_direction": np.array([0.0, 0.0, 1.0]),
            "sample_direction": np.array([0.0, 0.0, 1.0]),
        }
        entries["beta_fiber_brass-s-copper"] = {
            "name": "beta_fiber_brass-s-copper", "kind": "beta_fiber",
            "path": [],
        }
        beta_brass = anchors["brass_{110}<1-12>"]
        beta_s = _closest_symmetry_equivalent(
            beta_brass, anchors["s_{123}<63-4>"], "cubic"
        )
        beta_copper = _closest_symmetry_equivalent(
            beta_s, anchors["copper_{112}<11-1>"], "cubic"
        )
        entries["beta_fiber_brass-s-copper"]["path"] = [
            beta_brass, beta_s, beta_copper
        ]
        weights_by_process = {
            "rolling": {
                "copper_{112}<11-1>": 0.15, "s_{123}<63-4>": 0.14,
                "brass_{110}<1-12>": 0.13, "goss_{110}<001>": 0.06,
                "cube_{001}<100>": 0.05, "alpha_fiber_<110>//rd": 0.20,
                "gamma_fiber_<111>//nd": 0.07, "build_fiber_<001>//nd": 0.03,
                "beta_fiber_brass-s-copper": 0.17,
            },
            "recrystallization": {
                "cube_{001}<100>": 0.40, "goss_{110}<001>": 0.23,
                "gamma_fiber_<111>//nd": 0.12, "copper_{112}<11-1>": 0.06,
                "s_{123}<63-4>": 0.04, "brass_{110}<1-12>": 0.04,
                "alpha_fiber_<110>//rd": 0.04, "build_fiber_<001>//nd": 0.05,
                "beta_fiber_brass-s-copper": 0.02,
            },
            "additive_manufacturing": {
                "build_fiber_<001>//nd": 0.44, "gamma_fiber_<111>//nd": 0.22,
                "cube_{001}<100>": 0.14, "goss_{110}<001>": 0.05,
                "copper_{112}<11-1>": 0.03, "s_{123}<63-4>": 0.02,
                "brass_{110}<1-12>": 0.02, "alpha_fiber_<110>//rd": 0.03,
                "beta_fiber_brass-s-copper": 0.05,
            },
        }
        broad = {name: 1 / len(entries) for name in entries}
        weights = weights_by_process.get(process, broad)
        return [{**entries[name], "probability": probability}
                for name, probability in weights.items()]

    # For hexagonal and lower-symmetry examples, use explicit crystallographic
    # axis-to-processing-axis relationships.  These are generic manifolds, not
    # alloy-specific priors; calibrated projects should override their mixture.
    if symmetry in {"hexagonal", "hcp"}:
        return [
            {"name": "basal_<0001>//nd", "kind": "fiber",
             "crystal_direction": np.array([0.0, 0.0, 1.0]),
             "sample_direction": np.array([0.0, 0.0, 1.0]), "probability": 0.50},
            {"name": "basal_<0001>//rd", "kind": "fiber",
             "crystal_direction": np.array([0.0, 0.0, 1.0]),
             "sample_direction": np.array([1.0, 0.0, 0.0]), "probability": 0.20},
            {"name": "prismatic_<10-10>//rd", "kind": "fiber",
             "crystal_direction": np.array([1.0, 0.0, 0.0]),
             "sample_direction": np.array([1.0, 0.0, 0.0]), "probability": 0.30},
        ]
    return [
        {"name": "crystal_z_axis//nd", "kind": "fiber",
         "crystal_direction": np.array([0.0, 0.0, 1.0]),
         "sample_direction": np.array([0.0, 0.0, 1.0]), "probability": 0.35},
        {"name": "crystal_y_axis//td", "kind": "fiber",
         "crystal_direction": np.array([0.0, 1.0, 0.0]),
         "sample_direction": np.array([0.0, 1.0, 0.0]), "probability": 0.25},
        {"name": "crystal_x_axis//rd", "kind": "fiber",
         "crystal_direction": np.array([1.0, 0.0, 0.0]),
         "sample_direction": np.array([1.0, 0.0, 0.0]), "probability": 0.25},
        {"name": "crystal_xyz//rd-td-nd", "kind": "component",
         "center": np.array([1.0, 0.0, 0.0, 0.0]), "probability": 0.15},
    ]


@dataclass(frozen=True)
class PhysicalTextureSpace:
    """A Sobol-driven physical manifold mapped to an exact mixture descriptor.

    The hypercube remains useful as a reproducible low-discrepancy *source* of
    randomness, but its center coordinates select and perturb material
    manifolds instead of being interpreted as independent Haar rotations.
    A nonzero ``uniform_center_fraction`` is the explicit OOD escape channel.

    ``component_count_range`` permits a fixed-width descriptor to reserve
    inactive zero-weight component slots.  This keeps one tensor shape per
    corpus while allowing a 1--M component ODF family.  The active-slot mask
    in :meth:`decode` is authoritative; inactive centers and widths are not
    part of the represented ODF.
    """

    crystal_symmetry: str
    process_family: str = "broad"
    inference_mode: str = "likelihood"
    max_components: int = 5
    fwhm_strata_deg: tuple[tuple[float, float], ...] = ((8.0, 18.0), (18.0, 32.0), (32.0, 55.0))
    fwhm_strata_probabilities: tuple[float, ...] = (0.25, 0.45, 0.30)
    background_range: tuple[float, float] = (0.01, 0.95)
    background_beta: tuple[float, float] = (1.4, 1.4)
    dirichlet_concentration_range: tuple[float, float] = (0.12, 1.2)
    uniform_center_fraction: float = 0.25
    component_center_fwhm_deg: float = 14.0
    fiber_concentration: float = 45.0
    fiber_tube_fwhm_deg: float = 12.0
    component_count_range: tuple[int, int] | None = None
    pvd_crystal_family: str | None = None
    hybrid_process_probabilities: tuple[tuple[str, float], ...] = (
        ("rolling", 1 / 3),
        ("recrystallization", 1 / 3),
        ("additive_manufacturing", 1 / 3),
    )

    @property
    def dimension(self) -> int:
        # Background, Dirichlet concentration, and one five-coordinate block
        # per allocated component.  A variable active-count family receives an
        # independent Sobol coordinate so its count is not coupled to weight
        # concentration or an orientation coordinate.
        return (2 + self.max_components * 5
                + int(self.process_family == "hybrid_multi_domain")
                + int(self.component_count_range is not None))

    def __post_init__(self):
        if self.max_components < 1:
            raise ValueError("max_components must be positive.")
        if self.process_family not in {
            "broad", "rolling", "recrystallization", "additive_manufacturing",
            "hybrid_multi_domain", "sputter_pvd",
        }:
            raise ValueError(
                "process_family must be broad, rolling, recrystallization, "
                "additive_manufacturing, hybrid_multi_domain, or sputter_pvd."
            )
        if self.process_family == "hybrid_multi_domain" and self.crystal_symmetry.lower() not in {
            "cubic", "fcc", "bcc", "sc"
        }:
            raise ValueError("hybrid_multi_domain is currently defined for cubic texture catalogs.")
        if self.process_family == "sputter_pvd":
            family = (self.pvd_crystal_family or self.crystal_symmetry).lower()
            if family not in {"fcc", "bcc", "hcp", "hexagonal"}:
                raise ValueError("sputter_pvd requires pvd_crystal_family fcc, bcc, or hcp.")
            if family in {"fcc", "bcc"} and self.crystal_symmetry.lower() not in {"cubic", "fcc", "bcc"}:
                raise ValueError("FCC/BCC PVD families require cubic orientation symmetry.")
            if family in {"hcp", "hexagonal"} and self.crystal_symmetry.lower() not in {"hcp", "hexagonal"}:
                raise ValueError("HCP PVD families require hexagonal orientation symmetry.")
        if self.inference_mode not in {"likelihood", "broad_likelihood", "physical_prior"}:
            raise ValueError(
                "inference_mode must be 'likelihood', 'broad_likelihood', or 'physical_prior'."
            )
        if not 0 <= self.uniform_center_fraction <= 1:
            raise ValueError("uniform_center_fraction must lie in [0, 1].")
        b0, b1 = self.background_range
        if not 0 <= b0 <= b1 <= 1:
            raise ValueError("background_range must lie in [0, 1].")
        if any(value <= 0 for value in self.background_beta):
            raise ValueError("background_beta parameters must be positive.")
        a0, a1 = self.dirichlet_concentration_range
        if not 0 < a0 <= a1:
            raise ValueError("Dirichlet concentration range must be positive.")
        if self.component_count_range is not None:
            minimum, maximum = map(int, self.component_count_range)
            if not 1 <= minimum <= maximum <= self.max_components:
                raise ValueError(
                    "component_count_range must satisfy 1 <= minimum <= maximum <= max_components."
                )
        probabilities = dict(self.hybrid_process_probabilities)
        allowed_domains = {"rolling", "recrystallization", "additive_manufacturing"}
        if set(probabilities) - allowed_domains or any(value <= 0 for value in probabilities.values()):
            raise ValueError("hybrid_process_probabilities must use positive rolling/recrystallization/additive_manufacturing entries.")
        if not np.isclose(sum(probabilities.values()), 1.0):
            raise ValueError("hybrid_process_probabilities must sum to one.")
        if len(self.fwhm_strata_deg) != len(self.fwhm_strata_probabilities):
            raise ValueError("Each FWHM stratum requires one probability.")
        if not np.isclose(sum(self.fwhm_strata_probabilities), 1.0):
            raise ValueError("FWHM stratum probabilities must sum to one.")
        for low, high in self.fwhm_strata_deg:
            if not 0 < low <= high < 180:
                raise ValueError("All FWHM strata must lie in (0, 180).")
        if not 0 < self.component_center_fwhm_deg < 180:
            raise ValueError("component_center_fwhm_deg must lie in (0, 180).")
        if self.fiber_concentration <= 0 or not 0 < self.fiber_tube_fwhm_deg < 180:
            raise ValueError("Fiber concentration and tube FWHM must be positive and finite.")
        get_symmetry_operations(self.crystal_symmetry)

    def _background(self, coordinate: float) -> float:
        quantile = betaincinv(self.background_beta[0], self.background_beta[1],
                              np.clip(coordinate, 1e-14, 1 - 1e-14))
        return float(self.background_range[0]
                     + quantile * (self.background_range[1] - self.background_range[0]))

    def _spread(self, coordinate: float) -> float:
        probabilities = np.asarray(self.fwhm_strata_probabilities, dtype=float)
        cumulative = np.cumsum(probabilities)
        index = min(int(np.searchsorted(cumulative, coordinate, side="right")),
                    len(probabilities) - 1)
        lower_probability = 0.0 if index == 0 else cumulative[index - 1]
        local = (coordinate - lower_probability) / probabilities[index]
        low, high = self.fwhm_strata_deg[index]
        # A log interpolation resolves physically important narrow widths more
        # evenly than a linear map while keeping exact configured endpoints.
        return float(np.exp(np.log(low) + local * np.log(high / low)))

    def _active_component_count(self, coordinate: float) -> int:
        if self.component_count_range is None:
            return self.max_components
        minimum, maximum = map(int, self.component_count_range)
        return min(minimum + int(coordinate * (maximum - minimum + 1)), maximum)

    def _hybrid_process_domain(self, coordinate: float) -> str:
        probabilities = dict(self.hybrid_process_probabilities)
        domains = list(probabilities)
        cumulative = np.cumsum([probabilities[domain] for domain in domains])
        index = min(int(np.searchsorted(cumulative, coordinate, side="right")), len(domains) - 1)
        return domains[index]

    def _spread_over_range(self, coordinate: float,
                           width_range: tuple[float, float] | None) -> float:
        if width_range is None:
            return self._spread(coordinate)
        low, high = map(float, width_range)
        if not 0 < low <= high < 180:
            raise ValueError("A component-specific FWHM range must lie in (0, 180).")
        return float(np.exp(np.log(low) + coordinate * np.log(high / low)))

    def _catalog_choice(self, coordinates: np.ndarray,
                        process_domain: str | None = None) -> tuple[np.ndarray, str, str, str, tuple[float, float] | None]:
        selector, second, third = map(float, coordinates)
        if selector < self.uniform_center_fraction:
            local = selector / max(self.uniform_center_fraction, _TINY)
            uniform_tag = (
                "nanocrystalline_uniform" if self.process_family == "sputter_pvd"
                else "haar_uniform"
            )
            return (
                sobol_quaternions(np.array([local, second, third])),
                "uniform_so3_escape", "uniform", uniform_tag, None,
            )

        physical_selector = ((selector - self.uniform_center_fraction)
                             / max(1 - self.uniform_center_fraction, _TINY))
        process_tag = self.process_family
        if self.process_family == "hybrid_multi_domain":
            if process_domain is None:
                raise ValueError("hybrid_multi_domain requires an ODF-level process domain.")
            process_tag = process_domain
            catalog = _catalog_for(self.crystal_symmetry, process_tag, self.pvd_crystal_family)
        else:
            catalog = _catalog_for(
                self.crystal_symmetry, self.process_family, self.pvd_crystal_family
            )
        probabilities = np.asarray([entry["probability"] for entry in catalog], dtype=float)
        probabilities /= probabilities.sum()
        cumulative = np.cumsum(probabilities)
        index = min(int(np.searchsorted(cumulative, physical_selector, side="right")),
                    len(catalog) - 1)
        lower = 0.0 if index == 0 else cumulative[index - 1]
        local = (physical_selector - lower) / probabilities[index]
        entry = catalog[index]
        kind = entry["kind"]
        if kind == "component":
            center = _dvp_quantile_about(
                entry["center"], self.component_center_fwhm_deg,
                np.array([local, second, third]),
            )
        elif kind == "fiber":
            center = _fiber_center(
                entry["crystal_direction"], entry["sample_direction"],
                local, second, third, self.fiber_concentration,
            )
        elif kind == "beta_fiber":
            scaled = min(2 * local, 2 - 4 * np.finfo(float).eps)
            segment = min(int(scaled), 1)
            fraction = scaled - segment
            first, last = entry["path"][segment:segment + 2]
            center = _quaternion_slerp(first, last, fraction)
            relative = compose_rotations(_quaternion_conjugate(first), last)
            tangent = _unit_vector(relative[1:])
            trial = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            normal_a = _unit_vector(np.cross(tangent, trial))
            normal_b = np.cross(tangent, normal_a)
            sigma = np.deg2rad(self.fiber_tube_fwhm_deg) / (2 * np.sqrt(2 * np.log(2)))
            radius = sigma * np.sqrt(-2 * np.log(max(1 - second, 1e-14)))
            radius = min(radius, 3.5 * sigma)
            axis = np.cos(2 * np.pi * third) * normal_a + np.sin(2 * np.pi * third) * normal_b
            center = compose_rotations(center, axis_angle_to_quaternion(axis, radius))
        else:
            raise RuntimeError(f"Unsupported physical manifold kind {kind!r}.")
        return (
            center, entry["name"], kind,
            entry.get("process_tag", process_tag), entry.get("fwhm_range_deg"),
        )

    def decode(self, unit_cube: np.ndarray) -> tuple[ContinuousSymmetricMixtureODF, dict[str, Any]]:
        u = np.asarray(unit_cube, dtype=float)
        if u.shape != (self.dimension,) or np.any((u < 0) | (u > 1)):
            raise ValueError(f"Expected one Sobol point of shape ({self.dimension},).")
        background = self._background(float(u[0]))
        alpha_lo, alpha_hi = self.dirichlet_concentration_range
        alpha = float(np.exp(np.log(alpha_lo) + u[1] * np.log(alpha_hi / alpha_lo)))
        centers, names, kinds, process_tags, fwhm, gamma_quantiles = [], [], [], [], [], []
        offset = 2
        sample_process_domain = (
            self._hybrid_process_domain(float(u[offset]))
            if self.process_family == "hybrid_multi_domain" else self.process_family
        )
        offset += int(self.process_family == "hybrid_multi_domain")
        for _ in range(self.max_components):
            center, name, kind, process_tag, component_width_range = self._catalog_choice(
                u[offset:offset + 3], sample_process_domain
            )
            centers.append(reduce_to_fundamental_zone(center, self.crystal_symmetry))
            names.append(name)
            kinds.append(kind)
            process_tags.append(process_tag)
            fwhm.append(self._spread_over_range(float(u[offset + 3]), component_width_range))
            gamma_quantiles.append(float(np.clip(u[offset + 4], 1e-15, 1 - 1e-15)))
            offset += 5
        active_count = self._active_component_count(float(u[offset])) if self.component_count_range else self.max_components
        raw = gammaincinv(alpha, np.asarray(gamma_quantiles[:active_count]))
        weights = np.zeros(self.max_components, dtype=float)
        weights[:active_count] = (1 - background) * raw / raw.sum()
        order = np.lexsort((np.asarray(fwhm), -weights))
        odf = ContinuousSymmetricMixtureODF(
            np.asarray(centers)[order], np.asarray(fwhm)[order], weights[order],
            background, self.crystal_symmetry,
        )
        metadata = {
            "sampling_space": (
                "physical_texture_hybrid_v2"
                if self.process_family == "hybrid_multi_domain" or self.component_count_range
                else "physical_texture_manifold_v1"
            ),
            "inference_mode": self.inference_mode,
            "process_family": self.process_family,
            "sample_process_domain": sample_process_domain,
            "component_manifold": np.asarray(names)[order].tolist(),
            "component_manifold_kind": np.asarray(kinds)[order].tolist(),
            "component_process_tag": np.asarray(process_tags)[order].tolist(),
            "active_component_count": active_count,
            "component_active_mask": (weights[order] > 0).tolist(),
            "dirichlet_concentration": alpha,
        }
        return odf, metadata

    def from_unit_cube(self, unit_cube: np.ndarray) -> ContinuousSymmetricMixtureODF:
        return self.decode(unit_cube)[0]

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
