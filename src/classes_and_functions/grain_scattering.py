"""Per-grain shape, broadening, and integrated-intensity calculations.

The grain size is an equivalent-volume spherical diameter in the same length
unit as wavelength and lattice parameter.  ``aspect_ratio`` is the ellipsoid
long-axis/transverse-axis ratio; its long axis is crystal ``[001]`` and is
therefore carried to the sample frame by the grain orientation.  The model is
deliberately compact: Scherrer size broadening plus the project's existing
simple microstrain term are combined in quadrature.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .orientation import quaternion_to_rotation_matrix

FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def ellipsoid_axes(equivalent_diameter: float, aspect_ratio: float = 1.0) -> np.ndarray:
    """Return volume-preserving ellipsoid semi-axes ``[a, a, c]``."""
    if equivalent_diameter <= 0 or aspect_ratio <= 0:
        raise ValueError("Grain diameter and aspect ratio must be positive.")
    transverse = equivalent_diameter / (2.0 * aspect_ratio ** (1.0 / 3.0))
    return np.array([transverse, transverse, transverse * aspect_ratio])


def ellipsoid_volume(equivalent_diameter: float) -> float:
    """Volume of the equivalent sphere; invariant to the aspect ratio."""
    return np.pi * equivalent_diameter**3 / 6.0


def coherent_length(grain, reciprocal_vector_sample: np.ndarray) -> float:
    """Central ellipsoid chord length parallel to a reciprocal-lattice vector."""
    g = np.asarray(reciprocal_vector_sample, dtype=float)
    norm = np.linalg.norm(g)
    if norm == 0:
        return np.inf
    # Shape axes are defined in crystal coordinates, as is the [001] long axis.
    direction_crystal = quaternion_to_rotation_matrix(grain.orientation).T @ (g / norm)
    axes = ellipsoid_axes(grain.grain_size, grain.aspect_ratio)
    return 2.0 / np.sqrt(np.sum((direction_crystal / axes) ** 2))


@dataclass(frozen=True)
class PeakProperties:
    integrated_intensity: float
    fwhm_2theta_rad: float
    fwhm_radial_px: float
    fwhm_tangential_px: float
    coherent_length: float
    volume: float


def peak_properties(grain, reciprocal_vector_sample, two_theta, wavelength,
                    detector_distance, pixel_scale=1.0, shape_factor=0.9,
                    instrumental_fwhm_px=4.709640090061899,
                    intensity_scale=1.0) -> PeakProperties:
    """Compute per-reflection width and integrated intensity.

    ``instrumental_fwhm_px`` is a Gaussian detector contribution (default
    sigma=2 px, matching the old renderer).  Size broadening is the Scherrer
    FWHM ``K λ/(L cos θ)``.  Strain keeps the legacy simplified FWHM
    ``|epsilon| tan θ``.  The returned intensity excludes the structure factor
    and is scaled by grain volume relative to the configured reference grain.
    """
    theta = 0.5 * two_theta
    length = coherent_length(grain, reciprocal_vector_sample)
    size_fwhm = shape_factor * wavelength / max(length * np.cos(theta), np.finfo(float).eps)
    strain_fwhm = abs(getattr(grain, "grain_strain", 0.0)) * abs(np.tan(theta))
    angular_fwhm = np.hypot(size_fwhm, strain_fwhm)
    # r = D tan(2theta), so dr/d(2theta) = D sec^2(2theta).
    radial_sample_fwhm = angular_fwhm * detector_distance / np.cos(two_theta)**2
    radial_px = np.hypot(instrumental_fwhm_px, radial_sample_fwhm * pixel_scale)
    volume = ellipsoid_volume(grain.grain_size)
    reference_volume = getattr(grain, "reference_volume", ellipsoid_volume(grain.size_average))
    relative_volume = volume / reference_volume
    lorentz = 1.0 / max(abs(np.sin(two_theta)), 1e-12)
    polarization = (1.0 + np.cos(two_theta)**2) / 2.0
    return PeakProperties(float(intensity_scale * relative_volume * lorentz * polarization),
                          float(angular_fwhm), float(radial_px),
                          float(instrumental_fwhm_px), float(length), float(volume))
