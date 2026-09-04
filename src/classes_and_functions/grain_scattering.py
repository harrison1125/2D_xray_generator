"""Physical grain-volume, projection, and line-profile calculations.

The implementation follows the compatible parts of ``fable_xrdsim``: grain
volume is expressed in cubic micrometres, Scherrer broadening is a reciprocal-
space (Lorentzian) width, and real-space grain morphology is projected onto the
detector. The local model uses a volume-equivalent ellipsoid in place of
fable's tetrahedral mesh.
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


def projected_ellipsoid_covariance_px(grain, outgoing_direction, pixel_size) -> np.ndarray:
    """Return detector ``[row(z), col(y)]`` covariance for a uniform ellipsoid.

    The detector is the plane ``x=D``. Rays from every point in a grain travel
    in the same outgoing direction, so the exact source-to-plane Jacobian is
    used. A uniform solid ellipsoid with semi-axis ``a`` has variance ``a²/5``.
    This is a second-moment approximation to fable's path-length volume
    projection and is exact at the covariance level.
    """
    direction = np.asarray(outgoing_direction, dtype=float)
    if pixel_size is None or pixel_size <= 0 or abs(direction[0]) <= np.finfo(float).eps:
        return np.zeros((2, 2), dtype=float)
    axes = ellipsoid_axes(grain.grain_size, grain.aspect_ratio)
    rotation = quaternion_to_rotation_matrix(grain.orientation)
    covariance_sample = rotation @ np.diag(axes**2 / 5.0) @ rotation.T
    # row=z_s + (D-x_s) k_z/k_x; col=y_s + (D-x_s) k_y/k_x
    jacobian = np.array([
        [-direction[2] / direction[0], 0.0, 1.0],
        [-direction[1] / direction[0], 1.0, 0.0],
    ])
    covariance = jacobian @ covariance_sample @ jacobian.T / pixel_size**2
    return 0.5 * (covariance + covariance.T)


def voigt_fwhm(lorentzian_fwhm: float, gaussian_fwhm: float) -> float:
    """Accurate closed-form approximation to a Voigt profile's FWHM."""
    return (0.5346 * lorentzian_fwhm
            + np.sqrt(0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2))


@dataclass(frozen=True)
class PeakProperties:
    integrated_intensity: float
    fwhm_2theta_rad: float
    fwhm_radial_px: float
    fwhm_tangential_px: float
    coherent_length: float
    volume: float
    volume_um3: float
    size_fwhm_2theta_rad: float
    strain_fwhm_2theta_rad: float
    size_fwhm_radial_px: float
    gaussian_sigma_radial_px: float
    gaussian_sigma_tangential_px: float
    projected_sigma_radial_px: float
    projected_sigma_tangential_px: float
    excitation_error: float
    excitation_weight: float
    convergence_reciprocal_half_span: float


def convergence_averaged_excitation_weight(
    excitation_error: float,
    reciprocal_fwhm: float,
    two_theta: float,
    wavelength: float,
    full_convergence_angle_mrad: float = 0.0,
) -> tuple[float, float]:
    """Average a finite-size rocking curve over uniform incident-ray angles.

    The configured value is the full range, so 3.5 mrad integrates uniformly
    from -1.75 to +1.75 mrad in the reflection's scattering plane. For a small
    incident-ray tilt ``delta``, the radial Ewald mismatch changes by
    ``sin(2 theta) * delta / wavelength``. The normalized analytic integral
    conserves fixed total incident flux. Zero convergence exactly recovers the
    original finite-size-only Lorentzian excitation weight.
    """
    error = abs(float(excitation_error))
    width = float(reciprocal_fwhm)
    full_angle = float(full_convergence_angle_mrad)
    if width <= 0 or wavelength <= 0 or full_angle < 0:
        raise ValueError(
            "Reciprocal width/wavelength must be positive and convergence nonnegative."
        )
    point_weight = 1.0 / (1.0 + (2.0 * error / width) ** 2)
    if full_angle == 0:
        return float(point_weight), 0.0

    half_angle_rad = 0.5 * full_angle * 1e-3
    slope = abs(np.sin(two_theta)) / wavelength
    half_span = slope * half_angle_rad
    if half_span <= np.finfo(float).eps:
        return float(point_weight), float(half_span)

    upper = 2.0 * (error + half_span) / width
    lower = 2.0 * (error - half_span) / width
    averaged = width * (np.arctan(upper) - np.arctan(lower)) / (4.0 * half_span)
    return float(np.clip(averaged, 0.0, 1.0)), float(half_span)


def peak_properties(grain, reciprocal_vector_sample, two_theta, wavelength,
                    detector_distance, pixel_scale=1.0, shape_factor=0.9,
                    instrumental_fwhm_px=4.709640090061899,
                    intensity_scale=1.0, outgoing_direction=None,
                    radial_direction=None, pixel_size=None,
                    excitation_error=0.0, lorentz_model="none",
                    incident_convergence_full_angle_mrad=0.0) -> PeakProperties:
    """Compute per-reflection width and integrated intensity.

    Size broadening is the Lorentzian Scherrer FWHM ``K λ/(L cos θ)``.
    Microstrain is the Gaussian Williamson-Hall FWHM ``4 |epsilon| tan θ``.
    Detector PSF, strain, and the projected real-space ellipsoid are combined
    as Gaussian variances; the radial total is therefore a Voigt profile.

    Integrated intensity is relative (absolute beam flux and detector response
    are not modeled), but scales with illuminated grain volume in µm³. A
    Lorentzian excitation-error weight makes finite grain size affect whether a
    reciprocal point contributes in a static shot. When convergence is
    configured, that rocking curve is averaged over the incident angular
    interval. ``lorentz_model='legacy'`` retains the old scan-like
    ``1/sin(2theta)`` factor for comparisons only.
    """
    theta = 0.5 * two_theta
    length = coherent_length(grain, reciprocal_vector_sample)
    size_fwhm = shape_factor * wavelength / max(length * np.cos(theta), np.finfo(float).eps)
    strain_fwhm = 4.0 * abs(getattr(grain, "grain_strain", 0.0)) * abs(np.tan(theta))
    angular_fwhm = voigt_fwhm(size_fwhm, strain_fwhm)
    # r = D tan(2theta), so dr/d(2theta) = D sec^2(2theta).
    angular_to_px = detector_distance * pixel_scale / np.cos(two_theta)**2
    size_fwhm_px = size_fwhm * angular_to_px
    strain_fwhm_px = strain_fwhm * angular_to_px

    projected_covariance = projected_ellipsoid_covariance_px(
        grain, outgoing_direction, pixel_size
    ) if outgoing_direction is not None else np.zeros((2, 2), dtype=float)
    radial = np.asarray([1.0, 0.0] if radial_direction is None else radial_direction,
                        dtype=float)
    radial_norm = np.linalg.norm(radial)
    radial = radial / radial_norm if radial_norm else np.array([1.0, 0.0])
    tangential = np.array([-radial[1], radial[0]])
    projected_radial_sigma = np.sqrt(max(0.0, radial @ projected_covariance @ radial))
    projected_tangential_sigma = np.sqrt(max(0.0, tangential @ projected_covariance @ tangential))

    instrumental_sigma = instrumental_fwhm_px / FWHM_TO_SIGMA
    radial_gaussian_sigma = np.sqrt(
        instrumental_sigma**2
        + (strain_fwhm_px / FWHM_TO_SIGMA)**2
        + projected_radial_sigma**2
    )
    tangential_gaussian_sigma = np.sqrt(
        instrumental_sigma**2 + projected_tangential_sigma**2
    )
    radial_px = voigt_fwhm(size_fwhm_px, radial_gaussian_sigma * FWHM_TO_SIGMA)
    tangential_px = tangential_gaussian_sigma * FWHM_TO_SIGMA

    volume = ellipsoid_volume(grain.grain_size)
    length_to_microns = getattr(grain, "length_to_microns", 1e-3)
    volume_um3 = ellipsoid_volume(grain.grain_size * length_to_microns)
    reciprocal_fwhm = shape_factor / max(length, np.finfo(float).eps)
    excitation_weight, convergence_half_span = convergence_averaged_excitation_weight(
        excitation_error,
        reciprocal_fwhm,
        two_theta,
        wavelength,
        incident_convergence_full_angle_mrad,
    )
    if lorentz_model == "none":
        lorentz = 1.0
    elif lorentz_model == "legacy":
        lorentz = 1.0 / max(abs(np.sin(two_theta)), 1e-12)
    else:
        raise ValueError("lorentz_model must be 'none' or 'legacy'.")
    polarization = (1.0 + np.cos(two_theta)**2) / 2.0
    return PeakProperties(
        float(intensity_scale * volume_um3 * excitation_weight * lorentz * polarization),
        float(angular_fwhm), float(radial_px), float(tangential_px),
        float(length), float(volume), float(volume_um3), float(size_fwhm),
        float(strain_fwhm), float(size_fwhm_px), float(radial_gaussian_sigma),
        float(tangential_gaussian_sigma), float(projected_radial_sigma),
        float(projected_tangential_sigma), float(excitation_error),
        float(excitation_weight), float(convergence_half_span),
    )
