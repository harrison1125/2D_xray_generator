"""Planar detector projection and normalized per-grain spot rendering."""
from __future__ import annotations

import numpy as np
from scipy.special import voigt_profile

from .grain_scattering import peak_properties


class Detector:
    def __init__(self, ewald_sphere, experiment, detector_width, detector_height,
                 detector_distance, structure_factor_func, shape_factor=.9,
                 instrumental_fwhm_px=4.709640090061899, intensity_scale=1.0,
                 pixel_size_grain_units=None, profile="voigt",
                 lorentz_model="none", incident_convergence_full_angle_mrad=0.0):
        self.ewald_sphere, self.experiment = ewald_sphere, experiment
        self.detector_width, self.detector_height = detector_width, detector_height
        self.detector_distance = detector_distance
        self.structure_factor_func = structure_factor_func
        self.shape_factor = shape_factor
        self.instrumental_fwhm_px = instrumental_fwhm_px
        self.intensity_scale = intensity_scale
        self.pixel_size_grain_units = pixel_size_grain_units
        self.profile = profile
        self.lorentz_model = lorentz_model
        self.incident_convergence_full_angle_mrad = incident_convergence_full_angle_mrad

    def _deposit_peak(self, image, row, col, radial, intensity, properties):
        """Deposit a radial Voigt × tangential Gaussian without edge renormalization."""
        if intensity <= 0:
            return
        radial = np.asarray(radial, dtype=float)
        if np.linalg.norm(radial) == 0:
            radial = np.array([1.0, 0.0])
        radial /= np.linalg.norm(radial)
        tangential = np.array([-radial[1], radial[0]])
        sigma_radial = max(properties.gaussian_sigma_radial_px, np.finfo(float).eps)
        sigma_tangential = max(properties.gaussian_sigma_tangential_px, np.finfo(float).eps)
        gamma = max(0.0, properties.size_fwhm_radial_px / 2.0)

        # ±32 Lorentz HWHM contains about 98% of a pure Lorentz profile; the
        # Gaussian support extends beyond 4.5 sigma. Detector dimensions cap
        # pathological kernels while preserving normal experimental widths.
        radius = max(2, int(np.ceil(max(4.5 * sigma_radial,
                                     4.5 * sigma_tangential, 32.0 * gamma))))
        radius = min(radius, max(image.shape))
        row_center, col_center = int(np.rint(row)), int(np.rint(col))
        row_offsets = np.arange(-radius, radius + 1)
        col_offsets = np.arange(-radius, radius + 1)
        rows, cols = np.meshgrid(row_center + row_offsets,
                                 col_center + col_offsets, indexing="ij")
        offset = np.stack((rows - row, cols - col), axis=-1)
        radial_coordinate = offset @ radial
        tangential_coordinate = offset @ tangential

        if self.profile == "voigt":
            radial_profile = voigt_profile(radial_coordinate, sigma_radial, gamma)
        elif self.profile == "gaussian":
            total_sigma = max(properties.fwhm_radial_px / 2.3548200450309493,
                              np.finfo(float).eps)
            radial_profile = np.exp(-0.5 * (radial_coordinate / total_sigma)**2)
        else:
            raise ValueError("profile must be 'voigt' or 'gaussian'.")
        tangential_profile = np.exp(-0.5 * (tangential_coordinate / sigma_tangential)**2)
        kernel = radial_profile * tangential_profile
        kernel_sum = kernel.sum()
        if not np.isfinite(kernel_sum) or kernel_sum <= 0:
            return
        kernel /= kernel_sum

        valid = ((rows >= 0) & (rows < image.shape[0])
                 & (cols >= 0) & (cols < image.shape[1]))
        image[rows[valid], cols[valid]] += intensity * kernel[valid]

    def project_points(self, image=None, *, store_peaks=True):
        """Project candidates into an optional shared detector accumulation array."""
        if image is None:
            image = np.zeros((int(self.detector_height), int(self.detector_width)))
        elif image.shape != (int(self.detector_height), int(self.detector_width)):
            raise ValueError("Shared detector image shape does not match detector dimensions.")
        coordinates, two_theta_values, records = [], [], []
        center = np.asarray(self.ewald_sphere.center, dtype=float)
        scale_y = image.shape[1] / self.detector_width
        scale_z = image.shape[0] / self.detector_height
        for point in self.ewald_sphere.filter_points():
            reciprocal = np.asarray(point[:3], dtype=float)
            if np.linalg.norm(reciprocal) == 0:
                continue
            # k_out = k_in + G. The Ewald-sphere center is -k_in, but it is
            # not a real-space ray source. Project the outgoing ray from the
            # sample origin, matching fable_xrdsim's ray/plane intersection.
            outgoing = reciprocal - center
            if outgoing[0] <= 0:
                continue
            projected = np.array([
                self.detector_distance,
                self.detector_distance * outgoing[1] / outgoing[0],
                self.detector_distance * outgoing[2] / outgoing[0],
            ])
            if not (-self.detector_width / 2 <= projected[1] <= self.detector_width / 2
                    and -self.detector_height / 2 <= projected[2] <= self.detector_height / 2):
                continue
            row = projected[2] * scale_z + image.shape[0] / 2
            col = projected[1] * scale_y + image.shape[1] / 2
            if not (0 <= row < image.shape[0] and 0 <= col < image.shape[1]):
                continue
            two_theta = np.arctan2(np.hypot(projected[1], projected[2]),
                                   self.detector_distance)
            radial = np.array([row - image.shape[0] / 2,
                               col - image.shape[1] / 2])
            excitation_error = float(point[6]) if len(point) >= 7 else 0.0
            properties = peak_properties(
                self.ewald_sphere.grain, reciprocal, two_theta,
                self.experiment.wavelength, self.detector_distance,
                pixel_scale=(scale_y + scale_z) / 2,
                shape_factor=self.shape_factor,
                instrumental_fwhm_px=self.instrumental_fwhm_px,
                intensity_scale=self.intensity_scale,
                outgoing_direction=outgoing,
                radial_direction=radial,
                pixel_size=self.pixel_size_grain_units,
                excitation_error=excitation_error,
                lorentz_model=self.lorentz_model,
                incident_convergence_full_angle_mrad=(
                    self.incident_convergence_full_angle_mrad
                ),
            )
            hkl = tuple(map(int, point[3:6])) if len(point) >= 6 else None
            structure_intensity = (1.0 if self.structure_factor_func is None
                                   else abs(self.structure_factor_func(*hkl, 1))**2)
            intensity = properties.integrated_intensity * structure_intensity
            self._deposit_peak(image, row, col, radial, intensity, properties)
            coordinates.append(projected)
            two_theta_values.append(two_theta)
            if store_peaks:
                records.append({
                    "hkl": hkl,
                    "two_theta": two_theta,
                    "integrated_intensity": intensity,
                    "fwhm_2theta_rad": properties.fwhm_2theta_rad,
                    "size_fwhm_2theta_rad": properties.size_fwhm_2theta_rad,
                    "strain_fwhm_2theta_rad": properties.strain_fwhm_2theta_rad,
                    "fwhm_radial_px": properties.fwhm_radial_px,
                    "fwhm_tangential_px": properties.fwhm_tangential_px,
                    "size_fwhm_radial_px": properties.size_fwhm_radial_px,
                    "projected_sigma_radial_px": properties.projected_sigma_radial_px,
                    "projected_sigma_tangential_px": properties.projected_sigma_tangential_px,
                    "coherent_length": properties.coherent_length,
                    "grain_volume": properties.volume,
                    "grain_volume_um3": properties.volume_um3,
                    "excitation_error": properties.excitation_error,
                    "excitation_weight": properties.excitation_weight,
                    "convergence_reciprocal_half_span": (
                        properties.convergence_reciprocal_half_span
                    ),
                })
        return {
            "coordinate": np.asarray(coordinates),
            "two_theta": np.asarray(two_theta_values),
            "image": image,
            "peaks": records,
            "num_peaks": len(two_theta_values),
        }
