"""Detector projection and normalized, per-grain diffraction-spot rendering."""
from __future__ import annotations
import numpy as np
from .grain_scattering import FWHM_TO_SIGMA, peak_properties


class Detector:
    def __init__(self, ewald_sphere, experiment, detector_width, detector_height, detector_distance,
                 structure_factor_func, shape_factor=.9, instrumental_fwhm_px=4.709640090061899,
                 intensity_scale=1.0):
        self.ewald_sphere, self.experiment = ewald_sphere, experiment
        self.detector_width, self.detector_height = detector_width, detector_height
        self.detector_distance, self.structure_factor_func = detector_distance, structure_factor_func
        self.shape_factor, self.instrumental_fwhm_px, self.intensity_scale = shape_factor, instrumental_fwhm_px, intensity_scale

    def _deposit_peak(self, image, row, col, radial, intensity, radial_fwhm, tangential_fwhm):
        sigma_radial, sigma_tangential = radial_fwhm / FWHM_TO_SIGMA, tangential_fwhm / FWHM_TO_SIGMA
        radius = int(np.ceil(3 * max(sigma_radial, sigma_tangential)))
        row_min, row_max = max(0, row - radius), min(image.shape[0], row + radius + 1)
        col_min, col_max = max(0, col - radius), min(image.shape[1], col + radius + 1)
        rows, cols = np.meshgrid(np.arange(row_min, row_max), np.arange(col_min, col_max), indexing="ij")
        if np.linalg.norm(radial) == 0: radial = np.array([1., 0.])
        radial = radial / np.linalg.norm(radial); tangential = np.array([-radial[1], radial[0]])
        offset = np.stack((rows - row, cols - col), axis=-1)
        kernel = np.exp(-.5 * ((offset @ radial / sigma_radial)**2 + (offset @ tangential / sigma_tangential)**2))
        image[row_min:row_max, col_min:col_max] += intensity * kernel / kernel.sum()

    def project_points(self):
        """Project Ewald-valid reciprocal vectors and return one grain image."""
        image = np.zeros((int(self.detector_height), int(self.detector_width)))
        coordinates, two_theta_values, records = [], [], []
        center = np.asarray(self.ewald_sphere.center, dtype=float)
        scale_y, scale_z = image.shape[1] / self.detector_width, image.shape[0] / self.detector_height
        for point in self.ewald_sphere.filter_points():
            reciprocal = np.asarray(point[:3], dtype=float)
            if np.linalg.norm(reciprocal) == 0: continue
            direction = reciprocal - center
            if direction[0] == 0: continue
            projected = center + (self.detector_distance - center[0]) / direction[0] * direction
            if not (-self.detector_width / 2 <= projected[1] <= self.detector_width / 2 and
                    -self.detector_height / 2 <= projected[2] <= self.detector_height / 2): continue
            row, col = int(projected[2] * scale_z + image.shape[0] / 2), int(projected[1] * scale_y + image.shape[1] / 2)
            if not (0 <= row < image.shape[0] and 0 <= col < image.shape[1]): continue
            two_theta = np.arctan2(np.hypot(projected[1], projected[2]), self.detector_distance)
            properties = peak_properties(self.ewald_sphere.grain, reciprocal, two_theta, self.experiment.wavelength,
                                         self.detector_distance, pixel_scale=(scale_y + scale_z) / 2,
                                         shape_factor=self.shape_factor, instrumental_fwhm_px=self.instrumental_fwhm_px,
                                         intensity_scale=self.intensity_scale)
            hkl = tuple(map(int, point[3:6])) if len(point) >= 6 else None
            structure_intensity = 1.0 if self.structure_factor_func is None else abs(self.structure_factor_func(*hkl, 1))**2
            intensity = properties.integrated_intensity * structure_intensity
            radial = np.array([row - image.shape[0] / 2, col - image.shape[1] / 2])
            self._deposit_peak(image, row, col, radial, intensity, properties.fwhm_radial_px, properties.fwhm_tangential_px)
            coordinates.append(projected); two_theta_values.append(two_theta)
            records.append({"hkl": hkl, "two_theta": two_theta, "integrated_intensity": intensity,
                            "fwhm_2theta_rad": properties.fwhm_2theta_rad, "fwhm_radial_px": properties.fwhm_radial_px,
                            "fwhm_tangential_px": properties.fwhm_tangential_px, "coherent_length": properties.coherent_length,
                            "grain_volume": properties.volume})
        return {"coordinate": np.asarray(coordinates), "two_theta": np.asarray(two_theta_values), "image": image, "peaks": records}
