# ewald.py
import numpy as np


class EwaldSphere:
    """Defines an Ewald sphere for diffraction conditions."""

    def __init__(self, grain, experiment, tolerance):
        self.radius = experiment.inv_lambda
        self.center = (-self.radius, 0, 0)
        self.tolerance = tolerance
        self.grain = grain

    def filter_points(self):
        """Return candidate reciprocal points and their Ewald excitation error.

        ``tolerance`` is only a computational candidate window. The detector
        applies a finite-size line-profile weight to the continuous excitation
        error, so tolerance no longer acts as a flat, arbitrary diffraction
        probability within the shell.
        """
        distances = np.linalg.norm(
            self.grain.reciprocal_lattice_vectors - self.center, axis=1
        )
        mask = (distances >= self.radius - self.tolerance) & (
            distances <= self.radius + self.tolerance
        )
        vectors = self.grain.reciprocal_lattice_vectors[mask]
        hkls = np.array(self.grain.hkl_indices)[mask]
        excitation_error = np.abs(distances[mask] - self.radius)[:, None]

        # Ensure both arrays are 2D
        return np.hstack((vectors, hkls, excitation_error))
