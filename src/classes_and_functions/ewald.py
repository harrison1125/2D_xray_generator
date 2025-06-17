# ewald.py
import numpy as np


class EwaldSphere:
    """Defines an Ewald sphere for diffraction conditions."""

    def __init__(self, grain, experiment, tolerance=0.03):
        self.radius = experiment.inv_lambda
        self.center = (-self.radius, 0, 0)
        self.tolerance = tolerance
        self.grain = grain

    def filter_points(self):
        """Filters reciprocal lattice points that satisfy diffraction conditions."""
        distances = np.linalg.norm(
            self.grain.reciprocal_lattice_vectors - self.center, axis=1
        )
        mask = (distances >= self.radius - self.tolerance) & (
            distances <= self.radius + self.tolerance
        )
        vectors = self.grain.reciprocal_lattice_vectors[mask]
        hkls = np.array(self.grain.hkl_indices)[mask]

        # Ensure both arrays are 2D
        return np.hstack((vectors, hkls))
