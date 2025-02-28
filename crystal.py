# crystal.py
import numpy as np
import math


class Grain:
    """Defines a grain with properties for XRD simulation."""

    def __init__(
        self,
        size_avg,
        size_var,
        strain_avg,
        strain_var,
        aspect_ratio,
        lattice_param,
        experiment,
    ):
        self.size_avg = size_avg
        self.size_var = size_var
        self.strain_avg = strain_avg
        self.strain_var = strain_var
        self.aspect_ratio = aspect_ratio
        self.lattice_param = lattice_param
        self.experiment = experiment
        self._initialize_reciprocal_lattice()

    def _initialize_reciprocal_lattice(self):
        """Computes reciprocal lattice vectors based on lattice parameters."""
        sphere_range = math.floor(1 / self.experiment.wavelength)
        self.hkl_indices = [
            (h, k, l)
            for h in range(-sphere_range, sphere_range + 1)
            for k in range(-sphere_range, sphere_range + 1)
            for l in range(-sphere_range, sphere_range + 1)
        ]
        self.reciprocal_lattice_vectors = np.array(
            [
                (h / self.lattice_param, k / self.lattice_param, l / self.lattice_param)
                for h, k, l in self.hkl_indices
            ]
        )

    def randomize_properties(self):
        """Randomizes grain size, strain, and rotation."""
        self.size = np.random.normal(self.size_avg, np.sqrt(self.size_var))
        self.strain = np.random.normal(self.strain_avg, np.sqrt(self.strain_var))
        self._rotate_reciprocal_lattice()

    def _rotate_reciprocal_lattice(self):
        """Applies a random rotation to the reciprocal lattice vectors."""
        theta = np.radians(np.random.uniform(0, 360))
        phi = np.radians(np.random.uniform(0, 360))

        Rz = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        )
        R = Rx @ Rz
        self.reciprocal_lattice_vectors = self.reciprocal_lattice_vectors @ R.T
