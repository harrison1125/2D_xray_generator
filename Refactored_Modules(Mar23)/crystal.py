# crystal.py
import numpy as np
from scipy.spatial.transform import Rotation as R


class Grain:
    def __init__(self, lattice_parameters, orientation=None):
        """
        Represents a single grain in a polycrystalline material.

        :param lattice_parameters: Tuple containing the lattice constants (a, b, c, α, β, γ)
        :param orientation: 3x3 rotation matrix or Euler angles defining grain orientation
        """
        self.lattice_parameters = lattice_parameters

        # If no orientation is provided, set a random orientation
        if orientation is None:
            self.orientation = R.random().as_matrix()
        elif isinstance(orientation, (list, np.ndarray)) and len(orientation) == 3:
            self.orientation = R.from_euler(
                "xyz", orientation, degrees=True
            ).as_matrix()
        else:
            self.orientation = np.array(orientation)

    def rotate_vector(self, vector):
        """Apply the grain's orientation to a given vector."""
        return self.orientation @ np.array(vector)

    def get_reciprocal_lattice_vectors(self):
        """Computes reciprocal lattice vectors based on lattice parameters."""
        a, b, c, alpha, beta, gamma = self.lattice_parameters

        # Convert angles to radians
        alpha, beta, gamma = np.radians([alpha, beta, gamma])

        # Compute unit cell volume
        volume = (
            a
            * b
            * c
            * np.sqrt(
                1
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
        )

        # Compute reciprocal lattice vectors
        b1 = (
            np.cross(
                [b * np.cos(gamma), b * np.sin(gamma), 0],
                [c * np.cos(beta), 0, c * np.sin(beta)],
            )
            / volume
        )
        b2 = np.cross([c * np.cos(beta), 0, c * np.sin(beta)], [a, 0, 0]) / volume
        b3 = np.cross([a, 0, 0], [b * np.cos(gamma), b * np.sin(gamma), 0]) / volume

        return np.array([b1, b2, b3])

#This part of the code underneath is a little concerning for me (Harrison) right now. I am worried that it overlaps with the function fulfilled by the filter_points function under ewald. My main concerin is how the basic miller indices are considered - why (-2, 3) instead of being bound by an ewald limiting sphere, as well as deviations from values that are exactly the same for small angular shifts that necessarily might fulfill diffraction conditions by glancing off reciprocal lattice points with the ewald sphere rather than being dead on. 
'''
    def get_diffraction_spots(self, wavelength):
        """
        Simulates diffraction spots based on the reciprocal lattice and orientation.

        :param wavelength: X-ray wavelength (in same units as lattice parameters)
        :return: List of scattered wave vectors representing diffraction events
        """
        reciprocal_lattice = self.get_reciprocal_lattice_vectors()
        diffraction_spots = []

        # Consider basic Miller indices (hkl) within a range
        for h in range(-2, 3):
            for k in range(-2, 3):
                for l in range(-2, 3):
                    if (h, k, l) == (0, 0, 0):
                        continue  # Skip origin

                    # Compute reciprocal lattice vector for (hkl)
                    g_hkl = (
                        h * reciprocal_lattice[0]
                        + k * reciprocal_lattice[1]
                        + l * reciprocal_lattice[2]
                    )

                    # Apply orientation
                    g_rotated = self.rotate_vector(g_hkl)

                    # Compute diffraction condition
                    theta = np.arcsin(
                        np.linalg.norm(g_rotated) * wavelength / (4 * np.pi)
                    )

                    if np.isfinite(theta):  # Valid Bragg condition
                        diffraction_spots.append(g_rotated)

        return diffraction_spots
'''